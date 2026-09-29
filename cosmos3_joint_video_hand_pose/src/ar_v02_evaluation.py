"""Chunk-camera decoding and full-rate metrics for joint_chunk_cond_v1."""

import torch

from .action_representation import ActionRepresentationAdapter, FIXED_CAMERA, LEGACY
from .ar_v02_layout import STATE, LAYOUT_VERSION
from .ar_v02_inference import HISTORY_MODES

# Historical display helper only; never selects the current evaluation adapter.
CAMERA_AXIS_LEGACY = "fixed_camera_delta_latent_v1"


def raw_gt_in_chunk(layout, raw_gt, *, source_offset=0, reference):
    """Align raw world/metre R,L keypoints using explicit episode source frames."""
    if raw_gt is None:
        return None
    if (raw_gt.get("coordinate_frame") != "world" or raw_gt.get("units") != "metres"
            or tuple(raw_gt.get("hand_order", ())) != ("right", "left")):
        raise ValueError("raw GT requires world coordinates, metres, and right/left order")
    n = (layout.num_frames - 1) * 8 + 1
    sources = torch.as_tensor(raw_gt["source_indexes"], device=reference.device)
    expected = torch.arange(source_offset, source_offset + n, device=reference.device)
    if sources.dtype not in (torch.int32, torch.int64) or sources.shape != (n,) or not torch.equal(sources, expected):
        raise ValueError("raw GT source frames must exactly match the dense episode window")
    points = torch.as_tensor(raw_gt["keypoints"], device=reference.device, dtype=torch.float64)
    cameras = torch.as_tensor(raw_gt["camera_poses"], device=reference.device, dtype=torch.float64)
    if points.shape != (n, 2, 21, 3) or cameras.shape != (n, 4, 4):
        raise ValueError("raw GT needs [N+1,2,21,3] points and [N+1,4,4] camera poses")
    from .action_fixed_camera import _rigid
    _rigid(cameras)
    if not torch.isfinite(points).all():
        raise ValueError("raw GT keypoints must be finite")
    result = {}
    for b in layout.boundaries:
        inverse = torch.linalg.inv(cameras[b.source_start])
        selected = points[b.source_start:b.source_stop + 1]
        result[b.chunk_id] = (torch.einsum("ij,thnj->thni", inverse[:3, :3], selected)
                              + inverse[:3, 3]).to(reference)
    return result


def _anchor(encoded, adapter, source):
    if torch.count_nonzero(encoded[:9]):
        raise ValueError("chunk-camera state camera slots must be zero")
    state = adapter.decode_state(encoded, source_index=source)
    torch.testing.assert_close(state.rigid_camera[0], torch.eye(4, device=encoded.device), atol=1e-5, rtol=1e-4)
    return state


@torch.no_grad()
def decode_joint_actions(
    layout, predicted_payload, gt_future, boundary_states, *, state_normalizer, future_normalizer, history="gt", hand_codecs=None
):
    """Yield per-chunk poses in the GT chunk camera frame, with global poses separately.

    For generated rollout only, propagate the predicted camera between chunks.
    GT is used to express evaluation coordinates, never to reset generated state.
    """
    if history not in HISTORY_MODES:
        raise ValueError("unsupported history mode")
    adapter = ActionRepresentationAdapter(state_normalizer, future_normalizer, hand_codecs)
    if adapter.representation == FIXED_CAMERA and layout.chunk_size != 4:
        raise ValueError("fixed-camera evaluation requires C=4")
    _, predicted_future, source_indexes = layout.unpack_action(predicted_payload)
    if (
        predicted_future.shape != gt_future.shape
        or predicted_future.shape[1] != 64
        or boundary_states.shape != (layout.num_frames - 1, 64)
    ):
        raise ValueError("evaluation needs full-rate future targets and explicit boundary states")
    if not torch.equal(source_indexes, torch.arange(1, len(gt_future) + 1, device=source_indexes.device)):
        raise ValueError("future sources must cover every real action once")
    roles, chunks, _ = layout.action_metadata(device=predicted_payload.device)
    gt_camera = torch.eye(4, device=predicted_payload.device)
    pred_camera = gt_camera.clone()
    next_state = None
    records = []
    for b in layout.boundaries:
        rows = slice(b.source_start, b.source_stop)
        gt_anchor = _anchor(boundary_states[b.latent_start - 1], adapter, b.source_start)
        if history == "generated":
            state_rows = predicted_payload[(roles == STATE) & (chunks == b.chunk_id)]
            # Only the first observed state seeds rollout. Later payload state rows
            # may contain GT; the integrated prediction is the sole state source.
            pred_anchor = next_state if next_state is not None else (
                _anchor(state_rows[0], adapter, b.source_start) if len(state_rows) else gt_anchor)
        else:
            pred_anchor = gt_anchor
            pred_camera = gt_camera
        p = adapter.decode_action(pred_anchor, predicted_future[rows])
        g = adapter.decode_action(gt_anchor, gt_future[rows])
        p_global, g_global = pred_camera @ p.rigid_chunk, gt_camera @ g.rigid_chunk
        # Both camera and wrist poses must be transformed together for projection.
        p_in_gt_chunk = torch.linalg.inv(gt_camera) @ p_global
        point_fields = {}
        if adapter.representation == FIXED_CAMERA:
            alignment = torch.linalg.inv(gt_camera) @ pred_camera
            point_fields = dict(
                predicted_points=torch.einsum("ij,thnj->thni", alignment[:3, :3], p.keypoints_chunk) + alignment[:3, 3],
                gt_points=g.keypoints_chunk,
            )
        records.append(
            dict(
                representation=adapter.representation,
                **point_fields,
                boundary=b,
                predicted=p,
                gt=g,
                predicted_rigid=p_in_gt_chunk,
                gt_rigid=g.rigid_chunk,
                predicted_global=p_global,
                gt_global=g_global,
                gt_anchor=gt_anchor,
                predicted_anchor=pred_anchor,
            )
        )
        if history == "generated":
            next_state = p.end_state
            pred_camera = p_global[-1, 0]
        gt_camera = g_global[-1, 0]
    return records, source_indexes


def hand_keypoints(rigid, hand_latents, codecs, *, representation=LEGACY):
    """Decode the two frozen hand AEs into 21 points in the rigid pose frame."""
    points = []
    for side, codec in enumerate(codecs):
        local = codec.decode(hand_latents[:, side]).to(rigid)
        if local.shape != (len(rigid), 20, 3):
            raise ValueError("hand codec must decode [T,20,3]")
        local = torch.cat((local.new_zeros(len(rigid), 1, 3), local), dim=1)
        wrist = rigid[:, side + 1]
        if representation == CAMERA_AXIS_LEGACY:
            # Display already-decoded historical camera-axis latents only.
            points.append(local + wrist[:, None, :3, 3])
        elif representation in (LEGACY, FIXED_CAMERA):
            points.append(torch.einsum("tij,tnj->tni", wrist[:, :3, :3], local) + wrist[:, None, :3, 3])
        else:
            raise ValueError("unknown hand coordinate representation")
    return torch.stack(points, dim=1)


def record_hand_keypoints(item, role, codecs):
    if item["representation"] == FIXED_CAMERA:
        return item[f"{role}_points"]
    return hand_keypoints(item[f"{role}_rigid"], item[role].hand_latents, codecs)


def _rigid_metrics(record, prefix, p, g):
    for i, name in enumerate(("camera", "right_wrist", "left_wrist")):
        position = (p[:, i, :3, 3] - g[:, i, :3, 3]).norm(dim=-1) * 1000
        rotation = p[:, i, :3, :3].transpose(-1, -2) @ g[:, i, :3, :3]
        angle = torch.acos(((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)) * 180 / torch.pi
        record[f"{prefix}_{name}_mean_mm"] = float(position.mean())
        record[f"{prefix}_{name}_end_mm"] = float(position[-1])
        record[f"{prefix}_{name}_end_degrees"] = float(angle[-1])


@torch.no_grad()
def evaluate_joint_actions(
    layout,
    predicted_payload,
    gt_future,
    boundary_states,
    *,
    state_normalizer,
    future_normalizer,
    history="gt",
    hand_codecs=None,
    compute_hand_metrics=True,
    raw_gt=None,
    source_offset=0,
):
    decoded, source_indexes = decode_joint_actions(
        layout,
        predicted_payload,
        gt_future,
        boundary_states,
        state_normalizer=state_normalizer,
        future_normalizer=future_normalizer,
        history=history,
        hand_codecs=hand_codecs,
    )
    raw_chunks = raw_gt_in_chunk(layout, raw_gt, source_offset=source_offset, reference=predicted_payload)
    if compute_hand_metrics and raw_chunks is None and decoded[0]["representation"] == FIXED_CAMERA:
        raise ValueError("raw GT keypoints required for wrist-local primary hand metrics")
    records = []
    for item in decoded:
        b, p, g = item["boundary"], item["predicted_rigid"], item["gt_rigid"]
        baseline = item["gt_anchor"].rigid_camera.expand_as(g)
        record = dict(
            chunk=b.chunk_id,
            source_start=b.source_start,
            source_stop=b.source_stop,
            action_count=b.action_count,
            group="chunk1" if b.chunk_id == 1 else "later_chunks",
        )
        _rigid_metrics(record, "local", p, g)
        _rigid_metrics(record, "no_motion", baseline, g)
        if history == "generated":
            _rigid_metrics(record, "cumulative", item["predicted_global"], item["gt_global"])
        for i, name in enumerate(("camera", "right_wrist", "left_wrist")):
            dp, dg = p[:, i, :3, 3] - baseline[:, i, :3, 3], g[:, i, :3, 3] - baseline[:, i, :3, 3]
            valid = (dp.norm(dim=-1) > 1e-8) & (dg.norm(dim=-1) > 1e-8)
            record[f"local_{name}_direction_valid_count"] = int(valid.sum())
            record[f"local_{name}_direction_cosine"] = (
                float(torch.nn.functional.cosine_similarity(dp[valid], dg[valid]).mean()) if valid.any() else None
            )
        if hand_codecs is not None and compute_hand_metrics:
            pp = record_hand_keypoints(item, "predicted", hand_codecs)
            gp = record_hand_keypoints(item, "gt", hand_codecs)
            target = raw_chunks[b.chunk_id][1:] if raw_chunks is not None else gp
            for i, side in enumerate(("right", "left")):
                record[f"decoded_gt_aux_{side}_mpjpe_mm"] = float((pp[:, i] - gp[:, i]).norm(dim=-1).mean() * 1000)
                record[f"local_{side}_mpjpe_mm"] = float((pp[:, i] - target[:, i]).norm(dim=-1).mean() * 1000)
                shape_error = (pp[:, i] - pp[:, i, :1]) - (target[:, i] - target[:, i, :1])
                record[f"local_{side}_wrist_centered_mpjpe_mm"] = float(shape_error.norm(dim=-1).mean() * 1000)
                # Remove each wrist's own rigid pose to isolate wrist-local shape.
                pq = torch.einsum("tji,tnj->tni", p[:, i+1, :3, :3], pp[:, i, 1:] - p[:, i+1, None, :3, 3])
                gq = torch.einsum("tji,tnj->tni", g[:, i+1, :3, :3], target[:, i, 1:] - g[:, i+1, None, :3, 3])
                record[f"local_{side}_wrist_local_shape_mpjpe_mm"] = float((pq-gq).norm(dim=-1).mean()*1000)
        records.append(record)
    # "local" is a coordinate-frame label, not a claim that history error was reset.
    # Keep existing metric keys for archive consumers, but make their scope explicit.
    local_scope = (
        "rollout_error_in_gt_chunk_camera_including_history_drift"
        if history == "generated"
        else "chunk_error_from_gt_boundary_state"
    )
    return dict(
        layout_version=LAYOUT_VERSION,
        history=history,
        metric_frame="gt_chunk_camera",
        local_metric_scope=local_scope,
        boundary_state_reset_to_gt=history != "generated",
        local_metrics_include_history_drift=history == "generated",
        cumulative_drift_reported=history == "generated",
        hand_metrics_available=hand_codecs is not None and compute_hand_metrics,
        hand_metric_target="raw_gt_keypoints" if raw_chunks is not None else "decoded_gt_action_latents",
        hand_metric_status=("not_computed" if hand_codecs is None or not compute_hand_metrics
                            else "primary" if raw_chunks is not None else "legacy_decoded_gt_diagnostic_only"),
        decoded_gt_aux_metric_target="decoded_gt_action_latents",
        hand_metrics_include_gt_codec_reconstruction_error=raw_chunks is not None,
        source_indexes=source_indexes.cpu().tolist(),
        episode_source_indexes=(source_indexes + source_offset).cpu().tolist(),
        chunks=records,
    )
