"""Chunk-camera decoding and full-rate metrics for joint_chunk_cond_v1."""

import torch

from .ar_chunk_state import decode_chunk_camera_action, decode_chunk_camera_state
from .ar_v02_layout import ACTION, STATE, LAYOUT_VERSION
from .ar_v02_inference import HISTORY_MODES


def _anchor(encoded, normalizer, source):
    if torch.count_nonzero(encoded[:9]):
        raise ValueError("chunk-camera state camera slots must be zero")
    state = decode_chunk_camera_state(encoded, normalizer, source_index=source)
    torch.testing.assert_close(state.rigid_camera[0], torch.eye(4, device=encoded.device), atol=1e-5, rtol=1e-4)
    return state


@torch.no_grad()
def decode_joint_actions(
    layout, predicted_payload, gt_future, boundary_states, *, state_normalizer, future_normalizer, history="gt"
):
    """Yield per-chunk poses in the GT chunk camera frame, with global poses separately.

    For generated rollout only, propagate the predicted camera between chunks.
    GT is used to express evaluation coordinates, never to reset generated state.
    """
    if history not in HISTORY_MODES:
        raise ValueError("unsupported history mode")
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
        gt_anchor = _anchor(boundary_states[b.latent_start - 1], state_normalizer, b.source_start)
        if history == "generated":
            state_rows = predicted_payload[(roles == STATE) & (chunks == b.chunk_id)]
            pred_anchor = (
                _anchor(state_rows[0], state_normalizer, b.source_start)
                if len(state_rows)
                else (next_state or gt_anchor)
            )
        else:
            pred_anchor = gt_anchor
            pred_camera = gt_camera
        p = decode_chunk_camera_action(pred_anchor, predicted_future[rows], future_normalizer)
        g = decode_chunk_camera_action(gt_anchor, gt_future[rows], future_normalizer)
        p_global, g_global = pred_camera @ p.rigid_chunk, gt_camera @ g.rigid_chunk
        # Both camera and wrist poses must be transformed together for projection.
        p_in_gt_chunk = torch.linalg.inv(gt_camera) @ p_global
        records.append(
            dict(
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


def hand_keypoints(rigid, hand_latents, codecs):
    """Decode the two frozen hand AEs into 21 points in the rigid pose frame."""
    points = []
    for side, codec in enumerate(codecs):
        local = codec.decode(hand_latents[:, side]).to(rigid)
        if local.shape != (len(rigid), 20, 3):
            raise ValueError("hand codec must decode [T,20,3]")
        local = torch.cat((local.new_zeros(len(rigid), 1, 3), local), dim=1)
        wrist = rigid[:, side + 1]
        points.append(torch.einsum("tij,tnj->tni", wrist[:, :3, :3], local) + wrist[:, None, :3, 3])
    return torch.stack(points, dim=1)


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
):
    decoded, source_indexes = decode_joint_actions(
        layout,
        predicted_payload,
        gt_future,
        boundary_states,
        state_normalizer=state_normalizer,
        future_normalizer=future_normalizer,
        history=history,
    )
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
        if hand_codecs is not None:
            pp = hand_keypoints(p, item["predicted"].hand_latents, hand_codecs)
            gp = hand_keypoints(g, item["gt"].hand_latents, hand_codecs)
            for i, side in enumerate(("right", "left")):
                record[f"local_{side}_mpjpe_mm"] = float((pp[:, i] - gp[:, i]).norm(dim=-1).mean() * 1000)
                shape_error = (pp[:, i] - pp[:, i, :1]) - (gp[:, i] - gp[:, i, :1])
                record[f"local_{side}_wrist_centered_mpjpe_mm"] = float(shape_error.norm(dim=-1).mean() * 1000)
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
        hand_metrics_available=hand_codecs is not None,
        source_indexes=source_indexes.cpu().tolist(),
        chunks=records,
    )
