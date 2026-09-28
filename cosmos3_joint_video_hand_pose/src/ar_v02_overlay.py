"""Source-time replay and calibrated, per-chunk hand projection for AR v0.2."""

import numpy as np
import torch

from .ar_v02_evaluation import decode_joint_actions, hand_keypoints
from .ar_v02_layout import LAYOUT_VERSION


def image_pixel_transform(*, crop_xywh, resized_wh, pad_xy=(0, 0)):
    """Original pixels -> crop -> resize (pixel centers) -> top/left padding.

    Callers must pass the actual image preprocessing geometry; dimensions alone
    cannot distinguish cropping from resizing or padding.
    """
    x, y, w, h = map(float, crop_xywh)
    rw, rh = map(float, resized_wh)
    px, py = map(float, pad_xy)
    if not np.isfinite([x, y, w, h, rw, rh, px, py]).all() or min(w, h, rw, rh) <= 0:
        raise ValueError("invalid crop/resize geometry")
    sx, sy = rw / w, rh / h
    return np.array(
        [[sx, 0, px - sx * x + (sx - 1) / 2], [0, sy, py - sy * y + (sy - 1) / 2], [0, 0, 1]], dtype=np.float64
    )


def transformed_intrinsics(intrinsics, pixel_transform):
    k, affine = np.asarray(intrinsics, dtype=np.float64), np.asarray(pixel_transform, dtype=np.float64)
    if k.shape not in ((3, 3), (3, 4)) or affine.shape != (3, 3):
        raise ValueError("intrinsics must be [3,3] or [3,4]; pixel transform must be [3,3]")
    if not np.isfinite(k).all() or not np.isfinite(affine).all():
        raise ValueError("non-finite intrinsics or pixel transform")
    if k.shape == (3, 4) and np.count_nonzero(k[:, 3]):
        raise ValueError("nonzero projection translation is not a camera intrinsic matrix")
    return affine @ k[:, :3]


def project_chunk_hands(points, camera, intrinsics):
    """[T,2,21,3] and [T,4,4] in the same chunk frame -> image pixels."""
    points, camera = np.asarray(points, dtype=np.float64), np.asarray(camera, dtype=np.float64)
    if points.ndim != 4 or points.shape[1:] != (2, 21, 3) or camera.shape != (len(points), 4, 4):
        raise ValueError("projection needs full-rate hand points and matching camera poses")
    if not np.isfinite(points).all() or not np.isfinite(camera).all():
        raise ValueError("projection requires finite decoded poses")
    inverse = np.linalg.inv(camera)
    local = np.einsum("tij,tsnj->tsni", inverse[:, :3, :3], points) + inverse[:, None, None, :3, 3]
    pixel = local @ np.asarray(intrinsics).T
    depth = local[..., 2]
    denominator = pixel[..., 2]
    valid = (depth > 1e-3) & (np.abs(denominator) > 1e-9)
    uv = pixel[..., :2] / np.where(valid, denominator, 1)[..., None]
    valid &= np.isfinite(uv).all(-1)
    return uv, valid


def replay_timeline(layout, *, source_fps=30.0, speed_factor=0.5, mode="real_time", source_offset=0):
    """Every action updates once; generated RGB uses previous sampled source time."""
    if (
        mode not in ("real_time", "model_time")
        or not np.isfinite([source_fps, speed_factor]).all()
        or min(source_fps, speed_factor) <= 0
    ):
        raise ValueError("invalid replay mode or FPS")
    source, chunks, rgb_indexes, rgb_sources, rgb_chunks = [0], [1], [0], [0], [1]
    for b in layout.boundaries:
        for t in range(b.source_start + 1, b.source_stop + 1):
            source.append(t)
            chunks.append(b.chunk_id)
            rgb_index = (t - b.source_start) // 2
            # Never replace the previous prediction at b with the next GT U.
            if rgb_index == 0 and b.chunk_id > 1:
                previous = layout.boundaries[b.chunk_id - 2]
                rgb_indexes.append(previous.action_count // 2)
                rgb_chunks.append(previous.chunk_id)
            else:
                rgb_indexes.append(rgb_index)
                rgb_chunks.append(b.chunk_id)
            rgb_sources.append(b.source_start + 2 * rgb_index)
    source, rgb_sources = np.asarray(source), np.asarray(rgb_sources)
    return dict(
        layout_version=LAYOUT_VERSION,
        source_indexes=source + source_offset,
        clip_source_indexes=source,
        source_times=(source + source_offset) / source_fps,
        action_rows=source - 1,
        chunk_ids=np.asarray(chunks),
        generated_frame_indexes=np.asarray(rgb_indexes),
        generated_chunk_ids=np.asarray(rgb_chunks),
        generated_source_indexes=rgb_sources + source_offset,
        generated_is_sampled_time=source == rgb_sources,
        held_background=source != rgb_sources,
        output_fps=source_fps * (speed_factor if mode == "model_time" else 1),
        playback_speed=speed_factor if mode == "model_time" else 1.0,
        source_fps=source_fps,
        fps_video=source_fps / 2 * speed_factor,
        fps_action=source_fps * speed_factor,
        mode=mode,
    )


@torch.no_grad()
def decode_video_chunks(model, layout, video_payload):
    """Decode every complete [U,V] block independently; retain its boundary frame."""
    if video_payload.ndim != 5 or video_payload.shape[2] != layout.num_video_frames:
        raise ValueError("video payload does not match per-chunk layout")
    frames = []
    for b in layout.boundaries:
        indexes = layout.video_indexes(b.chunk_id).to(video_payload.device)
        decoded = model.decode(video_payload.index_select(2, indexes).to(model.tensor_kwargs["dtype"]))
        if decoded.shape[2] != 1 + b.action_count // 2 or not torch.isfinite(decoded).all():
            raise ValueError("decoded chunk must have one boundary plus stride-2 future RGB frames")
        frames.append(((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy())
    return frames


@torch.no_grad()
def render_joint_overlay(
    layout,
    predicted_payload,
    gt_future,
    boundary_states,
    *,
    gt_rgb,
    generated_rgb_chunks,
    intrinsics,
    gt_pixel_transform,
    generated_pixel_transform,
    state_normalizer,
    future_normalizer,
    hand_codecs,
    history="gt",
    source_fps=30.0,
    speed_factor=0.5,
    mode="real_time",
    source_offset=0,
):
    """Return RGB dual panels and an auditable 30Hz source/frame mapping.

    Images must already have their declared crop/resize/padding applied. Left
    uses GT camera; right uses predicted camera. Only actual generated timestamps
    are eligible for image/skeleton consistency metrics, not held backgrounds.
    """
    import cv2
    from .ar_overlay import draw_hand, GT_COLOR, PRED_COLOR

    timeline = replay_timeline(
        layout, source_fps=source_fps, speed_factor=speed_factor, mode=mode, source_offset=source_offset
    )
    gt_rgb = np.asarray(gt_rgb)
    if gt_rgb.ndim != 4 or gt_rgb.shape[0] != len(timeline["source_indexes"]) or gt_rgb.shape[-1] != 3:
        raise ValueError("GT RGB must contain every source frame, including the initial boundary")
    if len(generated_rgb_chunks) != len(layout.boundaries):
        raise ValueError("one independently decoded [U,V] RGB block is required per chunk")
    for b, rgb in zip(layout.boundaries, generated_rgb_chunks):
        if rgb.shape != (1 + b.action_count // 2, *gt_rgb.shape[1:]):
            raise ValueError("generated blocks must match display size and exact source timestamps")
    kg = transformed_intrinsics(intrinsics, gt_pixel_transform)
    kp = transformed_intrinsics(intrinsics, generated_pixel_transform)
    decoded, _ = decode_joint_actions(
        layout,
        predicted_payload,
        gt_future,
        boundary_states,
        state_normalizer=state_normalizer,
        future_normalizer=future_normalizer,
        history=history,
    )
    projections = {}
    for item in decoded:
        b = item["boundary"]
        pred_points = hand_keypoints(item["predicted_rigid"], item["predicted"].hand_latents, hand_codecs).cpu().numpy()
        gt_points = hand_keypoints(item["gt_rigid"], item["gt"].hand_latents, hand_codecs).cpu().numpy()
        gt_camera = item["gt_rigid"][:, 0].cpu().numpy()
        pred_camera = item["predicted_rigid"][:, 0].cpu().numpy()
        projections[b.chunk_id] = dict(
            left_gt=project_chunk_hands(gt_points, gt_camera, kg),
            left_pred=project_chunk_hands(pred_points, gt_camera, kg),
            right_pred=project_chunk_hands(pred_points, pred_camera, kp),
        )
    first = decoded[0]
    initial_points = {}
    for role in ("gt", "predicted"):
        anchor = first[f"{role}_anchor"]
        initial_points[role] = (
            hand_keypoints(anchor.rigid_camera[None], anchor.hand_latents[None], hand_codecs).cpu().numpy()
        )
    gt_camera = first["gt_anchor"].rigid_camera[None, 0].cpu().numpy()
    pred_camera = first["predicted_anchor"].rigid_camera[None, 0].cpu().numpy()
    initial = dict(
        left_gt=project_chunk_hands(initial_points["gt"], gt_camera, kg),
        left_pred=project_chunk_hands(initial_points["predicted"], gt_camera, kg),
        right_pred=project_chunk_hands(initial_points["predicted"], pred_camera, kp),
    )
    frames = []
    for frame_index, source in enumerate(timeline["clip_source_indexes"]):
        chunk = int(timeline["chunk_ids"][frame_index])
        rgb_index = int(timeline["generated_frame_indexes"][frame_index])
        rgb_chunk = int(timeline["generated_chunk_ids"][frame_index])
        panels = [
            np.ascontiguousarray(gt_rgb[frame_index].copy()),
            np.ascontiguousarray(generated_rgb_chunks[rgb_chunk - 1][rgb_index].copy()),
        ]
        row = source - layout.boundaries[chunk - 1].source_start - 1 if source else 0
        projected = projections[chunk] if source else initial
        for panel, name in zip(panels, ("left", "right")):
            overlays = (("gt", GT_COLOR), ("pred", PRED_COLOR)) if name == "left" else (("pred", PRED_COLOR),)
            for role, color in overlays:
                uv, valid = projected[f"{name}_{role}"]
                for side in range(2):
                    draw_hand(panel, uv[row, side], valid[row, side], color, 1)
        held = bool(timeline["held_background"][frame_index])
        for panel, name in zip(panels, ("GT camera", "pred camera")):
            label = f"{name} | {history} | c{chunk} | t={timeline['source_times'][frame_index]:.3f}s | {timeline['playback_speed']:g}x"
            cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if held:
            cv2.putText(
                panels[1],
                "held RGB; current 30Hz action",
                (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        frames.append(np.concatenate(panels, axis=1))
    timeline["history"] = history
    timeline["left_overlay"] = "gt_green_and_prediction_red_with_gt_camera"
    timeline["right_overlay"] = "prediction_red_only_with_predicted_camera"
    timeline["projection_nonfinite_policy"] = "raise_data_error"
    timeline["initial_state_drawn"] = True
    timeline["consistency_eligible"] = timeline["generated_is_sampled_time"] & (timeline["action_rows"] >= 0)
    return np.stack(frames), timeline


def save_joint_overlay(path, frames, timeline):
    """Write synchronized video and JSON metadata beside it."""
    import json
    from pathlib import Path
    import imageio.v2 as imageio

    path = Path(path)
    imageio.mimwrite(
        path,
        frames,
        format="FFMPEG",
        fps=timeline["output_fps"],
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    metadata = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in timeline.items()}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path
