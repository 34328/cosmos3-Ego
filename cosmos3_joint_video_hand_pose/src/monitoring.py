"""Prepare deterministic EgoVerse inference samples and render prediction replays."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import zarr

from .action import (
    DEFAULT_FUTURE_NORMALIZER,
    DEFAULT_STATE_NORMALIZER,
    Action57Builder,
    DecodedAction57,
)
from .dataset import (
    PROMPT_MODES,
    PROMPT_MODE_SEGMENT_ONLY,
    CosmosActionPromptFormatter,
    build_prompt_text,
)


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
HAND_COLORS = {"right": (255, 145, 90), "left": (55, 235, 190)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _segment_frames(row: dict[str, str]) -> int:
    return int(row["end_idx"]) - int(row["start_idx"])


def select_monitor_rows(
    rows: list[dict[str, str]],
    *,
    split: str,
    min_frames: int,
    max_frames: int,
    count: int,
    seed: int | None = None,
) -> list[dict[str, str]]:
    """Select complete, natively 4n+1-aligned segments of varied tasks and lengths."""
    if min_frames > max_frames:
        raise ValueError(f"min_frames {min_frames} exceeds max_frames {max_frames}")
    candidates = [
        row
        for row in rows
        if row["split"] == split
        and min_frames <= _segment_frames(row) <= max_frames
        and (_segment_frames(row) - 1) % 4 == 0
    ]
    candidates.sort(key=lambda row: (row["manifest_task"], row["episode_hash"], int(row["span_index"])))
    if seed is not None:
        random.Random(seed).shuffle(candidates)
        selected = []
        seen_episodes = set()
        for row in candidates:
            if row["episode_hash"] not in seen_episodes:
                selected.append(row)
                seen_episodes.add(row["episode_hash"])
                if len(selected) == count:
                    return selected
        for row in candidates:
            if row not in selected:
                selected.append(row)
                if len(selected) == count:
                    return selected
        raise ValueError(
            f"only found {len(selected)} complete 4n+1-aligned {split} segments "
            f"in [{min_frames}, {max_frames}] frames"
        )

    selected: list[dict[str, str]] = []
    seen_tasks: set[str] = set()
    seen_lengths: set[int] = set()
    for row in candidates:
        frames = _segment_frames(row)
        if row["manifest_task"] not in seen_tasks and frames not in seen_lengths:
            selected.append(row)
            seen_tasks.add(row["manifest_task"])
            seen_lengths.add(frames)
            if len(selected) == count:
                return selected
    for row in candidates:
        if row not in selected and row["manifest_task"] not in seen_tasks:
            selected.append(row)
            seen_tasks.add(row["manifest_task"])
            if len(selected) == count:
                return selected
    for row in candidates:
        if row not in selected:
            selected.append(row)
            if len(selected) == count:
                return selected
    raise ValueError(
        f"only found {len(selected)} complete 4n+1-aligned {split} segments "
        f"in [{min_frames}, {max_frames}] frames"
    )


def _jpeg_bytes(value: Any) -> bytes:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"invalid JPEG payload type {type(value).__name__}")
    return bytes(value)


def _prepare_one(
    *,
    row: dict[str, str],
    episode: dict[str, str],
    split: str,
    ordinal: int,
    output_dir: Path,
    action_builder: Action57Builder,
    prompt_mode: str,
) -> Path:
    name = f"{split}_{ordinal:02d}_{row['episode_hash']}_s{int(row['span_index']):03d}"
    sample_dir = output_dir / "inputs" / name
    sample_dir.mkdir(parents=True, exist_ok=False)
    start = int(row["start_idx"])
    end = int(row["end_idx"])
    frames = end - start
    if (frames - 1) % 4:
        raise ValueError(f"monitor segment must satisfy T=1+4n without trimming, got {frames}")
    indices = np.arange(start, end, dtype=np.int64)
    group = zarr.open_group(episode["abs_zarr_path"], mode="r")

    def read(key: str) -> np.ndarray:
        return np.asarray(group[key][indices], dtype=np.float64)

    action = action_builder.build(
        head_pose=read("obs_head_pose"),
        right_wrist_pose=read("right.obs_wrist_pose"),
        left_wrist_pose=read("left.obs_wrist_pose"),
        right_keypoints=read("right.obs_keypoints"),
        left_keypoints=read("left.obs_keypoints"),
    )
    first_frame = Image.open(BytesIO(_jpeg_bytes(group["images.front_1"][start]))).convert("RGB")
    if first_frame.size != (640, 360):
        raise ValueError(f"unexpected source resolution {first_frame.size}")
    image_path = sample_dir / "first_frame.png"
    first_frame.save(image_path)
    initial_action_path = sample_dir / "initial_action57.json"
    initial_action_path.write_text(json.dumps(action[0].tolist()) + "\n", encoding="utf-8")
    reference_action_path = sample_dir / "reference_action57.json"
    reference_action_path.write_text(json.dumps(action.tolist()) + "\n", encoding="utf-8")
    reference_video_path = sample_dir / "reference_video.mp4"
    reference_frames = [
        np.asarray(Image.open(BytesIO(_jpeg_bytes(group["images.front_1"][index]))).convert("RGB"))
        for index in indices
    ]
    imageio.mimwrite(
        reference_video_path,
        reference_frames,
        format="FFMPEG",
        fps=float(episode["fps"]),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"],
    )

    fps = int(round(float(episode["fps"])))
    prompt = build_prompt_text(
        row["text_normalized"],
        episode["task_description"],
        mode=prompt_mode,
    )
    structured_prompt = json.loads(CosmosActionPromptFormatter()(prompt, frames, fps))
    intrinsics = np.asarray(group.attrs["intrinsics"]["front_1"], dtype=np.float64)
    metadata = {
        "name": name,
        "split": split,
        "episode_hash": row["episode_hash"],
        "span_index": int(row["span_index"]),
        "sample_id": f"{row['episode_hash']}:{row['span_index']}:{row['start_idx']}:{row['end_idx']}",
        "task": row["manifest_task"],
        "task_description": episode["task_description"].strip(),
        "segment_prompt": row["text_normalized"].strip(),
        "prompt_mode": prompt_mode,
        "prompt": prompt,
        "structured_prompt": structured_prompt,
        "source_indices": indices.tolist(),
        "fps": fps,
        "intrinsics_front_1": intrinsics.tolist(),
        "source_resolution": [640, 360],
        "model_canvas": [640, 368],
        "bottom_padding_pixels": 8,
        "rigid_pose_frame_delta": action_builder.rigid_pose_frame_delta,
        "initial_action_path": str(initial_action_path.resolve()),
        "reference_action_path": str(reference_action_path.resolve()),
        "reference_video_path": str(reference_video_path.resolve()),
        "action_normalizers": {
            "state": {
                "path": str(action_builder.state_normalizer.path),
                "sha256": _sha256(action_builder.state_normalizer.path),
            },
            "future": {
                "path": str(action_builder.future_normalizer.path),
                "sha256": _sha256(action_builder.future_normalizer.path),
            },
        },
    }
    metadata_path = sample_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    inference_input = {
        "name": name,
        "model_mode": "wam",
        # Keep inference byte-compatible with the training adapter.  The
        # dataset trains on Cosmos' structured action JSON, not the compact
        # human-readable episode/segment string stored in ``prompt``.
        "prompt": json.dumps(structured_prompt, separators=(",", ":")),
        "vision_path": str(image_path.resolve()),
        "action_path": str(initial_action_path.resolve()),
        "domain_name": "hand_pose",
        "raw_action_dim": 57,
        "view_point": "ego_view",
        "fps": fps,
        "num_frames": frames,
        "action_chunk_size": frames,
        "seed": 1000 + ordinal + (0 if split == "train" else 100),
    }
    input_path = output_dir / "inference_inputs" / f"{name}.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(inference_input, indent=2) + "\n", encoding="utf-8")
    return input_path


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"monitor output already exists: {args.output}")
    args.output.mkdir(parents=True)
    train_episodes = {row["episode_hash"]: row for row in _read_csv(args.train_episodes)}
    train_rows = select_monitor_rows(
        _read_csv(args.train_segments),
        split="train",
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        count=args.count,
        seed=args.selection_seed,
    )
    split_rows = [("train", train_rows, train_episodes)]
    if not args.train_only:
        if args.test_episodes is None or args.test_segments is None:
            raise ValueError("--test-episodes and --test-segments are required unless --train-only is set")
        test_episodes = {row["episode_hash"]: row for row in _read_csv(args.test_episodes)}
        test_rows = select_monitor_rows(
            _read_csv(args.test_segments),
            split="test",
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            count=args.count,
            seed=None if args.selection_seed is None else args.selection_seed + 1,
        )
        split_rows.append(("test", test_rows, test_episodes))
    builder = Action57Builder(
        state_normalizer=args.state_normalizer,
        future_normalizer=args.future_normalizer,
        rigid_pose_frame_delta=args.rigid_pose_frame_delta,
    )
    inputs = []
    for split, rows, episodes in split_rows:
        for ordinal, row in enumerate(rows):
            inputs.append(
                _prepare_one(
                    row=row,
                    episode=episodes[row["episode_hash"]],
                    split=split,
                    ordinal=ordinal,
                    output_dir=args.output,
                    action_builder=builder,
                    prompt_mode=args.prompt_mode,
                )
            )
    manifest = {
        "selection": {
            "use_complete_segment": True,
            "require_native_4n_plus_1": True,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "count_per_split": args.count,
            "train_only": args.train_only,
            "selection_seed": args.selection_seed,
            "prompt_mode": args.prompt_mode,
        },
        "samples": [
            {
                "input": str(path),
                "frames": json.loads(path.read_text(encoding="utf-8"))["num_frames"],
            }
            for path in inputs
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\n".join(str(path) for path in inputs))


def project_f0_to_pixels(
    points_f0: np.ndarray, headcam_f0: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotation = headcam_f0[:3, :3]
    translation = headcam_f0[:3, 3]
    points_camera = (points_f0 - translation) @ rotation
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-6)
    pixels = np.full((len(points_f0), 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = intrinsics[0, 0] * points_camera[valid, 0] / points_camera[valid, 2] + intrinsics[0, 2]
    pixels[valid, 1] = intrinsics[1, 1] * points_camera[valid, 1] / points_camera[valid, 2] + intrinsics[1, 2]
    return pixels, valid


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _draw_hand_2d(draw: ImageDraw.ImageDraw, pixels: np.ndarray, valid: np.ndarray, color: tuple[int, int, int]) -> None:
    in_frame = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < 640) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 360)
    for a, b in HAND_EDGES:
        if in_frame[a] and in_frame[b]:
            draw.line((*pixels[a], *pixels[b]), fill=color, width=2)
    for x, y in pixels[in_frame]:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline=(10, 10, 10))


def _third_person_mapping(*decoded_items: DecodedAction57):
    point_groups = []
    for decoded in decoded_items:
        point_groups.extend(
            (
                decoded.right_keypoints_f0.numpy().reshape(-1, 3),
                decoded.left_keypoints_f0.numpy().reshape(-1, 3),
                decoded.headcam_f0[:, :3, 3].numpy(),
            )
        )
    points = np.concatenate(point_groups, axis=0)
    u = points[:, 0] - 0.65 * points[:, 1]
    v = points[:, 2] + 0.15 * points[:, 1]
    u0, u1 = np.nanpercentile(u, [1, 99])
    v0, v1 = np.nanpercentile(v, [1, 99])
    margin_u = max((u1 - u0) * 0.08, 0.05)
    margin_v = max((v1 - v0) * 0.08, 0.05)
    u0, u1 = u0 - margin_u, u1 + margin_u
    v0, v1 = v0 - margin_v, v1 + margin_v
    scale = min(600.0 / max(u1 - u0, 1e-3), 310.0 / max(v1 - v0, 1e-3))

    def xy(point: np.ndarray) -> tuple[float, float]:
        pu = point[0] - 0.65 * point[1]
        pv = point[2] + 0.15 * point[1]
        return ((pu - u0) * scale + 20, 340 - (pv - v0) * scale)

    return xy


def _wrap_prompt(prompt: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    normalized = " ".join(prompt.split())
    if not normalized:
        return ["(empty instruction)"]

    # Greedily fit by rendered pixel width instead of character count.  This
    # also handles Chinese text and a single overlong token without spaces.
    lines: list[str] = []
    current = ""
    for character in normalized:
        candidate = current + character
        if not current or font.getlength(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current.rstrip())
            current = character.lstrip()
    if current:
        lines.append(current.rstrip())
    return lines


def _draw_pose_row(
    *,
    frame_rgb: np.ndarray,
    decoded: DecodedAction57,
    frame_index: int,
    intrinsics: np.ndarray,
    xy,
    left_title: str,
    right_title: str,
) -> Image.Image:
    left_panel = Image.new("RGB", (640, 360), (0, 0, 0))
    left_panel.paste(Image.fromarray(frame_rgb[:360, :640]).convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(left_panel)
    headcam = decoded.headcam_f0[frame_index].numpy()
    for side, points in (
        ("right", decoded.right_keypoints_f0[frame_index].numpy()),
        ("left", decoded.left_keypoints_f0[frame_index].numpy()),
    ):
        pixels, valid = project_f0_to_pixels(points, headcam, intrinsics)
        _draw_hand_2d(draw, pixels, valid, HAND_COLORS[side])
    draw.rectangle((0, 0, 640, 27), fill=(5, 12, 18))
    draw.text((8, 5), f"{left_title} | frame {frame_index}", fill="white", font=_font(14))

    right_panel = Image.new("RGB", (640, 360), (238, 241, 244))
    draw3d = ImageDraw.Draw(right_panel)
    for side, points in (
        ("right", decoded.right_keypoints_f0[frame_index].numpy()),
        ("left", decoded.left_keypoints_f0[frame_index].numpy()),
    ):
        for a, b in HAND_EDGES:
            draw3d.line((*xy(points[a]), *xy(points[b])), fill=HAND_COLORS[side], width=3)
        for point in points:
            x, y = xy(point)
            draw3d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=HAND_COLORS[side], outline=(20, 20, 20))
    origin = headcam[:3, 3]
    rotation = headcam[:3, :3]
    for axis, color in zip(range(3), ((220, 60, 60), (60, 160, 70), (60, 90, 220)), strict=True):
        endpoint = origin + rotation[:, axis] * 0.15
        draw3d.line((*xy(origin), *xy(endpoint)), fill=color, width=3)
    ox, oy = xy(origin)
    draw3d.ellipse((ox - 5, oy - 5, ox + 5, oy + 5), fill=(40, 40, 40))
    draw3d.rectangle((0, 0, 640, 27), fill=(230, 233, 238))
    draw3d.text((8, 5), right_title, fill=(15, 20, 25), font=_font(14))

    row = Image.new("RGB", (1280, 360))
    row.paste(left_panel, (0, 0))
    row.paste(right_panel, (640, 0))
    return row


def _load_prediction(path: Path) -> torch.Tensor:
    payload = json.loads(path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs") or []
    if not outputs or "action" not in outputs[0].get("content", {}):
        raise ValueError(f"no predicted action in {path}")
    action = torch.tensor(outputs[0]["content"]["action"], dtype=torch.float32)
    if action.ndim != 2 or action.shape[1] != 57:
        raise ValueError(f"predicted action must be [T,57], got {tuple(action.shape)}")
    return action


def render(args: argparse.Namespace) -> None:
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    normalizer_metadata = metadata.get("action_normalizers", {})
    state_normalizer = args.state_normalizer or normalizer_metadata.get("state", {}).get("path")
    future_normalizer = args.future_normalizer or normalizer_metadata.get("future", {}).get("path")
    state_normalizer = Path(state_normalizer or DEFAULT_STATE_NORMALIZER)
    future_normalizer = Path(future_normalizer or DEFAULT_FUTURE_NORMALIZER)
    builder = Action57Builder(
        state_normalizer=state_normalizer,
        future_normalizer=future_normalizer,
        rigid_pose_frame_delta=bool(metadata.get("rigid_pose_frame_delta", False)),
    )
    decoded = builder.decode(_load_prediction(args.sample_outputs))
    reference_action = torch.tensor(
        json.loads(Path(metadata["reference_action_path"]).read_text(encoding="utf-8")), dtype=torch.float32
    )
    reference_decoded = builder.decode(reference_action)
    intrinsics = np.asarray(metadata["intrinsics_front_1"], dtype=np.float64)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"could not open generated video {args.video}")
    reference_capture = cv2.VideoCapture(metadata["reference_video_path"])
    if not reference_capture.isOpened():
        raise ValueError(f"could not open reference video {metadata['reference_video_path']}")
    output_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(metadata["fps"])
    title_font = _font(14)
    prompt_lines = _wrap_prompt(metadata["prompt"], title_font, 1148)
    banner_height = max(36, 10 + 19 * len(prompt_lines))
    if banner_height % 2:
        banner_height += 1
    output_height = banner_height + 720
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.output),
        format="FFMPEG",
        mode="I",
        fps=output_fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"],
    )
    xy = _third_person_mapping(decoded, reference_decoded)
    frame_index = 0
    try:
        while frame_index < len(decoded.headcam_f0):
            ok, frame_bgr = capture.read()
            reference_ok, reference_bgr = reference_capture.read()
            if not ok or not reference_ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)[:360, :640]
            reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)[:360, :640]
            prediction_row = _draw_pose_row(
                frame_rgb=rgb,
                decoded=decoded,
                frame_index=frame_index,
                intrinsics=intrinsics,
                xy=xy,
                left_title="PRED generated + predicted hand projection",
                right_title="PRED F0 third-person | headcam axes RGB",
            )
            reference_row = _draw_pose_row(
                frame_rgb=reference_rgb,
                decoded=reference_decoded,
                frame_index=frame_index,
                intrinsics=intrinsics,
                xy=xy,
                left_title="GT source video + decoded target projection",
                right_title="GT decoded target in F0 | headcam axes RGB",
            )
            panel = Image.new("RGB", (1280, output_height), (0, 0, 0))
            title_draw = ImageDraw.Draw(panel)
            title_draw.text((16, 5), "Instruction:", fill="white", font=title_font)
            for line_index, line in enumerate(prompt_lines):
                title_draw.text((112, 5 + 19 * line_index), line, fill="white", font=title_font)
            panel.paste(prediction_row, (0, banner_height))
            panel.paste(reference_row, (0, banner_height + 360))
            writer.append_data(np.asarray(panel))
            frame_index += 1
    finally:
        capture.release()
        reference_capture.release()
        writer.close()
    if frame_index != len(decoded.headcam_f0) or frame_index != len(reference_decoded.headcam_f0):
        raise ValueError(
            f"video/action length mismatch: rendered {frame_index}, "
            f"prediction {len(decoded.headcam_f0)}, reference {len(reference_decoded.headcam_f0)}"
        )
    render_metadata = {
        **metadata,
        "generated_video": str(args.video),
        "sample_outputs": str(args.sample_outputs),
        "rendered_frames": frame_index,
        "replay_layout": [
            ["prediction_video_projection", "prediction_f0_third_person"],
            ["ground_truth_video_projection", "ground_truth_f0_third_person"],
        ],
        "replay_codec": "h264",
        "replay_pixel_format": "yuv420p",
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(render_metadata, indent=2) + "\n", encoding="utf-8")
    print(args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--train-episodes", type=Path, required=True)
    prepare_parser.add_argument("--train-segments", type=Path, required=True)
    prepare_parser.add_argument("--test-episodes", type=Path)
    prepare_parser.add_argument("--test-segments", type=Path)
    prepare_parser.add_argument("--train-only", action="store_true")
    prepare_parser.add_argument("--min-frames", type=int, default=81)
    prepare_parser.add_argument("--max-frames", type=int, default=121)
    prepare_parser.add_argument("--count", type=int, default=2)
    prepare_parser.add_argument("--selection-seed", type=int)
    prepare_parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default=PROMPT_MODE_SEGMENT_ONLY)
    prepare_parser.add_argument(
        "--state-normalizer",
        type=Path,
        default=DEFAULT_STATE_NORMALIZER,
    )
    prepare_parser.add_argument(
        "--future-normalizer",
        type=Path,
        default=DEFAULT_FUTURE_NORMALIZER,
    )
    prepare_parser.add_argument("--rigid-pose-frame-delta", action="store_true")
    prepare_parser.set_defaults(func=prepare)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--video", type=Path, required=True)
    render_parser.add_argument("--sample-outputs", type=Path, required=True)
    render_parser.add_argument("--metadata", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument(
        "--state-normalizer",
        type=Path,
    )
    render_parser.add_argument(
        "--future-normalizer",
        type=Path,
    )
    render_parser.set_defaults(func=render)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
