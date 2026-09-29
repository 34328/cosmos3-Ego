#!/usr/bin/env python3
r"""Project GT and predicted hand actions onto GT and generated video frames.

Both hands are decoded from the 57D actions into the frame-0 head frame F0 and
projected with the episode's ``intrinsics.front_1`` through the **ground-truth**
camera pose of every frame (predicted camera poses are ignored). Output is a
side-by-side video: left = GT video, right = generated video; on both panels
green = GT hand, red = predicted hand.

    PYTHONPATH=$PWD:$PWD/packages/cosmos3 python -m cosmos3_joint_video_hand_pose.src.ar_overlay \
        --eval-dir outputs/joint_video_hand_pose/ar/eval/iter1200_r2_train [--episodes-manifest ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import zarr

from .action import Action57Builder, pose_matrices

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_ROOT = PROJECT_ROOT / "artifacts/cosmos3_action_contract"
TRAIN_EPISODES = PROJECT_ROOT / "artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/episodes.csv"
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
GT_COLOR, PRED_COLOR = (0, 220, 0), (255, 40, 40)  # RGB


def project(points_f0: np.ndarray, camera_f0: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[T,N,3]`` points in F0, ``[T,4,4]`` camera-to-F0 -> pixel ``[T,N,2]`` and in-front mask ``[T,N]``."""
    world_to_camera = np.linalg.inv(camera_f0)
    cam = np.einsum("tij,tnj->tni", world_to_camera[:, :3, :3], points_f0) + world_to_camera[:, None, :3, 3]
    uvw = cam @ intrinsics[:, :3].T
    depth = uvw[..., 2]
    uv = uvw[..., :2] / np.where(np.abs(depth) > 1e-9, depth, 1e-9)[..., None]
    return uv, depth > 1e-3


def draw_hand(frame: np.ndarray, uv: np.ndarray, valid: np.ndarray, color, thickness: int) -> None:
    for a, b in BONES:
        if valid[a] and valid[b] and np.all(np.abs(uv[[a, b]]) < 1e4):
            cv2.line(frame, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), color,
                     thickness, cv2.LINE_AA)
    for j in range(len(uv)):
        if valid[j] and np.all(np.abs(uv[j]) < 1e4):
            cv2.circle(frame, tuple(np.round(uv[j]).astype(int)), thickness + 1, color, -1, cv2.LINE_AA)


def load_episodes(paths: list[Path]) -> dict:
    episodes = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            episodes.update({row["episode_hash"]: row for row in csv.DictReader(handle)})
    return episodes


def overlay_sample(npz_path: Path, sample_id: str, episodes: dict, builder: Action57Builder, out_path: Path,
                   fps: float, label: str) -> dict:
    data = np.load(npz_path)
    episode_hash = sample_id.split(":")[0]
    group = zarr.open_group(episodes[episode_hash]["abs_zarr_path"], mode="r")
    intrinsics = np.asarray(group.attrs["intrinsics"]["front_1"], dtype=np.float64)
    video_idx = data["source_frame_indices"].astype(np.int64)  # [T]
    first = int(video_idx[0])
    rows = video_idx - first  # action row of every video frame
    head = pose_matrices(np.asarray(group["obs_head_pose"][first : int(video_idx[-1]) + 1]))  # [span,4,4]
    camera_f0 = np.linalg.inv(head[0]) @ head[rows]  # GT camera pose of each video frame in F0

    decoded = {"gt": builder.decode(torch.from_numpy(data["ref_action57"])),
               "pred": builder.decode(torch.from_numpy(data["pred_action57"]))}
    uv, valid = {}, {}
    for key, dec in decoded.items():
        for side in ("right", "left"):
            pts = getattr(dec, f"{side}_keypoints_f0").numpy().astype(np.float64)[rows]
            uv[key, side], valid[key, side] = project(pts, camera_f0, intrinsics)

    # Sanity: GT action re-projection must match projecting the raw zarr keypoints.
    raw_kp = np.asarray(group["right.obs_keypoints"][video_idx], dtype=np.float64).reshape(-1, 21, 3)
    raw_uv, _ = project(np.einsum("ij,tnj->tni", np.linalg.inv(head[0])[:3, :3], raw_kp)
                        + np.linalg.inv(head[0])[:3, 3], camera_f0, intrinsics)
    reproj_px = float(np.median(np.linalg.norm(raw_uv - uv["gt", "right"], axis=-1)))

    from .dataset import decode_rgb_video

    gt_video = decode_rgb_video(group["images.front_1"][video_idx])[:, :, :360].permute(1, 2, 3, 0).numpy()
    pred_video = data["pred_video"][:, :360]
    frames, errors = [], []
    for t in range(len(video_idx)):
        panels = []
        for name, base in (("GT video", gt_video[t]), ("generated video", pred_video[t])):
            frame = np.ascontiguousarray(base.copy())
            for side in ("right", "left"):
                draw_hand(frame, uv["gt", side][t], valid["gt", side][t], GT_COLOR, 2)
                if t > 0:
                    draw_hand(frame, uv["pred", side][t], valid["pred", side][t], PRED_COLOR, 1)
            cv2.putText(frame, f"{name} | {label} | t={t}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(frame)
        canvas = np.concatenate(panels, axis=1)
        cv2.putText(canvas, "green = GT hand   red = predicted hand (GT camera pose)", (8, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(canvas)
        if t > 0:
            errors.append(float(np.mean([np.linalg.norm(uv["pred", s][t] - uv["gt", s][t], axis=-1).mean()
                                         for s in ("right", "left")])))
    import imageio.v2 as imageio

    imageio.mimwrite(out_path, frames, format="FFMPEG", fps=fps, codec="libx264", pixelformat="yuv420p",
                     macro_block_size=1, ffmpeg_log_level="error")
    return {"video": str(out_path), "gt_reprojection_vs_raw_px_median": reproj_px,
            "pred_vs_gt_hand_px_mean": float(np.mean(errors)), "pred_vs_gt_hand_px_last": errors[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, required=True, action="append")
    parser.add_argument("--episodes-manifest", type=Path, action="append", default=[TRAIN_EPISODES])
    parser.add_argument("--fps", type=float, default=15.0, help="playback fps (15 = real time for stride 2)")
    args = parser.parse_args()
    builder = Action57Builder(
        state_normalizer=ACTION_ROOT / "v2/normalizers/state_normalizer.json",
        future_normalizer=ACTION_ROOT / "v4_frame_delta_30hz/normalizers/future_frame_delta_normalizer.json",
        rigid_pose_frame_delta=True,
    )
    episodes = load_episodes(args.episodes_manifest)
    for eval_dir in args.eval_dir:
        results = sorted(eval_dir.glob("results_*.json"))
        samples = json.loads(results[-1].read_text())["samples"] if results else []
        ids = {(s["index"], s["history"]): s["sample_id"] for s in samples}
        if not ids:  # still running: recover ids from the log
            for line in (eval_dir / "run.log").read_text(errors="ignore").splitlines():
                if line.startswith("AR_SAMPLE "):
                    s = json.loads(line[len("AR_SAMPLE "):])
                    ids[s["index"], s["history"]] = s["sample_id"]
        report = []
        for (index, history), sample_id in sorted(ids.items()):
            npz = eval_dir / f"{index:04d}_{history}.npz"
            out = eval_dir / f"{index:04d}_{history}_overlay.mp4"
            record = overlay_sample(npz, sample_id, episodes, builder, out, args.fps, f"{eval_dir.name} {history}")
            record.update(index=index, history=history, sample_id=sample_id)
            report.append(record)
            print("OVERLAY " + json.dumps(record), flush=True)
        (eval_dir / "overlay_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit("Pre-V0.2 experiment CLI retired. Use ar_v02_eval / ar_v02_overlay; shared helpers remain for V0.2.")
