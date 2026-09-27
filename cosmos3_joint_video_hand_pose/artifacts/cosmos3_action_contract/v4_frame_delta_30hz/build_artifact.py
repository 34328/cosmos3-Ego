#!/usr/bin/env python3
"""Fit the B3 future rigid-frame-delta normalizer at the native 30 Hz action rate (AR v0.1).

v3 was fitted on uniformly subsampled clips. The AR contract keeps every source
frame, so each future action is the one-frame SE(3) increment T[t-1]^-1 @ T[t] at
30 Hz. Statistics use every consecutive pair inside every train segment of the
36-episode subset; the frame-0 state keeps the v2 state normalizer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import zarr

from cosmos3_joint_video_hand_pose.src.action import pose27_from_streams

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SUBSET = ROOT / "artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1"
TRANSLATION_CHANNELS = np.array([0, 1, 2, 9, 10, 11, 18, 19, 20])


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        name: np.quantile(values, quantile, axis=0).astype(float).tolist()
        for name, quantile in (("p01", 0.01), ("p50", 0.50), ("p99", 0.99))
    } | {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
    }


def main() -> None:
    normalizer_path = OUT / "normalizers/future_frame_delta_normalizer.json"
    report_path = OUT / "normalizer_report.json"
    if normalizer_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite B3 artifact under {OUT}")
    with (SUBSET / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        episodes = {row["episode_hash"]: row for row in csv.DictReader(handle) if row["split"] == "train"}
    with (SUBSET / "segments.csv").open(newline="", encoding="utf-8") as handle:
        segments = [row for row in csv.DictReader(handle) if row["split"] == "train" and row["episode_hash"] in episodes]
    futures: list[np.ndarray] = []
    fps_values = set()
    for row in segments:
        episode = episodes[row["episode_hash"]]
        fps_values.add(float(episode["fps"]))
        start, end = int(row["start_idx"]), int(row["end_idx"])
        group = zarr.open_group(episode["abs_zarr_path"], mode="r")
        end = min(end, int(group.attrs["total_frames"]))
        if end - start < 2:
            continue
        indexes = np.arange(start, end)
        pose27 = pose27_from_streams(
            np.asarray(group["obs_head_pose"][indexes]),
            np.asarray(group["right.obs_wrist_pose"][indexes]),
            np.asarray(group["left.obs_wrist_pose"][indexes]),
            rigid_pose_frame_delta=True,
        )
        futures.append(pose27[1:].astype(np.float64, copy=False))
    future = np.concatenate(futures, axis=0)
    if future.ndim != 2 or future.shape[1] != 27 or not np.isfinite(future).all():
        raise RuntimeError(f"invalid B3 future statistics: {future.shape}")

    q01 = np.quantile(future, 0.01, axis=0)
    q99 = np.quantile(future, 0.99, axis=0)
    center = (q01 + q99) / 2
    scale = np.maximum((q99 - q01) / 2, 1e-8)
    std = np.maximum(future.std(axis=0), 1e-8)
    center[TRANSLATION_CHANNELS] = 0.0
    scale[TRANSLATION_CHANNELS] = std[TRANSLATION_CHANNELS]
    z = (future - center) / scale
    normalized = np.where(np.abs(z) <= 1, z, np.sign(z) * (1 + np.arcsinh(np.abs(z) - 1)))
    payload = {
        "method": "piecewise_asinh_rot",
        "beta": 1.0,
        "fit_split": "train_only",
        "stats": {"center": center.tolist(), "scale": scale.tolist()},
        "b3_contract": {
            "a0": "unchanged_state_normalizer_v2",
            "future_camera_right_left": "T[t-1]^-1 @ T[t] at the native source frame rate",
            "translation_center": "zero",
            "translation_scale": "train_only_std",
            "rotation_center_scale": "train_only_q01_q99",
            "clamp_after_normalization": False,
        },
    }
    report = {
        "schema_version": 1,
        "artifact_id": "egoverse_action_future_frame_delta_b3_30hz_ar_v0_1",
        "fit_split": "train_only",
        "fit_subset": SUBSET.relative_to(ROOT.parent).as_posix(),
        "fit_segments": len(futures),
        "fit_tokens": len(future),
        "source_fps": sorted(fps_values),
        "future_physical": summary(future),
        "normalized": summary(normalized),
        "all_channel_tail_fraction": float((np.abs(z) > 1).any(axis=1).mean()),
    }
    normalizer_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(normalizer_path, payload)
    atomic_json(report_path, report)
    print(json.dumps({"fit_segments": len(futures), "fit_tokens": len(future), "normalizer": str(normalizer_path)}, indent=2))


if __name__ == "__main__":
    main()
