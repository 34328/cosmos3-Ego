#!/usr/bin/env python3
"""Fit the B3 future rigid-frame-delta normalizer on the 75K overfit subset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from cosmos3_joint_video_hand_pose.src.action import pose27_from_streams
from cosmos3_joint_video_hand_pose.src.dataset import EgoVerseSegmentDataset
from cosmos3_joint_video_hand_pose.src.temporal import uniform_frame_indices


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SUBSET = ROOT / "artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1"
EPISODES = SUBSET / "episodes.csv"
SEGMENTS = SUBSET / "segments.csv"
MAX_SEQUENCE_LENGTH = 75_000
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
    normalizer_dir = OUT / "normalizers"
    normalizer_path = normalizer_dir / "future_frame_delta_normalizer.json"
    report_path = OUT / "normalizer_report.json"
    if normalizer_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite B3 artifact under {OUT}")

    dataset = EgoVerseSegmentDataset(
        EPISODES,
        SEGMENTS,
        action_builder=object(),
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        prompt_mode="episode_context_and_segment",
    )
    futures: list[np.ndarray] = []
    for row in dataset.rows:
        episode = dataset.episodes[row["episode_hash"]]
        source_frames = int(row["end_idx"]) - int(row["start_idx"])
        source_indices = int(row["start_idx"]) + uniform_frame_indices(
            source_frames,
            int(row["_selected_frames"]),
        )
        group = zarr.open_group(episode["abs_zarr_path"], mode="r")
        pose27 = pose27_from_streams(
            np.asarray(group["obs_head_pose"][source_indices]),
            np.asarray(group["right.obs_wrist_pose"][source_indices]),
            np.asarray(group["left.obs_wrist_pose"][source_indices]),
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
    normalized = np.where(
        np.abs(z) <= 1,
        z,
        np.sign(z) * (1 + np.arcsinh(np.abs(z) - 1)),
    )
    payload = {
        "method": "piecewise_asinh_rot",
        "beta": 1.0,
        "fit_split": "train_only",
        "stats": {"center": center.tolist(), "scale": scale.tolist()},
        "b3_contract": {
            "a0": "unchanged_state_normalizer_v2",
            "future_camera_right_left": "T[t-1]^-1 @ T[t]",
            "translation_center": "zero",
            "translation_scale": "train_only_std",
            "rotation_center_scale": "train_only_q01_q99",
            "clamp_after_normalization": False,
        },
    }
    report = {
        "schema_version": 1,
        "artifact_id": "egoverse_action_future_frame_delta_b3_overfit_75k_v1",
        "fit_split": "train_only",
        "fit_subset": str(SUBSET),
        "fit_samples": len(dataset.rows),
        "fit_tokens": len(future),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "retention_counts": dataset.retention_counts,
        "future_physical": summary(future),
        "normalized": summary(normalized),
        "all_channel_tail_fraction": float((np.abs(z) > 1).any(axis=1).mean()),
    }
    normalizer_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(normalizer_path, payload)
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "fit_samples": len(dataset.rows),
                "fit_tokens": len(future),
                "normalizer": str(normalizer_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
