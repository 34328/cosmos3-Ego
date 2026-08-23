#!/usr/bin/env python3
"""Build action contract v2 from the live 85K data sampler."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import zarr

from cosmos3_joint_video_hand_pose.src.action import pose27_from_streams
from cosmos3_joint_video_hand_pose.src.dataset import EgoVerseSegmentDataset
from cosmos3_joint_video_hand_pose.src.temporal import uniform_frame_indices


ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / "artifacts/cosmos3_action_contract/v1"
OUT = Path(__file__).resolve().parent
TRAINING_MANIFESTS = ROOT.parent / "training_manifests"
EPISODES = TRAINING_MANIFESTS / "mecka_100h_v1_episodes.csv"
SEGMENTS = TRAINING_MANIFESTS / "mecka_100h_v1_segments.csv"
MAX_SEQUENCE_LENGTH = 85_000
TRANSLATION_CHANNELS = np.array([0, 1, 2, 9, 10, 11, 18, 19, 20])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def asinh_normalize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (values - center) / scale
    return np.where(np.abs(z) <= 1, z, np.sign(z) * (1 + np.arcsinh(np.abs(z) - 1)))


def asinh_denormalize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = np.where(np.abs(values) <= 1, values, np.sign(values) * (1 + np.sinh(np.abs(values) - 1)))
    return z * scale + center


def summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        name: np.quantile(values, quantile, axis=0).astype(float).tolist()
        for name, quantile in (("p01", 0.01), ("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
    } | {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-8).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
    }


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite immutable V2 artifact: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    normalizer_dir = OUT / "normalizers"
    normalizer_dir.mkdir(exist_ok=True)

    # This is deliberately the production planning implementation, rather
    # than a duplicate estimate of JSON-token count or retention behavior.
    dataset = EgoVerseSegmentDataset(
        EPISODES,
        SEGMENTS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        prompt_mode="episode_context_and_segment",
    )
    futures: list[np.ndarray] = []
    plan_counts = dict(dataset.retention_counts)
    for row in dataset.rows:
        episode = dataset.episodes[row["episode_hash"]]
        source_frames = int(row["end_idx"]) - int(row["start_idx"])
        source_indices = int(row["start_idx"]) + uniform_frame_indices(source_frames, int(row["_selected_frames"]))
        group = zarr.open_group(episode["abs_zarr_path"], mode="r")
        pose27 = pose27_from_streams(
            np.asarray(group["obs_head_pose"][source_indices]),
            np.asarray(group["right.obs_wrist_pose"][source_indices]),
            np.asarray(group["left.obs_wrist_pose"][source_indices]),
        )
        futures.append(pose27[1:].astype(np.float64, copy=False))
    future = np.concatenate(futures, axis=0)
    if future.shape[1] != 27 or not np.isfinite(future).all():
        raise RuntimeError(f"invalid future pose statistics array: {future.shape}")

    v1_future = json.loads((V1 / "normalizers/future_delta_normalizer.json").read_text(encoding="utf-8"))
    v1_state = V1 / "normalizers/state_normalizer.json"
    # Mirror PiecewiseAsinhNormalizer's v1 float32 arithmetic exactly. v2
    # must not perturb rotation channels merely through JSON roundoff.
    q01 = np.asarray(v1_future["stats"]["q01"], dtype=np.float32)
    q99 = np.asarray(v1_future["stats"]["q99"], dtype=np.float32)
    center = ((q01 + q99) / np.float32(2)).astype(np.float64)
    scale = np.maximum((q99 - q01) / np.float32(2), np.float32(1e-8)).astype(np.float64)
    std = np.maximum(future.std(axis=0), 1e-8)
    center[TRANSLATION_CHANNELS] = 0.0
    scale[TRANSLATION_CHANNELS] = std[TRANSLATION_CHANNELS]

    normalized = asinh_normalize(future, center, scale)
    recovered = asinh_denormalize(normalized, center, scale)
    translation = future[:, TRANSLATION_CHANNELS]
    report = {
        "schema_version": 2,
        "fit_split": "train_only",
        "fit_tokens": int(len(future)),
        "fit_samples": int(len(dataset.rows)),
        "sampler": {
            "implementation": "EgoVerseSegmentDataset._build_temporal_plan",
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "prompt_mode": "episode_context_and_segment",
            "canvas": "640x368",
            "temporal": "T=1+4n",
            "retention_counts": plan_counts,
        },
        "translation_channels": TRANSLATION_CHANNELS.tolist(),
        "translation_scale_method": "center_zero_train_only_std",
        "future_physical": summary(future),
        "translation_physical": summary(translation),
        "normalized": summary(normalized),
        "normalized_translation": summary(normalized[:, TRANSLATION_CHANNELS]),
        "translation_linear_tail_fraction": float((np.abs(translation / std[TRANSLATION_CHANNELS]) > 1).any(axis=1).mean()),
        "all_channel_tail_fraction": float((np.abs((future - center) / scale) > 1).any(axis=1).mean()),
        "inverse_roundtrip_max_abs_float64": float(np.max(np.abs(recovered - future))),
    }
    future_payload = {
        "method": "piecewise_asinh_rot",
        "beta": 1.0,
        "fit_split": "train_only",
        "stats": {"center": center.tolist(), "scale": scale.tolist()},
        "v6_contract": {
            "translation_channels": TRANSLATION_CHANNELS.tolist(),
            "translation_center": "zero",
            "translation_scale": "train_only_std_from_actual_85k_sampler",
            "rotation_channels": "v1_q01_q99_center_scale_unchanged",
            "clamp_after_normalization": False,
        },
    }
    shutil.copyfile(v1_state, normalizer_dir / "state_normalizer.json")
    atomic_json(normalizer_dir / "future_delta_normalizer.json", future_payload)
    atomic_json(OUT / "normalizer_report.json", report)

    v1_manifest = json.loads((V1 / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 2,
        "artifact_id": "egoverse_cosmos3_action_contract_v2",
        "status": "ready",
        "training_ready": True,
        "supersedes": {"artifact_id": v1_manifest["artifact_id"], "manifest_sha256": sha256(V1 / "manifest.json")},
        "contract": v1_manifest["contract"],
        "codec": v1_manifest["codec"],
        "source_data": v1_manifest["source_data"],
        "normalizers": {
            "state": {
                "path": "normalizers/state_normalizer.json",
                "sha256": sha256(normalizer_dir / "state_normalizer.json"),
                "source": "v1_byte_identical",
            },
            "future_delta": {
                "path": "normalizers/future_delta_normalizer.json",
                "sha256": sha256(normalizer_dir / "future_delta_normalizer.json"),
                "method": "piecewise_asinh_rot",
                "translation_scale_method": "center_zero_train_only_std",
                "rotation_scale_method": "v1_q01_q99_center_scale_unchanged",
                "report": {"path": "normalizer_report.json", "sha256": sha256(OUT / "normalizer_report.json")},
            },
        },
        "checkpoint_binding": v1_manifest["checkpoint_binding"],
        "blockers": [],
    }
    atomic_json(OUT / "manifest.json", manifest)
    print(json.dumps({"fit_samples": len(dataset.rows), "fit_tokens": len(future), "tail_fraction": report["translation_linear_tail_fraction"]}, indent=2))


if __name__ == "__main__":
    main()
