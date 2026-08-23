"""Evaluate whether a generated action memorized its paired physical trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .action import DEFAULT_FUTURE_NORMALIZER, DEFAULT_STATE_NORMALIZER, Action57Builder
from .monitoring import _load_prediction


TRANSFORM_STREAMS = ("camera", "right_wrist", "left_wrist")


def _correlation(prediction: np.ndarray, target: np.ndarray) -> float | None:
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    denominator = float(np.linalg.norm(prediction) * np.linalg.norm(target))
    return float(np.sum(prediction * target) / denominator) if denominator > 1e-12 else None


def _rotation_errors_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(prediction, -1, -2) @ target
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.rad2deg(np.arccos(cosine))


def _local_points(transforms: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum(
        "tji,tnj->tni",
        transforms[:, :3, :3],
        points - transforms[:, None, :3, 3],
    )


def _translation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    prediction = prediction[1:]
    target_future = target[1:]
    static = np.broadcast_to(target[0], target_future.shape)
    error = np.linalg.norm(prediction - target_future, axis=-1)
    static_error = np.linalg.norm(static - target_future, axis=-1)
    centered_target = target_future - target_future.mean(axis=0, keepdims=True)
    denominator = float(np.sum(centered_target**2))
    r2 = 1 - float(np.sum((prediction - target_future) ** 2)) / denominator if denominator > 1e-12 else None
    target_motion = float(np.sqrt(np.mean(np.sum(centered_target**2, axis=-1))))
    prediction_centered = prediction - prediction.mean(axis=0, keepdims=True)
    prediction_motion = float(np.sqrt(np.mean(np.sum(prediction_centered**2, axis=-1))))
    prediction_mm = float(error.mean() * 1000)
    static_mm = float(static_error.mean() * 1000)
    return {
        "mean_error_mm": prediction_mm,
        "p95_error_mm": float(np.quantile(error, 0.95) * 1000),
        "static_mean_error_mm": static_mm,
        "model_over_static": prediction_mm / static_mm if static_mm > 1e-9 else None,
        "constant_baseline_r2": r2,
        "temporal_position_correlation": _correlation(prediction, target_future),
        "delta_correlation": _correlation(np.diff(prediction, axis=0), np.diff(target_future, axis=0)),
        "prediction_motion_rms_mm": prediction_motion * 1000,
        "target_motion_rms_mm": target_motion * 1000,
        "prediction_over_target_motion": prediction_motion / target_motion if target_motion > 1e-12 else None,
    }


def evaluate_actions(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    builder: Action57Builder,
) -> dict:
    prediction = torch.as_tensor(prediction, dtype=torch.float32)
    reference = torch.as_tensor(reference, dtype=torch.float32)
    if prediction.shape != reference.shape or prediction.ndim != 2 or prediction.shape[1] != 57:
        raise ValueError(
            f"prediction and reference must have identical [T,57] shape, got "
            f"{tuple(prediction.shape)} and {tuple(reference.shape)}"
        )
    if len(prediction) < 3:
        raise ValueError("trajectory metrics require at least three action slots")
    pred_decoded = builder.decode(prediction)
    ref_decoded = builder.decode(reference)
    pred_transforms = {
        "camera": pred_decoded.headcam_f0.numpy(),
        "right_wrist": pred_decoded.right_wrist_f0.numpy(),
        "left_wrist": pred_decoded.left_wrist_f0.numpy(),
    }
    ref_transforms = {
        "camera": ref_decoded.headcam_f0.numpy(),
        "right_wrist": ref_decoded.right_wrist_f0.numpy(),
        "left_wrist": ref_decoded.left_wrist_f0.numpy(),
    }

    streams = {}
    for name in TRANSFORM_STREAMS:
        pred_transform = pred_transforms[name]
        ref_transform = ref_transforms[name]
        rotation_error = _rotation_errors_deg(pred_transform[1:, :3, :3], ref_transform[1:, :3, :3])
        streams[name] = {
            "translation": _translation_metrics(pred_transform[:, :3, 3], ref_transform[:, :3, 3]),
            "rotation_mean_deg": float(rotation_error.mean()),
            "rotation_p95_deg": float(np.quantile(rotation_error, 0.95)),
        }

    hands = {}
    for side in ("right", "left"):
        pred_wrist = pred_transforms[f"{side}_wrist"]
        ref_wrist = ref_transforms[f"{side}_wrist"]
        pred_points = getattr(pred_decoded, f"{side}_keypoints_f0").numpy()
        ref_points = getattr(ref_decoded, f"{side}_keypoints_f0").numpy()
        pred_local = _local_points(pred_wrist, pred_points)
        ref_local = _local_points(ref_wrist, ref_points)
        copy_initial = np.broadcast_to(ref_local[0], ref_local[1:].shape)
        hands[side] = {
            "full_mpjpe_mm": float(np.linalg.norm(pred_points[1:] - ref_points[1:], axis=-1).mean() * 1000),
            "local_shape_mpjpe_mm": float(np.linalg.norm(pred_local[1:] - ref_local[1:], axis=-1).mean() * 1000),
            "copy_initial_local_shape_mpjpe_mm": float(
                np.linalg.norm(copy_initial - ref_local[1:], axis=-1).mean() * 1000
            ),
        }

    normalized_blocks = {
        "camera_translation": (slice(0, 3)),
        "camera_rotation6d": (slice(3, 9)),
        "right_wrist_translation": (slice(9, 12)),
        "right_wrist_rotation6d": (slice(12, 18)),
        "right_hand_latent": (slice(18, 33)),
        "left_wrist_translation": (slice(33, 36)),
        "left_wrist_rotation6d": (slice(36, 42)),
        "left_hand_latent": (slice(42, 57)),
    }
    difference = prediction[1:] - reference[1:]
    normalized_rmse = {
        name: float(torch.sqrt(torch.mean(difference[:, block] ** 2)))
        for name, block in normalized_blocks.items()
    }
    ratios = [streams[name]["translation"]["model_over_static"] for name in TRANSFORM_STREAMS]
    correlations = [streams[name]["translation"]["temporal_position_correlation"] for name in TRANSFORM_STREAMS]
    delta_correlations = [streams[name]["translation"]["delta_correlation"] for name in TRANSFORM_STREAMS]
    checks = {
        "all_translation_model_over_static_lt_0_5": all(value is not None and value < 0.5 for value in ratios),
        "all_position_correlation_gt_0_8": all(value is not None and value > 0.8 for value in correlations),
        "all_delta_correlation_gt_0_25": all(value is not None and value > 0.25 for value in delta_correlations),
        "both_full_hand_mpjpe_lt_30mm": all(hands[side]["full_mpjpe_mm"] < 30 for side in hands),
    }
    return {
        "frames": len(prediction),
        "condition_slot_normalized_max_abs_error": float(torch.max(torch.abs(prediction[0] - reference[0]))),
        "streams": streams,
        "hands": hands,
        "normalized_rmse_secondary": normalized_rmse,
        "overfit_gate": {"passed": all(checks.values()), "checks": checks},
    }


def main(args: argparse.Namespace) -> None:
    prediction = _load_prediction(args.sample_outputs)
    reference = torch.tensor(json.loads(args.reference_action.read_text(encoding="utf-8")), dtype=torch.float32)
    builder = Action57Builder(
        state_normalizer=args.state_normalizer,
        future_normalizer=args.future_normalizer,
    )
    result = evaluate_actions(prediction, reference, builder)
    result["inputs"] = {
        "sample_outputs": str(args.sample_outputs.resolve()),
        "reference_action": str(args.reference_action.resolve()),
        "state_normalizer": str(args.state_normalizer.resolve()),
        "future_normalizer": str(args.future_normalizer.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-outputs", type=Path, required=True)
    parser.add_argument("--reference-action", type=Path, required=True)
    parser.add_argument("--state-normalizer", type=Path, default=DEFAULT_STATE_NORMALIZER)
    parser.add_argument("--future-normalizer", type=Path, default=DEFAULT_FUTURE_NORMALIZER)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
