import numpy as np
import torch

from cosmos3_joint_video_hand_pose.src.action import Action57Builder
from cosmos3_joint_video_hand_pose.src.trajectory_metrics import evaluate_actions


def _moving_pose(length: int, offset: tuple[float, float, float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    time = np.linspace(0, 1, length)
    pose = np.zeros((length, 7), dtype=np.float64)
    pose[:, :3] = np.asarray(offset) + np.stack(
        (0.12 * time, 0.04 * np.sin(np.pi * time), 0.06 * time**2), axis=-1
    )
    xyzw = Rotation.from_euler(
        "xyz",
        np.stack((8 * time, -5 * time**2, 12 * np.sin(np.pi * time)), axis=-1),
        degrees=True,
    ).as_quat()
    pose[:, 3:] = xyzw[:, [3, 0, 1, 2]]
    return pose


def _reference_action(length: int = 9) -> tuple[Action57Builder, torch.Tensor]:
    builder = Action57Builder()
    head = _moving_pose(length, (1.0, -0.2, 0.5))
    right = _moving_pose(length, (1.2, -0.1, 0.8))
    left = _moving_pose(length, (0.8, -0.1, 0.8))
    offsets = np.zeros((length, 21, 3), dtype=np.float64)
    offsets[:, :, 1] = np.arange(21) * 0.002
    return builder, builder.build(
        head_pose=head,
        right_wrist_pose=right,
        left_wrist_pose=left,
        right_keypoints=(right[:, None, :3] + offsets).reshape(length, 63),
        left_keypoints=(left[:, None, :3] + offsets).reshape(length, 63),
    )


def test_perfect_trajectory_passes_strict_memorization_gate():
    builder, reference = _reference_action()
    metrics = evaluate_actions(reference.clone(), reference, builder)
    assert metrics["overfit_gate"]["passed"]
    assert metrics["condition_slot_normalized_max_abs_error"] == 0
    assert metrics["streams"]["right_wrist"]["translation"]["temporal_position_correlation"] > 0.999
    assert metrics["hands"]["right"]["full_mpjpe_mm"] == 0


def test_reversed_motion_cannot_pass_on_plausible_amplitude_alone():
    builder, reference = _reference_action()
    prediction = reference.clone()
    prediction[1:] = torch.flip(reference[1:], dims=(0,))
    metrics = evaluate_actions(prediction, reference, builder)
    right = metrics["streams"]["right_wrist"]["translation"]
    assert 0.8 < right["prediction_over_target_motion"] < 1.2
    assert right["temporal_position_correlation"] < 0
    assert not metrics["overfit_gate"]["passed"]
