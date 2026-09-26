from pathlib import Path

import numpy as np
import torch

from cosmos3_joint_video_hand_pose.src.action import Action57Builder, pose_matrices, wrist_local_non_wrist_points


ROOT = Path(__file__).resolve().parents[1]


def _poses(length: int, x_offset: float = 0.0) -> np.ndarray:
    result = np.zeros((length, 7), dtype=np.float64)
    result[:, 0] = x_offset + np.arange(length) * 0.01
    result[:, 3] = 1.0
    return result


def test_wrist_local_points_remove_world_translation():
    wrist = _poses(2, 1.0)
    local = np.zeros((2, 21, 3), dtype=np.float64)
    local[:, :, 1] = np.arange(21) * 0.001
    matrices = pose_matrices(wrist)
    world = local + matrices[:, None, :3, 3]
    recovered = wrist_local_non_wrist_points(wrist, world.reshape(2, 63))
    torch.testing.assert_close(recovered, torch.from_numpy(local[:, 1:].astype(np.float32)))


def test_action_builder_returns_native_57d_contract():
    length = 5
    head = _poses(length)
    right = _poses(length, 0.2)
    left = _poses(length, -0.2)
    right_points = np.repeat(right[:, None, :3], 21, axis=1)
    left_points = np.repeat(left[:, None, :3], 21, axis=1)
    builder = Action57Builder()
    action = builder.build(
        head_pose=head,
        right_wrist_pose=right,
        left_wrist_pose=left,
        right_keypoints=right_points.reshape(length, 63),
        left_keypoints=left_points.reshape(length, 63),
    )
    assert action.shape == (length, 57)
    assert torch.isfinite(action).all()


def test_action_decode_recovers_pose_contract_and_finite_hand_points():
    length = 5
    head = _poses(length)
    right = _poses(length, 0.2)
    left = _poses(length, -0.2)
    local = np.zeros((length, 21, 3), dtype=np.float64)
    local[:, :, 1] = np.arange(21) * 0.001
    right_points = local + right[:, None, :3]
    left_points = local + left[:, None, :3]
    builder = Action57Builder()
    action = builder.build(
        head_pose=head,
        right_wrist_pose=right,
        left_wrist_pose=left,
        right_keypoints=right_points.reshape(length, 63),
        left_keypoints=left_points.reshape(length, 63),
    )

    decoded = builder.decode(action)
    assert decoded.headcam_f0.shape == (length, 4, 4)
    assert decoded.right_wrist_f0.shape == (length, 4, 4)
    assert decoded.left_wrist_f0.shape == (length, 4, 4)
    assert decoded.right_keypoints_f0.shape == (length, 21, 3)
    assert decoded.left_keypoints_f0.shape == (length, 21, 3)
    torch.testing.assert_close(decoded.headcam_f0[:, :3, 3], torch.from_numpy(head[:, :3]).float())
    torch.testing.assert_close(decoded.right_wrist_f0[:, :3, 3], torch.from_numpy(right[:, :3]).float())
    torch.testing.assert_close(decoded.left_wrist_f0[:, :3, 3], torch.from_numpy(left[:, :3]).float())
    assert torch.isfinite(decoded.right_keypoints_f0).all()
    assert torch.isfinite(decoded.left_keypoints_f0).all()


def test_action_decode_recovers_rotated_f0_anchored_transforms():
    """Future wrist deltas must be left-multiplied by the F0 initial wrist pose."""
    from scipy.spatial.transform import Rotation

    length = 5
    angles = np.array([[4, -8, 12], [8, -3, 17], [13, 2, 22], [18, 7, 27], [23, 12, 32]])

    def poses(offset):
        result = np.zeros((length, 7), dtype=np.float64)
        result[:, :3] = np.asarray(offset) + np.arange(length)[:, None] * [0.01, -0.004, 0.006]
        # scipy emits xyzw; the dataset/action contract stores wxyz.
        xyzw = Rotation.from_euler("xyz", angles, degrees=True).as_quat()
        result[:, 3:] = xyzw[:, [3, 0, 1, 2]]
        return result

    head = poses([1.0, -0.2, 0.5])
    right = poses([1.2, -0.1, 0.8])
    left = poses([0.8, -0.1, 0.8])
    right_points = np.repeat(right[:, None, :3], 21, axis=1)
    left_points = np.repeat(left[:, None, :3], 21, axis=1)
    builder = Action57Builder()
    decoded = builder.decode(
        builder.build(
            head_pose=head,
            right_wrist_pose=right,
            left_wrist_pose=left,
            right_keypoints=right_points.reshape(length, 63),
            left_keypoints=left_points.reshape(length, 63),
        )
    )

    head_m = pose_matrices(head)
    expected_head = np.linalg.inv(head_m[0]) @ head_m
    expected_right = np.linalg.inv(head_m[0]) @ pose_matrices(right)
    expected_left = np.linalg.inv(head_m[0]) @ pose_matrices(left)
    np.testing.assert_allclose(decoded.headcam_f0.numpy(), expected_head, atol=3e-6)
    np.testing.assert_allclose(decoded.right_wrist_f0.numpy(), expected_right, atol=3e-6)
    np.testing.assert_allclose(decoded.left_wrist_f0.numpy(), expected_left, atol=3e-6)
