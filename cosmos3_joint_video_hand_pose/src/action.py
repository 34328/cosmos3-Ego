from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .codec import FrozenHandMLPAE15
from .normalization import PiecewiseAsinhNormalizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/cosmos3_action_contract/v2"
CODEC_ROOT = PROJECT_ROOT / "artifacts/cosmos3_hand_codecs/v2_4/option_b_mlp15"
DEFAULT_STATE_NORMALIZER = ACTION_ARTIFACT_ROOT / "normalizers/state_normalizer.json"
DEFAULT_FUTURE_NORMALIZER = ACTION_ARTIFACT_ROOT / "normalizers/future_delta_normalizer.json"


def pose_matrices(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"pose must have shape [T,7], got {pose.shape}")
    result = np.broadcast_to(np.eye(4), (len(pose), 4, 4)).copy()
    result[:, :3, :3] = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
    result[:, :3, 3] = pose[:, :3]
    return result


def pose9(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform)
    rotation = transform[:, :3, :3]
    return np.concatenate((transform[:, :3, 3], rotation[:, :, 0], rotation[:, :, 1]), axis=-1)


def pose27_from_streams(
    head_pose: np.ndarray,
    right_wrist_pose: np.ndarray,
    left_wrist_pose: np.ndarray,
    *,
    rigid_pose_frame_delta: bool = False,
) -> np.ndarray:
    """Build physical camera/right/left values before normalization.

    The legacy representation stores every future rigid pose relative to its
    first frame. The optional B3 representation keeps the clean a0 state
    unchanged but stores each future pose as T[t-1]^-1 @ T[t].
    """
    head = pose_matrices(head_pose)
    right = pose_matrices(right_wrist_pose)
    left = pose_matrices(left_wrist_pose)
    length = len(head)
    if not (len(right) == len(left) == length) or length < 1:
        raise ValueError("camera and wrist streams must have identical non-zero lengths")

    camera9 = np.empty((length, 9), dtype=np.float64)
    right9 = np.empty((length, 9), dtype=np.float64)
    left9 = np.empty((length, 9), dtype=np.float64)
    camera9[0] = pose9(np.eye(4)[None])[0]
    right9[0] = pose9((np.linalg.inv(head[0]) @ right[0])[None])[0]
    left9[0] = pose9((np.linalg.inv(head[0]) @ left[0])[None])[0]
    if length > 1:
        if rigid_pose_frame_delta:
            camera9[1:] = pose9(np.linalg.inv(head[:-1]) @ head[1:])
            right9[1:] = pose9(np.linalg.inv(right[:-1]) @ right[1:])
            left9[1:] = pose9(np.linalg.inv(left[:-1]) @ left[1:])
        else:
            camera9[1:] = pose9(np.linalg.inv(head[0]) @ head[1:])
            right9[1:] = pose9(np.linalg.inv(right[0]) @ right[1:])
            left9[1:] = pose9(np.linalg.inv(left[0]) @ left[1:])
    return np.concatenate((camera9, right9, left9), axis=-1).astype(np.float32)


def wrist_local_non_wrist_points(wrist_pose: np.ndarray, keypoints: np.ndarray) -> torch.Tensor:
    wrist = pose_matrices(wrist_pose)
    points = np.asarray(keypoints, dtype=np.float64).reshape(-1, 21, 3)
    if len(points) != len(wrist):
        raise ValueError("wrist pose and keypoint lengths differ")
    delta = points - wrist[:, None, :3, 3]
    local = np.einsum("tji,tnj->tni", wrist[:, :3, :3], delta)
    if not np.isfinite(local).all():
        raise ValueError("non-finite wrist-local hand keypoints")
    return torch.from_numpy(local[:, 1:].astype(np.float32))


def _pose9_matrices(values: torch.Tensor) -> torch.Tensor:
    """Convert translation plus the first two rotation columns to rigid transforms."""
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError(f"pose9 values must have shape [T,9], got {tuple(values.shape)}")
    translation = values[:, :3]
    first = torch.nn.functional.normalize(values[:, 3:6], dim=-1, eps=1e-8)
    second_raw = values[:, 6:9]
    second = second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    third = torch.linalg.cross(first, second, dim=-1)
    result = torch.eye(4, dtype=values.dtype, device=values.device).repeat(len(values), 1, 1)
    result[:, :3, :3] = torch.stack((first, second, third), dim=-1)
    result[:, :3, 3] = translation
    return result


def _transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tij,tnj->tni", transform[:, :3, :3], points) + transform[:, None, :3, 3]


def _integrate_rigid_increments(values: torch.Tensor) -> torch.Tensor:
    """Interpret pose9[0] as state and pose9[1:] as right-composable increments."""
    increments = _pose9_matrices(values)
    integrated = increments.clone()
    for frame in range(1, len(integrated)):
        integrated[frame] = integrated[frame - 1] @ increments[frame]
    return integrated


@dataclass(frozen=True)
class DecodedAction57:
    headcam_f0: torch.Tensor
    right_wrist_f0: torch.Tensor
    left_wrist_f0: torch.Tensor
    right_keypoints_f0: torch.Tensor
    left_keypoints_f0: torch.Tensor


class Action57Builder:
    """Build the fixed camera/right/left 57D action contract."""

    def __init__(
        self,
        *,
        state_normalizer: str | Path = DEFAULT_STATE_NORMALIZER,
        future_normalizer: str | Path = DEFAULT_FUTURE_NORMALIZER,
        right_codec: str | Path = CODEC_ROOT / "right_mlp15_primary.pt",
        left_codec: str | Path = CODEC_ROOT / "left_mlp15_primary.pt",
        rigid_pose_frame_delta: bool = False,
    ):
        self.state_normalizer = PiecewiseAsinhNormalizer(state_normalizer)
        self.future_normalizer = PiecewiseAsinhNormalizer(future_normalizer)
        self.right_codec = FrozenHandMLPAE15(right_codec)
        self.left_codec = FrozenHandMLPAE15(left_codec)
        self.rigid_pose_frame_delta = bool(rigid_pose_frame_delta)

    def build(
        self,
        *,
        head_pose: np.ndarray,
        right_wrist_pose: np.ndarray,
        left_wrist_pose: np.ndarray,
        right_keypoints: np.ndarray,
        left_keypoints: np.ndarray,
    ) -> torch.Tensor:
        pose27 = torch.from_numpy(
            pose27_from_streams(
                head_pose,
                right_wrist_pose,
                left_wrist_pose,
                rigid_pose_frame_delta=self.rigid_pose_frame_delta,
            )
        )
        length = len(pose27)
        normalized27 = torch.empty_like(pose27)
        normalized27[0] = self.state_normalizer.normalize(pose27[0])
        if length > 1:
            normalized27[1:] = self.future_normalizer.normalize(pose27[1:])

        right_latent = self.right_codec.encode(wrist_local_non_wrist_points(right_wrist_pose, right_keypoints))
        left_latent = self.left_codec.encode(wrist_local_non_wrist_points(left_wrist_pose, left_keypoints))
        action = torch.empty((length, 57), dtype=torch.float32)
        action[:, 0:9] = normalized27[:, 0:9]
        action[:, 9:18] = normalized27[:, 9:18]
        action[:, 18:33] = right_latent
        action[:, 33:42] = normalized27[:, 18:27]
        action[:, 42:57] = left_latent
        if not torch.isfinite(action).all():
            raise ValueError("non-finite values in constructed action")
        return action

    @torch.no_grad()
    def decode(self, action: torch.Tensor) -> DecodedAction57:
        """Invert the 57D contract into camera/wrist transforms and 21-point hands in F0."""
        action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim != 2 or action.shape[1] != 57 or len(action) < 1:
            raise ValueError(f"action must have shape [T,57] with T>=1, got {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise ValueError("cannot decode non-finite action values")

        normalized27 = torch.cat((action[:, 0:18], action[:, 33:42]), dim=-1)
        pose27 = torch.empty_like(normalized27)
        pose27[0] = self.state_normalizer.denormalize(normalized27[0])
        if len(action) > 1:
            pose27[1:] = self.future_normalizer.denormalize(normalized27[1:])

        if self.rigid_pose_frame_delta:
            headcam_f0 = _integrate_rigid_increments(pose27[:, 0:9])
            right_wrist_f0 = _integrate_rigid_increments(pose27[:, 9:18])
            left_wrist_f0 = _integrate_rigid_increments(pose27[:, 18:27])
        else:
            headcam_f0 = _pose9_matrices(pose27[:, 0:9])
            right_relative = _pose9_matrices(pose27[:, 9:18])
            left_relative = _pose9_matrices(pose27[:, 18:27])
            right_wrist_f0 = right_relative.clone()
            left_wrist_f0 = left_relative.clone()
            if len(action) > 1:
                right_wrist_f0[1:] = right_relative[0] @ right_relative[1:]
                left_wrist_f0[1:] = left_relative[0] @ left_relative[1:]

        wrist_origin = torch.zeros((len(action), 1, 3), dtype=action.dtype, device=action.device)
        right_local = torch.cat((wrist_origin, self.right_codec.decode(action[:, 18:33])), dim=1)
        left_local = torch.cat((wrist_origin, self.left_codec.decode(action[:, 42:57])), dim=1)
        return DecodedAction57(
            headcam_f0=headcam_f0,
            right_wrist_f0=right_wrist_f0,
            left_wrist_f0=left_wrist_f0,
            right_keypoints_f0=_transform_points(right_wrist_f0, right_local),
            left_keypoints_f0=_transform_points(left_wrist_f0, left_local),
        )
