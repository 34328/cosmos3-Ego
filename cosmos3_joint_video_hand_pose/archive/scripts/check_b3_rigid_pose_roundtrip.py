#!/usr/bin/env python3
"""Small CPU check for B3 frame-delta rigid-pose encoding."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from cosmos3_joint_video_hand_pose.src.action import (
    _integrate_rigid_increments,
    pose27_from_streams,
)


def pose_stream(translations: np.ndarray, rotvecs: np.ndarray) -> np.ndarray:
    quaternions_xyzw = Rotation.from_rotvec(rotvecs).as_quat()
    return np.concatenate(
        (translations, quaternions_xyzw[:, 3:4], quaternions_xyzw[:, :3]),
        axis=1,
    )


def matrices(stream: np.ndarray) -> np.ndarray:
    result = np.broadcast_to(np.eye(4), (len(stream), 4, 4)).copy()
    result[:, :3, :3] = Rotation.from_quat(stream[:, [4, 5, 6, 3]]).as_matrix()
    result[:, :3, 3] = stream[:, :3]
    return result


def main() -> None:
    frames = 9
    time = np.arange(frames, dtype=np.float64)
    head = pose_stream(
        np.stack((0.01 * time, -0.002 * time, 0.001 * time), axis=1),
        np.stack((0.001 * time, -0.002 * time, 0.004 * time), axis=1),
    )
    right = pose_stream(
        np.stack((0.35 + 0.006 * time, -0.20 + 0.003 * time, 0.55 - 0.002 * time), axis=1),
        np.stack((0.01 + 0.003 * time, -0.04 + 0.001 * time, 0.02 * time), axis=1),
    )
    left = pose_stream(
        np.stack((-0.30 + 0.004 * time, -0.18 - 0.002 * time, 0.52 + 0.001 * time), axis=1),
        np.stack((-0.02 + 0.002 * time, 0.03 - 0.001 * time, -0.015 * time), axis=1),
    )
    encoded = pose27_from_streams(
        head,
        right,
        left,
        rigid_pose_frame_delta=True,
    )
    decoded = [
        _integrate_rigid_increments(torch.from_numpy(encoded[:, start : start + 9])).numpy()
        for start in (0, 9, 18)
    ]
    head_m, right_m, left_m = matrices(head), matrices(right), matrices(left)
    expected = [
        np.linalg.inv(head_m[0]) @ head_m,
        np.linalg.inv(head_m[0]) @ right_m,
        np.linalg.inv(head_m[0]) @ left_m,
    ]
    errors = [
        float(np.max(np.abs(actual - target)))
        for actual, target in zip(decoded, expected, strict=True)
    ]
    if max(errors) > 2e-6:
        raise AssertionError(f"B3 round-trip error too large: {errors}")
    print({"frames": frames, "max_abs_errors": errors})


if __name__ == "__main__":
    main()
