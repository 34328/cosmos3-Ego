"""Candidate 21-landmark hand topology for the geometry experiment.

The edge order follows the experiment document's standard MediaPipe hypothesis:
wrist 0, then four joints for thumb, index, middle, ring and little fingers.
Bone supervision must remain disabled until this is visually checked on a real
Mecka sample.  The frozen codec omits landmark 0, so geometry code prepends a
zero wrist in current-frame wrist-local coordinates before using these edges.
"""

from __future__ import annotations

import torch


HAND_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def prepend_wrist(points: torch.Tensor) -> torch.Tensor:
    """Convert ``[...,20,3]`` non-wrist points to wrist-local ``[...,21,3]``."""
    if points.shape[-2:] != (20, 3):
        raise ValueError(f"non-wrist hand points must end in [20,3], got {tuple(points.shape)}")
    wrist = torch.zeros((*points.shape[:-2], 1, 3), dtype=points.dtype, device=points.device)
    return torch.cat((wrist, points), dim=-2)


def bone_lengths(points: torch.Tensor) -> torch.Tensor:
    """Return the 20 canonical bone lengths from ``[...,21,3]`` points."""
    if points.shape[-2:] != (21, 3):
        raise ValueError(f"hand points must end in [21,3], got {tuple(points.shape)}")
    starts = torch.tensor([edge[0] for edge in HAND_BONE_EDGES], device=points.device)
    ends = torch.tensor([edge[1] for edge in HAND_BONE_EDGES], device=points.device)
    return torch.linalg.vector_norm(points[..., ends, :] - points[..., starts, :], dim=-1)
