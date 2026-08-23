from __future__ import annotations

import numpy as np


CANVAS_HEIGHT = 368
CANVAS_WIDTH = 640
SPATIAL_TOKENS_PER_LATENT_FRAME = 240
# Validated across a complete CP=2 data window and the following packed batch.
MAX_SEQUENCE_LENGTH = 85_000
RETENTION_LADDER = (0.8, 0.7, 0.6, 0.5)


def cosmos_wam_token_count(text_tokens: int, frames: int) -> int:
    if frames < 1 or (frames - 1) % 4:
        raise ValueError(f"frames must satisfy T=1+4n, got {frames}")
    latent_frames = 1 + (frames - 1) // 4
    return int(text_tokens + 1 + SPATIAL_TOKENS_PER_LATENT_FRAME * latent_frames + 2 + frames)


def aligned_frame_count(source_frames: int) -> int:
    if source_frames <= 1:
        return source_frames
    return source_frames - ((source_frames - 1) % 4)


def retention_candidates(source_frames: int) -> list[tuple[float, int]]:
    """Return original, 80%, 70%, 60%, and 50% legal frame counts."""
    aligned = aligned_frame_count(source_frames)
    if aligned <= 1:
        return []
    result = [(1.0, aligned)]
    for ratio in RETENTION_LADDER:
        frames = 1 + 4 * int(ratio * (aligned - 1) // 4)
        if frames > 1 and frames != result[-1][1]:
            result.append((ratio, frames))
    return result


def uniform_frame_indices(source_frames: int, target_frames: int) -> np.ndarray:
    aligned = aligned_frame_count(source_frames)
    if target_frames <= 1 or target_frames > aligned or (target_frames - 1) % 4:
        raise ValueError(f"invalid target_frames={target_frames} for source_frames={source_frames}")
    if target_frames == aligned:
        return np.arange(aligned, dtype=np.int64)
    future = np.rint(np.linspace(1, aligned - 1, target_frames - 1, dtype=np.float64)).astype(np.int64)
    if np.unique(future).size != future.size:
        raise RuntimeError("uniform temporal sampling produced duplicate indexes")
    return np.concatenate((np.array([0], dtype=np.int64), future))


def retained_fps(source_fps: float, source_frames: int, target_frames: int) -> float:
    aligned = aligned_frame_count(source_frames)
    if source_fps <= 0 or aligned <= 1 or target_frames <= 1 or target_frames > aligned:
        raise ValueError("invalid source FPS or frame counts")
    return float(source_fps) * (target_frames - 1) / (aligned - 1)


def select_frame_indices(
    source_frames: int,
    text_tokens: int,
    *,
    cap: int = MAX_SEQUENCE_LENGTH,
) -> np.ndarray | None:
    """Select the first legal retention tier below the strict packer cap."""
    for _, frames in retention_candidates(source_frames):
        if cosmos_wam_token_count(text_tokens, frames) < cap:
            return uniform_frame_indices(source_frames, frames)
    return None
