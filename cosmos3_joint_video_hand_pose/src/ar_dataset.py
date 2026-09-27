"""EgoVerse clips for joint video-action AR training (lingbot-va ``frame_stride`` layout).

Each sample is one continuous window of a segment. Video keeps every
``frame_stride``-th source frame (``T = 4n + 1`` frames, ``n + 1`` VAE latents);
actions keep every source frame and are grouped ``K = 4 * frame_stride`` per latent.
Latent group 0 repeats the first-frame state ``K`` times (a clean condition); group
``j >= 1`` holds the actions of source frames ``K(j-1)+1 .. Kj`` of the window, whose
last frame is latent ``j``'s last video frame. Hand visibility follows the action rows.

FPS labels are the real rates times the source's speed factor (EgoVerse 0.5 slows
human motion to half speed in model time); ``fps_action == fps_video * K / 4`` holds
by construction, which the temporal-causal packer requires for ``K != 4``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from .action import Action57Builder
from .dataset import (
    PROMPT_MODE_SEGMENT_ONLY,
    PROMPT_MODES,
    CosmosActionPromptFormatter,
    CosmosTextTokenCounter,
    EgoVerseCosmosDataset,
    build_prompt_text,
    decode_rgb_video,
)
from .temporal import SPATIAL_TOKENS_PER_LATENT_FRAME

VAE_TEMPORAL_COMPRESSION = 4
DEFAULT_CLIP_FRAME_TIERS = (129, 65, 33)
DEFAULT_SPEED_FACTORS = {"egoverse": 0.5}
AR_MAX_SEQUENCE_LENGTH = 75_000


def clip_span(clip_frames: int, frame_stride: int) -> int:
    """Source frames covered by a clip of ``clip_frames`` video frames."""
    return frame_stride * (clip_frames - 1) + 1


def select_clip_frames(source_frames: int, frame_stride: int, tiers=DEFAULT_CLIP_FRAME_TIERS) -> int | None:
    """Largest tier whose span fits in the segment, or None when the segment is too short."""
    for frames in sorted(tiers, reverse=True):
        if (frames - 1) % VAE_TEMPORAL_COMPRESSION:
            raise ValueError(f"clip tiers must satisfy T = 4n + 1, got {frames}")
        if clip_span(frames, frame_stride) <= source_frames:
            return frames
    return None


def ar_latent_frames(clip_frames: int) -> int:
    return 1 + (clip_frames - 1) // VAE_TEMPORAL_COMPRESSION


def ar_token_count(text_tokens: int, clip_frames: int, action_tokens_per_latent: int) -> int:
    """Packed length of one pass: text, generation specials, video patches and action tokens."""
    latents = ar_latent_frames(clip_frames)
    return int(text_tokens + 1 + SPATIAL_TOKENS_PER_LATENT_FRAME * latents + 2 + latents * action_tokens_per_latent)


def ar_fps_labels(source_fps: float, frame_stride: int, speed_factor: float) -> tuple[float, float]:
    """(video, action) FPS labels: real rates times the speed factor."""
    if source_fps <= 0 or frame_stride < 1 or speed_factor <= 0:
        raise ValueError("source_fps, frame_stride and speed_factor must be positive")
    return float(source_fps) / frame_stride * speed_factor, float(source_fps) * speed_factor


def expand_state_group(per_frame: torch.Tensor, tokens_per_latent: int) -> torch.Tensor:
    """``[1 + (L-1)K, ...]`` source-frame rows -> ``[L*K, ...]`` with row 0 repeated K times."""
    if (per_frame.shape[0] - 1) % tokens_per_latent:
        raise ValueError(f"{per_frame.shape[0]} rows are not 1 + a multiple of K={tokens_per_latent}")
    repeat = [tokens_per_latent] + [1] * (per_frame.ndim - 1)
    return torch.cat([per_frame[:1].repeat(*repeat), per_frame[1:]], dim=0)


class EgoVerseARSegmentDataset(Dataset):
    """Map-style manifest dataset producing unpadded 57D AR clips."""

    def __init__(
        self,
        episodes_manifest: str | Path,
        segments_manifest: str | Path,
        *,
        split: str = "train",
        frame_stride: int = 2,
        clip_frame_tiers: tuple[int, ...] = DEFAULT_CLIP_FRAME_TIERS,
        source_name: str = "egoverse",
        speed_factors: dict[str, float] | None = None,
        random_window: bool = True,
        token_counter: Callable[[str], int] | None = None,
        prompt_formatter: Callable[[str, int, float], str] | None = None,
        action_builder: Action57Builder | None = None,
        max_sequence_length: int = AR_MAX_SEQUENCE_LENGTH,
        prompt_mode: str = PROMPT_MODE_SEGMENT_ONLY,
    ):
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"unsupported prompt mode {prompt_mode!r}; expected one of {PROMPT_MODES}")
        speed_factors = dict(DEFAULT_SPEED_FACTORS if speed_factors is None else speed_factors)
        if source_name not in speed_factors:
            raise ValueError(f"no speed factor configured for source {source_name!r}")
        with Path(episodes_manifest).open(newline="", encoding="utf-8") as handle:
            episodes = {row["episode_hash"]: row for row in csv.DictReader(handle) if row["split"] == split}
        with Path(segments_manifest).open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split and row["episode_hash"] in episodes]
        if not rows:
            raise ValueError(f"no {split!r} segments found")
        self.episodes = episodes
        self.frame_stride = int(frame_stride)
        self.tokens_per_latent = VAE_TEMPORAL_COMPRESSION * self.frame_stride
        self.clip_frame_tiers = tuple(int(t) for t in clip_frame_tiers)
        self.speed_factor = float(speed_factors[source_name])
        self.random_window = bool(random_window)
        self.token_counter = token_counter or CosmosTextTokenCounter()
        self.prompt_formatter = prompt_formatter or CosmosActionPromptFormatter()
        self.action_builder = action_builder or Action57Builder(rigid_pose_frame_delta=True)
        self.max_sequence_length = int(max_sequence_length)
        self.prompt_mode = prompt_mode
        self.tier_counts = {str(t): 0 for t in sorted(self.clip_frame_tiers, reverse=True)} | {"dropped": 0}
        self.dropped_segments = []
        self.rows = []
        for row in rows:
            plan = self._build_clip_plan(row, episodes[row["episode_hash"]])
            if plan is None:
                continue
            row.update(plan)
            self.rows.append(row)
            self.tier_counts[str(row["_clip_frames"])] += 1
        if not self.rows:
            raise ValueError(f"no {split!r} segment is long enough for the smallest clip tier")

    def _build_clip_plan(self, row: dict, episode: dict) -> dict | None:
        sample_id = f"{row['episode_hash']}:{row['span_index']}:{row['start_idx']}:{row['end_idx']}"
        source_frames = int(row["end_idx"]) - int(row["start_idx"])
        frames = select_clip_frames(source_frames, self.frame_stride, self.clip_frame_tiers)
        if frames is None:
            self._drop(sample_id, source_frames, "shorter_than_smallest_clip_tier")
            return None
        fps_video, fps_action = ar_fps_labels(float(episode["fps"]), self.frame_stride, self.speed_factor)
        caption = build_prompt_text(row["text_normalized"], episode["task_description"], mode=self.prompt_mode)
        prompt = self.prompt_formatter(caption, frames, fps_video)
        if ar_token_count(self.token_counter(prompt), frames, self.tokens_per_latent) >= self.max_sequence_length:
            self._drop(sample_id, source_frames, "exceeds_packing_cap")
            return None
        return {
            "_clip_frames": frames,
            "_clip_span": clip_span(frames, self.frame_stride),
            "_fps_video": fps_video,
            "_fps_action": fps_action,
            "_structured_prompt": prompt,
        }

    def _drop(self, sample_id: str, source_frames: int, reason: str) -> None:
        self.tier_counts["dropped"] += 1
        self.dropped_segments.append({"sample_id": sample_id, "source_frames": source_frames, "reason": reason})

    def __len__(self) -> int:
        return len(self.rows)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return [(index, 1) for index in range(len(self))]

    def window_start(self, row: dict) -> int:
        """Source index of the clip's first frame (random offset in training, fixed at 0 otherwise)."""
        start, end = int(row["start_idx"]), int(row["end_idx"])
        slack = (end - start) - int(row["_clip_span"])
        return start + (random.randint(0, slack) if self.random_window and slack > 0 else 0)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        episode = self.episodes[row["episode_hash"]]
        frames, span, stride = int(row["_clip_frames"]), int(row["_clip_span"]), self.frame_stride
        first = self.window_start(row)
        action_indexes = first + np.arange(span, dtype=np.int64)
        video_indexes = action_indexes[::stride]
        group = zarr.open_group(episode["abs_zarr_path"], mode="r")
        if action_indexes[-1] >= min(int(row["end_idx"]), int(group.attrs["total_frames"])):
            raise IndexError("clip window exceeds the manifest segment or episode")

        def read(name: str, dtype=np.float64):
            return np.asarray(group[name][action_indexes], dtype=dtype)

        per_frame_action = self.action_builder.build(
            head_pose=read("obs_head_pose"),
            right_wrist_pose=read("right.obs_wrist_pose"),
            left_wrist_pose=read("left.obs_wrist_pose"),
            right_keypoints=read("right.obs_keypoints"),
            left_keypoints=read("left.obs_keypoints"),
        )  # [span,57]
        def visibility_of(side: str) -> np.ndarray:
            name = f"{side}.obs_palm_in_fov_front_1"
            if name in group:
                return read(name, np.uint8)
            # Episodes outside the materialized training subsets: same projection, computed on the fly.
            from .materialize_visibility import compute_visibility

            return compute_visibility(group, int(action_indexes[-1]) + 1, side)[action_indexes]

        per_frame_visibility = torch.from_numpy(
            np.stack((visibility_of("right"), visibility_of("left")), axis=-1).astype(np.bool_)
        )  # [span,2]
        video = decode_rgb_video(group["images.front_1"][video_indexes])
        action = expand_state_group(per_frame_action, self.tokens_per_latent)
        visibility = expand_state_group(per_frame_visibility, self.tokens_per_latent)
        rows = ar_latent_frames(frames) * self.tokens_per_latent
        assert video.shape == (3, frames, 368, 640)
        assert action.shape == (rows, 57) and visibility.shape == (rows, 2)
        return {
            "ai_caption": json.loads(row["_structured_prompt"]),
            "video": video,
            "action": action,
            "hand_visibility": visibility,
            "conditioning_fps": float(row["_fps_video"]),
            "conditioning_fps_action": float(row["_fps_action"]),
            "mode": "wam",
            "domain_id": torch.tensor(3, dtype=torch.long),
            "viewpoint": "ego_view",
            "sample_id": f"{row['episode_hash']}:{row['span_index']}:{row['start_idx']}:{row['end_idx']}",
            "clip_frames": frames,
            "source_frame_indices": torch.from_numpy(video_indexes.copy()),
            "action_source_frame_indices": torch.from_numpy(action_indexes.copy()),
        }


def get_egoverse_ar_dataset(
    *,
    episodes_manifest: str,
    segments_manifest: str,
    tokenizer_config: dict,
    cfg_dropout_rate: float = 0.1,
    iterable_shuffle: bool = True,
    seed: int = 42,
    max_sequence_length: int = AR_MAX_SEQUENCE_LENGTH,
    prompt_mode: str = PROMPT_MODE_SEGMENT_ONLY,
    state_normalizer: str | None = None,
    future_normalizer: str | None = None,
    frame_stride: int = 2,
    clip_frame_tiers: tuple[int, ...] = DEFAULT_CLIP_FRAME_TIERS,
    speed_factors: dict[str, float] | None = None,
    random_window: bool = True,
    split: str = "train",
):
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

    builder_kwargs = {"rigid_pose_frame_delta": True}
    if state_normalizer is not None:
        builder_kwargs["state_normalizer"] = state_normalizer
    if future_normalizer is not None:
        builder_kwargs["future_normalizer"] = future_normalizer
    raw = EgoVerseARSegmentDataset(
        episodes_manifest,
        segments_manifest,
        split=split,
        frame_stride=frame_stride,
        clip_frame_tiers=tuple(clip_frame_tiers),
        speed_factors=None if speed_factors is None else dict(speed_factors),
        random_window=random_window,
        action_builder=Action57Builder(**builder_kwargs),
        max_sequence_length=max_sequence_length,
        prompt_mode=prompt_mode,
    )
    transform = ActionTransformPipeline(
        pad_keys=[],
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=64,
        action_channel_masking=True,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        # The adapter already applied Cosmos' JSON formatter with the real canvas and slowed FPS.
        format_prompt_as_json=False,
    )
    dataset = EgoVerseCosmosDataset(raw, transform)
    return ActionIterableShuffleDataset(dataset, seed=seed) if iterable_shuffle else dataset
