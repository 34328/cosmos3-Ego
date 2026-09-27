"""CPU tests for the AR clip layout (frame_stride=2, K=8, speed factor 0.5)."""

import json
from pathlib import Path
import random

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_dataset import (
    EgoVerseARSegmentDataset,
    ar_fps_labels,
    ar_latent_frames,
    ar_token_count,
    clip_span,
    expand_state_group,
    select_clip_frames,
)

SUBSET = Path(__file__).resolve().parents[1] / (
    "cosmos3_joint_video_hand_pose/artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1"
)


def test_clip_tiers_follow_segment_length():
    assert clip_span(129, 2) == 257 and clip_span(65, 2) == 129 and clip_span(33, 2) == 65
    assert select_clip_frames(257, 2) == 129
    assert select_clip_frames(256, 2) == 65
    assert select_clip_frames(129, 2) == 65
    assert select_clip_frames(128, 2) == 33
    assert select_clip_frames(65, 2) == 33
    assert select_clip_frames(64, 2) is None
    with pytest.raises(ValueError):
        select_clip_frames(300, 2, tiers=(30,))


def test_group_layout_matches_design():
    # T=33 video frames -> 9 latents; 65 source frames -> 1 state row + 64 action rows.
    assert ar_latent_frames(33) == 9
    per_frame = torch.arange(65).float()[:, None]
    grouped = expand_state_group(per_frame, 8)
    assert grouped.shape == (72, 1)
    assert torch.all(grouped[:8] == 0)
    # Group j >= 1 holds source frames 8(j-1)+1 .. 8j; its last row is latent j's last video frame (source 8j).
    for j in range(1, 9):
        assert grouped[8 * j : 8 * (j + 1), 0].tolist() == list(range(8 * (j - 1) + 1, 8 * j + 1))
    with pytest.raises(ValueError):
        expand_state_group(torch.zeros(10, 2), 8)


def test_fps_labels_are_slowed_and_consistent_with_k():
    video, action = ar_fps_labels(30.0, 2, 0.5)
    assert (video, action) == (7.5, 15.0)
    assert action == pytest.approx(video * 8 / 4)
    assert ar_fps_labels(30.0, 2, 1.0) == (15.0, 30.0)


def test_token_count_includes_k_action_tokens_per_latent():
    assert ar_token_count(100, 33, 8) == 100 + 1 + 240 * 9 + 2 + 9 * 8


def test_manifest_plan_on_subset():
    dataset = EgoVerseARSegmentDataset(
        SUBSET / "episodes.csv",
        SUBSET / "segments.csv",
        token_counter=lambda prompt: 200,
        prompt_formatter=lambda caption, frames, fps: json.dumps({"frames": frames, "fps": fps}),
        action_builder=object(),
        prompt_mode="episode_context_and_segment",
    )
    counts = dataset.tier_counts
    assert sum(counts.values()) == 181
    assert counts["dropped"] == 181 - len(dataset)
    for row in dataset.rows:
        source = int(row["end_idx"]) - int(row["start_idx"])
        assert row["_clip_frames"] == select_clip_frames(source, 2)
        assert (row["_fps_video"], row["_fps_action"]) == (7.5, 15.0)
        assert json.loads(row["_structured_prompt"]) == {"frames": row["_clip_frames"], "fps": 7.5}
    # Windows stay inside the segment; evaluation uses the segment start.
    random.seed(0)
    for row in dataset.rows[:20]:
        first = dataset.window_start(row)
        assert int(row["start_idx"]) <= first and first + row["_clip_span"] <= int(row["end_idx"])
    dataset.random_window = False
    assert all(dataset.window_start(row) == int(row["start_idx"]) for row in dataset.rows[:20])
