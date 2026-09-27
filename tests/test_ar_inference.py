"""CPU tests for AR inference helpers."""

import torch

from cosmos3_joint_video_hand_pose.src.ar_dataset import expand_state_group
from cosmos3_joint_video_hand_pose.src.ar_inference import (
    ARSampler,
    chunk_frame_ranges,
    collapse_state_group,
    flow_sigmas,
)


def test_chunk_ranges_match_training_partition():
    assert chunk_frame_ranges(9, 4) == [(1, 5), (5, 9)]
    assert chunk_frame_ranges(33, 4) == [(1 + 4 * i, 5 + 4 * i) for i in range(8)]
    assert chunk_frame_ranges(6, 4) == [(1, 5), (5, 6)]
    assert chunk_frame_ranges(5, 1) == [(1, 2), (2, 3), (3, 4), (4, 5)]


def test_flow_sigmas_are_monotone_and_shifted():
    sigmas = flow_sigmas(10, 5.0)
    assert sigmas.shape == (11,)
    assert float(sigmas[0]) == 1.0 and float(sigmas[-1]) == 0.0
    assert torch.all(sigmas[1:] < sigmas[:-1])
    assert float(flow_sigmas(2, 1.0)[1]) == 0.5
    assert float(flow_sigmas(2, 5.0)[1]) > 0.5


def test_collapse_inverts_expand():
    per_frame = torch.randn(1 + 3 * 8, 5)
    assert torch.equal(collapse_state_group(expand_state_group(per_frame, 8), 8), per_frame)


def _stub_sampler(num_frames=9, chunk_size=4, k=8, dim=3):
    sampler = object.__new__(ARSampler)
    sampler.chunk_size, sampler.tokens_per_latent, sampler.num_frames = chunk_size, k, num_frames
    sampler.raw_action_dim = None
    sampler.gt_video = torch.full((1, 2, num_frames, 2, 2), 100.0)
    sampler.gt_action = torch.full((num_frames * k, dim), 100.0)
    seen = []

    def forward(video, action, first_noisy, end, frame_sigmas, clean_video=None, clean_action=None):
        seen.append((video[:, :, :first_noisy].clone(), action[: first_noisy * k].clone()))
        return torch.ones_like(video), torch.ones_like(action)  # constant velocity, never GT

    sampler.forward = forward
    return sampler, seen


def test_gt_history_keeps_every_chunk_prediction():
    sampler, seen = _stub_sampler()
    video, action = sampler.sample(2, 2, 1.0, 1.0, history="gt", seed=0)
    # History fed to the model is ground truth (teacher forcing) ...
    for hist_video, hist_action in seen:
        assert torch.all(hist_video == 100.0) and torch.all(hist_action == 100.0)
    # ... but every generated chunk in the output is a prediction, not GT.
    assert torch.all(video[:, :, 0] == 100.0) and torch.all(action[:8] == 100.0)
    for start, end in chunk_frame_ranges(9, 4):
        assert not torch.any(video[:, :, start:end] == 100.0)
        assert not torch.any(action[start * 8 : end * 8] == 100.0)


def test_generated_history_feeds_predictions_back():
    sampler, seen = _stub_sampler()
    video, action = sampler.sample(2, 2, 1.0, 1.0, history="generated", seed=0)
    last_hist_video, last_hist_action = seen[-1]
    assert torch.equal(last_hist_video[:, :, 1:5], video[:, :, 1:5])
    assert torch.equal(last_hist_action[8:40], action[8:40])


def test_oracle_actions_read_gt_current_video_but_return_generated_video():
    sampler, _ = _stub_sampler()
    current_is_gt = []

    def forward(video, action, first_noisy, end, frame_sigmas, clean_video=None, clean_action=None):
        current_is_gt.append(bool(torch.all(video[:, :, first_noisy:end] == 100.0)))
        return torch.ones_like(video), torch.ones_like(action)

    sampler.forward = forward
    video, action = sampler.sample(2, 2, 1.0, 1.0, history="oracle", seed=0)
    # Per chunk: two video solver steps on the generated chunk, then two action steps on GT video.
    assert current_is_gt == [False, False, True, True] * 2
    for start, end in chunk_frame_ranges(9, 4):
        assert not torch.any(video[:, :, start:end] == 100.0)
        assert not torch.any(action[start * 8 : end * 8] == 100.0)
