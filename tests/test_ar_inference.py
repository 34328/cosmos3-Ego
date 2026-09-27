"""CPU tests for AR inference helpers."""

import torch

from cosmos3_joint_video_hand_pose.src.ar_dataset import expand_state_group
from cosmos3_joint_video_hand_pose.src.ar_inference import chunk_frame_ranges, collapse_state_group, flow_sigmas


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
