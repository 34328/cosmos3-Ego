"""Actual compiled FlexAttention values, history gradients, and causal isolation."""

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, VIDEO, JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v03_attention import JointDiffusionForcingAttention
from test_ar_v03_attention import (
    assert_history_gradient_paths, assert_values_and_gradients, memory_value, ownership,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_single_pass_values_and_gradients(dtype):
    assert_values_and_gradients("cuda", dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_noisy_history_has_both_gradient_paths(dtype):
    assert_history_gradient_paths("cuda", dtype)


@torch.no_grad()
def test_future_other_sample_and_evicted_history_cannot_change_current_prediction():
    torch.manual_seed(221)
    layouts = [JointChunkLayout(18, 2, 1), JointChunkLayout(4, 1, 1)]
    roles, chunks, samples = ownership(layouts, "cuda")
    n, dim = len(roles), 32
    q, k, v = [torch.randn(1, 1, n, 2, dim, device="cuda") for _ in range(3)]
    text_k, text_v = [torch.randn(1, 7, 2, dim, device="cuda") for _ in range(2)]
    attention = JointDiffusionForcingAttention(layouts, "cuda", text_lengths=[3, 4])
    run = lambda ak, av, tk, tv: attention(q, ak, av, tk, tv, memory_value("cuda", 7))
    base = run(k, v, text_k, text_v)
    # Chunk 16 still sees chunk 1; chunk 17 has evicted it at H=15.
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, :, (samples == 0) & (chunks == 1)] += 7
    changed_v[:, :, (samples == 0) & (chunks == 1)] -= 5
    changed = run(changed_k, changed_v, text_k, text_v)
    target = (roles == VIDEO) | (roles == ACTION)
    q16 = (samples == 0) & (chunks == 16) & target
    q17 = (samples == 0) & (chunks == 17) & target
    assert (changed[:, :, q16] - base[:, :, q16]).abs().max() > 1e-3
    torch.testing.assert_close(changed[:, :, q17], base[:, :, q17], atol=0, rtol=0)
    forbidden = (samples != 0) | ((samples == 0) & (chunks > 2))
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, :, forbidden] += 13
    changed_v[:, :, forbidden] -= 11
    tk, tv = text_k.clone(), text_v.clone()
    tk[:, 3:] += 9
    tv[:, 3:] -= 7
    query = (samples == 0) & (chunks == 2)
    torch.testing.assert_close(run(changed_k, changed_v, tk, tv)[:, :, query],
                               base[:, :, query], atol=0, rtol=0)
