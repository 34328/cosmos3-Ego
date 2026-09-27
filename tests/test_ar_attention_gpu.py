"""GPU check: FlexAttention lingbot override equals a dense masked reference (values and gradients)."""

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_attention import (
    ARChunkLayout,
    LingbotTeacherForcingAttention,
    make_mask_mod,
    token_block_ids,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels need CUDA")


def _memory_value(noisy, text_len, clean_k=None, clean_v=None):
    from cosmos_framework.model.generator.utils.kv_cache import TFNoisyMemoryValue, TFReplayCleanMemoryValue

    device = torch.device("cuda")
    dummy = torch.zeros(1, 1, 1, 1, device=device)
    common = dict(
        vision_token_shapes=[(1, 1, 1)],
        num_action_tokens_per_supertoken=0,
        has_new_caption=torch.tensor(True, device=device),
        has_caption=torch.tensor(text_len > 0, device=device),
        has_cached_gen=torch.tensor(False, device=device),
        und_kv_offsets=torch.tensor([0, max(text_len, 1)], dtype=torch.int32, device=device),
        gen_q_offsets=torch.tensor([0, 1], dtype=torch.int32, device=device),
        gen_ca_cached_kv_offsets=torch.tensor([0, 1], dtype=torch.int32, device=device),
        cached_und_k=dummy,
        cached_und_v=dummy,
        cached_gen_k=dummy,
        cached_gen_v=dummy,
        max_gen_cache_tokens=1,
        clamp_empty_varlen_kv=True,
    )
    if noisy:
        return TFNoisyMemoryValue(**common, cached_clean_gen_k=clean_k, cached_clean_gen_v=clean_v)
    return TFReplayCleanMemoryValue(**common)


def _reference(q, k, v, text_k, text_v, clean_k, clean_v, layout, text_len, noisy):
    """Dense fp32 attention with the same predicate evaluated on the unpadded streams."""
    n = layout.num_tokens
    heads, kv_heads = q.shape[-2], k.shape[-2]
    text_pad = text_k.shape[1]
    mask_mod = make_mask_mod(
        token_block_ids(layout).to(q.device),
        gen_pad_len=n,
        text_pad_len=text_pad,
        text_len=text_len,
        noisy=noisy,
        window=layout.window,
    )
    keys = [text_k, k.reshape(1, n, kv_heads, -1)] + ([clean_k] if noisy else [])
    values = [text_v, v.reshape(1, n, kv_heads, -1)] + ([clean_v] if noisy else [])
    key = torch.cat(keys, dim=1).float().repeat_interleave(heads // kv_heads, dim=2)  # [1,KV,H,D]
    value = torch.cat(values, dim=1).float().repeat_interleave(heads // kv_heads, dim=2)
    query = q.reshape(1, n, heads, -1).float()
    allowed = mask_mod(0, 0, torch.arange(n, device=q.device)[:, None], torch.arange(key.shape[1], device=q.device)[None])
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / query.shape[-1] ** 0.5
    scores = scores.masked_fill(~allowed, float("-inf"))
    out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), value)
    return out.reshape(q.shape)


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("window", [None, 2])
def test_flex_override_matches_dense_reference(noisy, window):
    torch.manual_seed(0)
    layout = ARChunkLayout(num_frames=4, action_tokens=8, vision_tokens=40, chunk_size=2, window=window)
    heads, kv_heads, dim, text_len, text_rows = 8, 2, 64, 11, 13
    device, dtype = torch.device("cuda"), torch.bfloat16
    shape_q = (1, layout.num_frames, layout.tokens_per_frame, heads, dim)
    shape_kv = (1, layout.num_frames, layout.tokens_per_frame, kv_heads, dim)

    def leaf(*shape):
        return torch.randn(*shape, device=device, dtype=dtype).requires_grad_()

    q, k, v = leaf(*shape_q), leaf(*shape_kv), leaf(*shape_kv)
    text_k, text_v = leaf(1, text_rows, kv_heads, dim), leaf(1, text_rows, kv_heads, dim)
    clean_k, clean_v = leaf(1, layout.num_tokens, kv_heads, dim), leaf(1, layout.num_tokens, kv_heads, dim)
    memory = _memory_value(noisy, text_len, clean_k, clean_v)
    attention = LingbotTeacherForcingAttention(layout, device)

    out = attention(q, k, v, text_k, text_v, memory)
    ref = _reference(q, k, v, text_k, text_v, clean_k, clean_v, layout, text_len, noisy)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    grad = torch.randn_like(ref)
    leaves = [q, k, v, text_k, text_v] + ([clean_k, clean_v] if noisy else [])
    flex_grads = torch.autograd.grad(out.float(), leaves, grad)
    ref_grads = torch.autograd.grad(ref, leaves, grad)
    for name, got, want in zip(["q", "k", "v", "text_k", "text_v", "clean_k", "clean_v"], flex_grads, ref_grads):
        torch.testing.assert_close(got.float(), want.float(), atol=5e-2, rtol=5e-2, msg=lambda m: f"{name}: {m}")
