"""CPU mask and real eager FlexAttention contracts for joint diffusion forcing."""

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_v02_attention import JointTeacherForcingAttention
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import (
    ACTION, CONDITION_VIDEO, STATE, VIDEO, JointChunkLayout,
)
from cosmos3_joint_video_hand_pose.src.ar_v03_attention import JointDiffusionForcingAttention


def memory_value(device, text_len):
    from cosmos_framework.model.generator.utils.kv_cache import KVTrainMemoryValue

    dummy = torch.zeros(1, 1, 1, 1, device=device)
    return KVTrainMemoryValue(
        vision_token_shapes=[(1, 1, 1)], num_action_tokens_per_supertoken=0,
        has_new_caption=torch.tensor(text_len > 0, device=device),
        has_caption=torch.tensor(text_len > 0, device=device),
        has_cached_gen=torch.tensor(False, device=device),
        und_kv_offsets=torch.tensor([0, max(text_len, 1)], device=device, dtype=torch.int32),
        gen_q_offsets=torch.tensor([0, 1], device=device, dtype=torch.int32),
        gen_ca_cached_kv_offsets=torch.tensor([0, 1], device=device, dtype=torch.int32),
        cached_und_k=dummy, cached_und_v=dummy, cached_gen_k=dummy, cached_gen_v=dummy,
        max_gen_cache_tokens=1, clamp_empty_varlen_kv=True,
    )


def ownership(layouts, device):
    metadata = [x.metadata(device=device) for x in layouts]
    return (
        torch.cat([x[0] for x in metadata]), torch.cat([x[1] for x in metadata]),
        torch.cat([torch.full((x.num_tokens,), i, device=device) for i, x in enumerate(layouts)]),
    )


def reference_mask(layouts, text_lengths, text_capacity, device):
    """Independent specification, without production predicates or block masks."""
    roles, chunks, samples = ownership(layouts, device)
    condition = (roles == STATE) | (roles == CONDITION_VIDEO)
    allowed = torch.zeros(len(roles), text_capacity + len(roles), device=device, dtype=torch.bool)
    text_start = 0
    for sample, length in enumerate(text_lengths):
        allowed[samples == sample, text_start:text_start + length] = True
        text_start += length
    for query in range(len(roles)):
        same_sample = samples == samples[query]
        same_chunk = chunks == chunks[query]
        if condition[query]:
            visible = same_chunk & condition
        else:
            visible = (chunks <= chunks[query]) & (chunks >= chunks[query] - 15)
        allowed[query, text_capacity:] = same_sample & visible
    return allowed


def dense_reference(q, k, v, text_k, text_v, allowed):
    n, heads, kv_heads = q.shape[2], q.shape[-2], k.shape[-2]
    keys = torch.cat([text_k, k.reshape(1, n, kv_heads, -1)], 1).float().repeat_interleave(heads // kv_heads, 2)
    values = torch.cat([text_v, v.reshape(1, n, kv_heads, -1)], 1).float().repeat_interleave(heads // kv_heads, 2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.reshape(1, n, heads, -1).float(), keys) / q.shape[-1] ** 0.5
    return torch.einsum("bhqk,bkhd->bqhd", scores.masked_fill(~allowed, -torch.inf).softmax(-1), values).reshape_as(q)


@pytest.fixture
def eager_cpu_flex(monkeypatch):
    import cosmos_framework.model.generator.mot.flex_attention as kernels
    from torch.nn.attention.flex_attention import flex_attention

    # Exercise production packing/mask and torch's real CPU FlexAttention/autograd.
    # Fused compiled Triton values/backward are covered separately on the GPU.
    monkeypatch.setattr(kernels, "_COMPILED_FLEX_ATTENTION", flex_attention)


@pytest.mark.parametrize("c", [1, 4])
@pytest.mark.parametrize("text_lengths", [[3, 5], [0, 0]])
def test_single_pass_mask_exactly_matches_v02_clean_pass(c, text_lengths):
    layouts = [JointChunkLayout(1 + 16 * c + 1, 2, c), JointChunkLayout(c + 2, 1, c)]
    actual = JointDiffusionForcingAttention(layouts, "cpu", text_lengths=text_lengths)
    old = JointTeacherForcingAttention(layouts, "cpu", text_lengths=text_lengths)
    # V0.2's all-empty text owner list has no entries; use its implicit single
    # owner fallback since no text keys are visible in this equivalence case.
    if sum(text_lengths) == 0:
        old.text_sample_ids = None
    nt = sum(text_lengths)
    mask = actual.block_mask(text_pad_len=128, text_len=nt)
    clean = old.block_mask(noisy=False, text_pad_len=128, text_len=nt)
    q = torch.arange(actual.gen_pad_len)[:, None]
    k = torch.arange(128 + actual.gen_pad_len)[None]
    allowed = mask.mask_mod(0, 0, q, k)
    torch.testing.assert_close(allowed, clean.mask_mod(0, 0, q, k), atol=0, rtol=0)
    torch.testing.assert_close(mask.to_dense(), clean.to_dense(), atol=0, rtol=0)
    expected = reference_mask(layouts, text_lengths, 128, "cpu")
    torch.testing.assert_close(allowed[:actual.gen_len, :128 + actual.gen_len], expected, atol=0, rtol=0)
    assert mask.full_kv_num_blocks is None
    assert actual.block_mask(text_pad_len=128, text_len=nt) is mask
    assert not allowed[:actual.gen_len, 128 + actual.gen_len:].any()
    if actual.gen_pad_len > actual.gen_len:
        assert allowed[actual.gen_len:].any(dim=1).all()
        assert not allowed[actual.gen_len:, :128 + actual.gen_len].any()


def assert_values_and_gradients(device, dtype=torch.float32):
    torch.manual_seed(42)
    layouts = [JointChunkLayout(6, 2, 4), JointChunkLayout(4, 1, 4)]
    n, nt, heads, kv_heads, dim = sum(x.num_tokens for x in layouts), 8, 4, 2, 32
    shapes = [(1, 1, n, heads, dim), (1, 1, n, kv_heads, dim), (1, 1, n, kv_heads, dim),
              (1, nt + 3, kv_heads, dim), (1, nt + 3, kv_heads, dim)]
    leaves = [torch.randn(s, device=device, dtype=dtype, requires_grad=True) for s in shapes]
    refs = [x.detach().float().requires_grad_() for x in leaves]
    actual = JointDiffusionForcingAttention(layouts, device, text_lengths=[3, 5])(
        *leaves, memory_value(device, nt),
    )
    expected = dense_reference(*refs, reference_mask(layouts, [3, 5], nt + 3, device))
    upstream = torch.randn_like(actual)
    got = torch.autograd.grad(actual, leaves, upstream)
    want = torch.autograd.grad(expected, refs, upstream.float())
    atol, rtol = (2e-5, 1e-4) if dtype == torch.float32 else (2e-2, 3e-2)
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
    for ag, eg in zip(got, want):
        assert torch.isfinite(ag).all()
        torch.testing.assert_close(ag.float(), eg, atol=atol, rtol=rtol)
    assert got[3][:, nt:].count_nonzero() == got[4][:, nt:].count_nonzero() == 0


def test_cpu_single_pass_values_and_gradients(eager_cpu_flex):
    assert_values_and_gradients("cpu")


def assert_history_gradient_paths(device, dtype=torch.float32):
    """Two real attention layers: noisy block k supplies own and later losses."""
    torch.manual_seed(171)
    layouts = [JointChunkLayout(14, 2, 4), JointChunkLayout(6, 1, 4)]
    roles, chunks, samples = ownership(layouts, device)
    n, width, heads, dim = len(roles), 64, 2, 32
    hidden = torch.randn(n, width, device=device, dtype=dtype, requires_grad=True)
    attention = JointDiffusionForcingAttention(layouts, device, text_lengths=[3, 2])
    text_k, text_v = [torch.randn(1, 5, 1, dim, device=device, dtype=dtype) for _ in range(2)]
    layer_weights = [[torch.randn(out, width, device=device, dtype=dtype) / width ** 0.5
                      for out in (heads * dim, dim, dim)] for _ in range(2)]

    def forward():
        state = hidden
        for weights in layer_weights:
            # Tokenwise learned projections preserve the actual Q/K/V -> hidden path.
            q, k, v = [torch.nn.functional.linear(state, w).reshape(1, 1, n, h, dim)
                       for w, h in zip(weights, (heads, 1, 1))]
            state = attention(q, k, v, text_k, text_v, memory_value(device, 5)).reshape(n, width)
        return state

    target_roles = (roles == VIDEO) | (roles == ACTION)
    block_k = (samples == 0) & (chunks == 2) & target_roles
    block_next = (samples == 0) & (chunks == 3) & target_roles
    # Rebuild the same deterministic graph for each decomposition. This exercises
    # torch's production donated-buffer backward instead of disabling it just to
    # use retain_graph, which the compiled runtime explicitly forbids.
    own_grad = torch.autograd.grad(forward()[block_k].float().square().mean(), hidden)[0]
    later_grad = torch.autograd.grad(forward()[block_next].float().square().mean(), hidden)[0]
    state = forward()
    total_grad = torch.autograd.grad(state[block_k].float().square().mean()
                                     + state[block_next].float().square().mean(), hidden)[0]
    metrics = {}
    for name, role in (("video", VIDEO), ("action", ACTION)):
        path = block_k & (roles == role)
        metrics[name] = {"self_grad_norm": own_grad[path].float().norm().item(),
                         "later_grad_norm": later_grad[path].float().norm().item()}
        assert torch.isfinite(own_grad[path]).all() and torch.isfinite(later_grad[path]).all()
        assert metrics[name]["self_grad_norm"] > 0 and metrics[name]["later_grad_norm"] > 0
    tolerance = dict(atol=2e-7, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-4, rtol=3e-2)
    torch.testing.assert_close(total_grad.float(), (own_grad + later_grad).float(), **tolerance)
    # Future and other-sample input hidden states cannot affect an earlier loss.
    assert own_grad[(samples != 0) | (chunks > 2)].count_nonzero() == 0
    assert later_grad[(samples != 0) | (chunks > 3)].count_nonzero() == 0
    current_condition = (samples == 0) & (chunks == 3) & ~target_roles
    condition_loss = forward()[current_condition].float().square().mean()
    condition_grad = torch.autograd.grad(condition_loss, hidden)[0]
    assert condition_grad[target_roles].count_nonzero() == 0
    assert condition_grad[current_condition].float().norm() > 0
    import json
    print(json.dumps({"device": str(device), "dtype": str(dtype), "block_k": 2,
                      "history_gradient_paths": metrics}), flush=True)


def test_cpu_noisy_block_gets_self_and_later_loss_gradients(eager_cpu_flex):
    assert_history_gradient_paths("cpu")


def test_packed_samples_require_text_ownership():
    with pytest.raises(ValueError, match="text_lengths"):
        JointDiffusionForcingAttention([JointChunkLayout(3, 1, 4)] * 2, "cpu")
    with pytest.raises(ValueError, match="nonnegative"):
        JointDiffusionForcingAttention(JointChunkLayout(3, 1, 4), "cpu", text_lengths=[-1])
