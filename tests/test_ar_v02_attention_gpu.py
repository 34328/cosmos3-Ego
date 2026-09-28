"""GPU-only FlexAttention values, gradients, and fixed-noisy-input leakage checks."""

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_v02_attention import JointTeacherForcingAttention
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, CONDITION_VIDEO, STATE, VIDEO, JointChunkLayout
from test_ar_attention_gpu import _memory_value

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; main agent schedules GPU runs")


def ownership(layouts, device):
    rows = [x.metadata(device=device) for x in layouts]
    return (
        torch.cat([x[0] for x in rows]),
        torch.cat([x[1] for x in rows]),
        torch.cat([torch.full((x.num_tokens,), i, device=device) for i, x in enumerate(layouts)]),
    )


def reference_mask(layouts, text_lengths, text_capacity, noisy, device):
    # Independent specification; do not call production joint_mask_mod.
    roles, chunks, samples = ownership(layouts, device)
    cond = (roles == STATE) | (roles == CONDITION_VIDEO)
    n = len(roles)
    allowed = torch.zeros(n, text_capacity + n * (2 if noisy else 1), device=device, dtype=torch.bool)
    start = 0
    for sample, length in enumerate(text_lengths):
        allowed[samples == sample, start : start + length] = True
        start += length
    for query in range(n):
        own = samples == samples[query]
        same = own & (chunks == chunks[query])
        past = own & (chunks < chunks[query]) & (chunks >= chunks[query] - 15)
        if cond[query]:
            allowed[query, text_capacity + (n if noisy else 0) : text_capacity + (2 * n if noisy else n)] = same & cond
        elif noisy:
            allowed[query, text_capacity : text_capacity + n] = same & ~cond
            allowed[query, text_capacity + n :] = past | (same & cond)
        else:
            allowed[query, text_capacity:] = past | same
    return allowed


def dense_reference(q, k, v, tk, tv, ck, cv, allowed, noisy):
    n, heads, kh = q.shape[2], q.shape[-2], k.shape[-2]
    key = (
        torch.cat([tk, k.reshape(1, n, kh, -1)] + ([ck] if noisy else []), 1).float().repeat_interleave(heads // kh, 2)
    )
    value = (
        torch.cat([tv, v.reshape(1, n, kh, -1)] + ([cv] if noisy else []), 1).float().repeat_interleave(heads // kh, 2)
    )
    scores = torch.einsum("bqhd,bkhd->bhqk", q.reshape(1, n, heads, -1).float(), key) / q.shape[-1] ** 0.5
    return torch.einsum("bhqk,bkhd->bqhd", scores.masked_fill(~allowed, -torch.inf).softmax(-1), value).reshape_as(q)


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("noisy", [False, True])
def test_multisample_flex_values_and_gradients(c, noisy):
    torch.manual_seed(42)
    layouts = [JointChunkLayout(1 + 16 * c + 1, 2, c), JointChunkLayout(c + 2, 1, c)]
    n, h, kh, d = sum(x.num_tokens for x in layouts), 4, 2, 32
    text_lengths = [3, 5]
    nt = sum(text_lengths)
    leaf = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float32, requires_grad=True)
    q, k, v = leaf(1, 1, n, h, d), leaf(1, 1, n, kh, d), leaf(1, 1, n, kh, d)
    tk, tv, ck, cv = leaf(1, nt + 3, kh, d), leaf(1, nt + 3, kh, d), leaf(1, n, kh, d), leaf(1, n, kh, d)
    memory = _memory_value(noisy, nt, ck, cv)
    out = JointTeacherForcingAttention(layouts, "cuda", text_lengths=text_lengths)(q, k, v, tk, tv, memory)
    allowed = reference_mask(layouts, text_lengths, nt + 3, noisy, q.device)
    reference = dense_reference(q, k, v, tk, tv, ck, cv, allowed, noisy)
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-4)
    assert torch.isfinite(out).all()
    leaves = [q, k, v, tk, tv] + ([ck, cv] if noisy else [])
    grad = torch.randn_like(out)
    got = torch.autograd.grad(out, leaves, grad)
    want = torch.autograd.grad(reference, leaves, grad)
    for name, actual, expected in zip(["q", "k", "v", "text_k", "text_v", "clean_k", "clean_v"], got, want):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4, msg=lambda m: f"{name}: {m}")
    assert got[3][:, nt:].count_nonzero() == got[4][:, nt:].count_nonzero() == 0


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@torch.no_grad()
def test_fixed_noisy_input_rejects_clean_answers_future_conditions_and_other_sample(c):
    torch.manual_seed(123)
    layouts = [JointChunkLayout(2 * c + 2, 2, c), JointChunkLayout(c + 2, 1, c)]
    roles, chunks, samples = ownership(layouts, "cuda")
    n, nt, d = len(roles), 7, 32
    q, k, v = [torch.randn(1, 1, n, 2, d, device="cuda") for _ in range(3)]
    tk, tv = [torch.randn(1, nt, 2, d, device="cuda") for _ in range(2)]
    ck, cv = [torch.randn(1, n, 2, d, device="cuda") for _ in range(2)]
    attn = JointTeacherForcingAttention(layouts, "cuda", text_lengths=[3, 4])
    run = lambda ak, av, at, au, ac, ad: attn(q, ak, av, at, au, _memory_value(True, nt, ac, ad))
    base = run(k, v, tk, tv, ck, cv)
    query = (samples == 0) & (chunks == 2)
    forbidden_clean = (samples != 0) | (chunks > 2) | ((chunks == 2) & ((roles == VIDEO) | (roles == ACTION)))
    ck2, cv2 = ck.clone(), cv.clone()
    ck2[:, forbidden_clean] += 17
    cv2[:, forbidden_clean] -= 19
    torch.testing.assert_close(run(k, v, tk, tv, ck2, cv2)[:, :, query], base[:, :, query], atol=1e-5, rtol=1e-4)
    k2, v2, tk2, tv2 = k.clone(), v.clone(), tk.clone(), tv.clone()
    k2[:, :, samples == 1] += 13
    v2[:, :, samples == 1] -= 11
    tk2[:, 3:] += 7
    tv2[:, 3:] -= 5
    torch.testing.assert_close(run(k2, v2, tk2, tv2, ck2, cv2)[:, :, query], base[:, :, query], atol=1e-5, rtol=1e-4)
    # Positive control: current U/S really affect predictions; an all-masked model cannot pass.
    cv3 = cv.clone()
    current_cond = (samples == 0) & (chunks == 2) & ((roles == CONDITION_VIDEO) | (roles == STATE))
    cv3[:, current_cond] += 2
    changed = run(k, v, tk, tv, ck, cv3)
    target = query & ((roles == VIDEO) | (roles == ACTION))
    assert (changed[:, :, target] - base[:, :, target]).abs().max() > 1e-3


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@torch.no_grad()
def test_bfloat16_flex_matches_fp32_reference(c):
    torch.manual_seed(71)
    layouts = [JointChunkLayout(c + 2, 1, c), JointChunkLayout(2 * c + 2, 1, c)]
    n, nt, d = sum(x.num_tokens for x in layouts), 7, 32
    make = lambda *s: torch.randn(*s, device="cuda", dtype=torch.bfloat16)
    q, k, v = [make(1, 1, n, 2, d) for _ in range(3)]
    tk, tv, ck, cv = make(1, nt, 2, d), make(1, nt, 2, d), make(1, n, 2, d), make(1, n, 2, d)
    actual = JointTeacherForcingAttention(layouts, "cuda", text_lengths=[3, 4])(
        q, k, v, tk, tv, _memory_value(True, nt, ck, cv)
    )
    expected = dense_reference(q, k, v, tk, tv, ck, cv, reference_mask(layouts, [3, 4], nt, True, "cuda"), True)
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=3e-2)
    assert torch.isfinite(actual).all()
    assert (actual.float() - expected).norm() / expected.norm().clamp_min(1e-6) <= 1e-2


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("backward_schedule", ["immediate", "reverse"])
def test_bfloat16_alternating_layout_gradients(c, backward_schedule):
    """Keep compiled Flex warm across masks; dense leaves stay FP32, not BF16."""
    import json
    from cosmos3_joint_video_hand_pose.src.ar_v02_inference import assert_numerically_close

    torch.manual_seed(170 + c)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        a = [JointChunkLayout(1 + 16 * c + 1, 2, c), JointChunkLayout(c + 2, 1, c)]
        b = [JointChunkLayout(2 * c + 2, 1, c), JointChunkLayout(c + 2, 2, c), JointChunkLayout(3 * c + 2, 1, c)]
        # A and A-swapped have identical tensor shapes but different text ownership.
        # Reverse backward also retains older graphs across later mask closures.
        cases = [
            (a, [257, 769], False),
            (b, [513, 129, 385], True),
            (b, [513, 129, 385], False),
            (a, [257, 769], True),
            (a, [769, 257], False),
            (a, [769, 257], True),
        ]
        pending = []

        def verify(item):
            ordinal, noisy, nt, actual, expected, leaves, refs, upstream = item
            context = f"C={c} schedule={backward_schedule} case={ordinal} noisy={noisy}"
            indexes = list(range(7 if noisy else 5))
            got = torch.autograd.grad(actual, [leaves[i] for i in indexes], upstream)
            want = torch.autograd.grad(expected, [refs[i] for i in indexes], upstream.float())
            metrics = {}
            for name, ag, eg in zip(("q", "k", "v", "text_k", "text_v", "clean_k", "clean_v"), got, want):
                error = ag.float() - eg
                metrics[name] = {
                    "max_abs": error.abs().max().item(),
                    "relative_l2": (error.norm() / eg.norm().clamp_min(1e-6)).item(),
                    "norm_ratio": (ag.float().norm() / eg.norm().clamp_min(1e-6)).item(),
                }
            print(json.dumps({"context": context, "gradients": metrics}), flush=True)
            assert_numerically_close(actual, expected, fp32=False, context=context + " forward")
            for name, ag, eg in zip(("q", "k", "v", "text_k", "text_v", "clean_k", "clean_v"), got, want):
                assert_numerically_close(ag, eg, fp32=False, context=context + " " + name)
            assert got[3][:, nt:].count_nonzero() == got[4][:, nt:].count_nonzero() == 0

        for ordinal, (layouts, text_lengths, noisy) in enumerate(cases):
            n, nt, h, kh, d = sum(x.num_tokens for x in layouts), sum(text_lengths), 4, 2, 32
            shapes = [
                (1, 1, n, h, d),
                (1, 1, n, kh, d),
                (1, 1, n, kh, d),
                (1, nt + 11, kh, d),
                (1, nt + 11, kh, d),
                (1, n, kh, d),
                (1, n, kh, d),
            ]
            leaves = [torch.randn(s, device="cuda", dtype=torch.bfloat16, requires_grad=True) for s in shapes]
            # Quantize inputs identically, then keep reference arithmetic and leaf
            # gradients FP32 so BF16 leaf rounding cannot conceal discrepancies.
            refs = [x.detach().float().requires_grad_() for x in leaves]
            q, k, v, tk, tv, ck, cv = leaves
            actual = JointTeacherForcingAttention(layouts, "cuda", text_lengths=text_lengths)(
                q, k, v, tk, tv, _memory_value(noisy, nt, ck, cv)
            )
            expected = dense_reference(*refs, reference_mask(layouts, text_lengths, nt + 11, noisy, "cuda"), noisy)
            upstream = torch.randn_like(actual)
            item = (ordinal, noisy, nt, actual, expected, leaves, refs, upstream)
            if backward_schedule == "immediate":
                verify(item)
            else:
                pending.append(item)
        for item in reversed(pending):
            verify(item)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
