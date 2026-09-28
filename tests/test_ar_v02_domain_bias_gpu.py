"""Repeated embodiment bias gradients must accumulate before bf16 rounding."""

import pytest
import torch
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_stream_resolves_unindexed_cuda_device():
    from test_ar_v02_streaming import PerfectModel, normalizers
    from cosmos3_joint_video_hand_pose.src.ar_v02_streaming import StreamingJointSampler

    model = PerfectModel()
    model.tensor_kwargs["device"] = "cuda"
    state, future = normalizers()
    sampler = StreamingJointSampler(
        model, text_ids=[3, 4, 5], latent_shape=(4, 2, 2), state_normalizer=state, future_normalizer=future
    )
    assert sampler.device == torch.device("cuda", torch.cuda.current_device())
    sampler._check(torch.zeros_like(sampler._condition), sampler._condition.shape, "condition")


@pytest.mark.parametrize("count", [512, 4096])
def test_domain_bias_gradient_matches_fp32_sum_with_zero_gradient_rows(count):
    torch.manual_seed(512 + count)
    layer = DomainAwareLinear(16, 32, num_domains=3).cuda().bfloat16()
    x = torch.randn(count, 16, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(count, device="cuda") % 3
    upstream = torch.randn(count, 32, device="cuda", dtype=torch.bfloat16)
    expected = torch.zeros(3, 32, device="cuda").index_add(0, ids, upstream.float()).bfloat16()
    for pad in (False, True):
        if pad:
            inputs = x.repeat_interleave(2, 0)
            domains = ids.repeat_interleave(2)
            gradient = torch.zeros(count * 2, 32, device="cuda", dtype=torch.bfloat16)
            gradient[::2] = upstream
        else:
            inputs, domains, gradient = x, ids, upstream
        result = layer(inputs, domains)
        (got,) = torch.autograd.grad(result, layer.bias.weight, gradient)
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
