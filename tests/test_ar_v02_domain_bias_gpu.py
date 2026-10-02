"""Repeated embodiment bias gradients must accumulate before bf16 rounding."""

import pytest
import torch
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")



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
