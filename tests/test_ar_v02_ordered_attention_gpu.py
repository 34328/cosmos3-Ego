"""Padding must not reorder bf16 online-softmax reductions for real queries."""

import pytest
import torch
from cosmos3_joint_video_hand_pose.src.ar_v02_attention import create_joint_block_mask
from cosmos3_joint_video_hand_pose.src.ar_attention import _create_block_mask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.no_grad()
def test_full_and_partial_tiles_have_one_order_even_when_padding_queries_change():
    from cosmos_framework.model.generator.mot.flex_attention import _COMPILED_FLEX_ATTENTION

    torch.manual_seed(123)
    q = torch.randn(1, 4, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 768, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    outputs = []
    for compact in (False, True):

        def predicate(b, h, qi, ki):
            condition = (ki < 137) | ((ki >= 256) & (ki < 497))
            # Real condition queries see precisely the same keys in both layouts.
            return torch.where(qi < 241, condition, (ki >= 497) if compact else ki < 768)

        args = dict(B=None, H=None, Q_LEN=256 if compact else 512, KV_LEN=768, device="cuda", BLOCK_SIZE=128)
        stock = _create_block_mask(False)(predicate, **args)
        ordered = create_joint_block_mask(predicate, **args)
        assert ordered.full_kv_num_blocks is None
        torch.testing.assert_close(ordered.to_dense(), stock.to_dense(), atol=0, rtol=0)
        for counts, indices in zip(
            ordered.kv_num_blocks.flatten(), ordered.kv_indices.reshape(-1, ordered.kv_indices.shape[-1])
        ):
            retained = indices[: int(counts)]
            assert bool((retained[1:] > retained[:-1]).all())
        out = _COMPILED_FLEX_ATTENTION(q[:, :, : args["Q_LEN"]].contiguous(), k, v, block_mask=ordered, enable_gqa=True)
        outputs.append(out[:, :, :241])
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
