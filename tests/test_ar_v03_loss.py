import os

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_v03_loss import (
    AR_V03_ACTION_CHANNEL_WEIGHTS,
    whole_action_flow_loss,
)
from cosmos3_joint_video_hand_pose.src.loss import (
    ACTION_SUBBLOCKS,
    whole_action_flow_loss as v02_loss,
)


def inputs(width=64, dtype=torch.float32, device="cpu"):
    gen = torch.Generator().manual_seed(53)
    prediction = [
        torch.randn(n, width, generator=gen).to(device=device, dtype=dtype).requires_grad_()
        for n in (5, 9, 2)
    ]
    target = [torch.randn(p.shape, generator=gen).to(p) for p in prediction]
    conditions = [torch.tensor([1, 0, 0, 0, 0], device=device),
                  torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 0], device=device),
                  torch.ones(2, device=device)]
    visible = [torch.tensor([[1, 0]] * len(p), device=device) for p in prediction]
    valid = [torch.ones_like(p, dtype=torch.bool) for p in prediction]
    valid[0][2, :4] = False
    valid[1][-1] = False
    return dict(
        pred=prediction, target=target, condition_mask=conditions, visibility=visible,
        valid_mask=valid, mask_out_of_fov=False, collect_field_metrics=True,
    )


def assert_metrics_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if isinstance(actual[name], dict):
            assert_metrics_equal(actual[name], expected[name])
        else:
            assert torch.equal(actual[name], expected[name]), name


@pytest.mark.parametrize("width", [57, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_one_weights_are_bitwise_equal_to_v02_including_gradients(width, dtype):
    kwargs = inputs(width, dtype)
    old, old_metrics = v02_loss(**kwargs)
    new, new_metrics = whole_action_flow_loss(channel_weights=torch.ones(57), **kwargs)
    assert torch.equal(old, new)
    assert_metrics_equal(new_metrics, old_metrics)
    old_grad = torch.autograd.grad(old, kwargs["pred"], retain_graph=True)
    new_grad = torch.autograd.grad(new, kwargs["pred"])
    assert all(torch.equal(a, b) for a, b in zip(old_grad, new_grad, strict=True))


def test_wrist_objective_weights_denominator_and_keeps_raw_field_logs():
    p = torch.ones(3, 64, requires_grad=True)
    with torch.no_grad():
        p[:, 9:18] = 2
        p[:, 33:42] = 4
        p[0] = 3e38
        p[:, 57:] = 3e38
    kwargs = dict(
        pred=[p], target=[torch.zeros_like(p)],
        condition_mask=[torch.tensor([1, 0, 0])],
        visibility=[torch.zeros(3, 2)], mask_out_of_fov=False,
        collect_field_metrics=True,
    )
    old, old_metrics = v02_loss(**kwargs)
    loss, metrics = whole_action_flow_loss(
        channel_weights=AR_V03_ACTION_CHANNEL_WEIGHTS, **kwargs
    )
    expected = torch.tensor((39 + 27 * 4 + 27 * 16) / 93)
    torch.testing.assert_close(loss, expected)
    assert metrics["valid_coordinates"].tolist() == [114]
    assert metrics["active_samples"].tolist() == [True]
    assert_metrics_equal(metrics["field_per_sample_losses"], old_metrics["field_per_sample_losses"])
    assert set(metrics["field_per_sample_losses"]) == set(ACTION_SUBBLOCKS)
    assert all(not x.requires_grad for x in metrics["field_per_sample_losses"].values())
    loss.backward()
    assert torch.isfinite(p.grad).all()
    assert torch.count_nonzero(p.grad[0]) == 0
    assert torch.count_nonzero(p.grad[:, 57:]) == 0
    torch.testing.assert_close(p.grad[1, 0], torch.tensor(1 / 93))
    torch.testing.assert_close(p.grad[1, 9], torch.tensor(6 / 93))
    assert not torch.equal(loss, old)


@pytest.mark.parametrize("shape", [(57,), (64,), (1, 57), (1, 64), (2, 57), (2, 64)])
def test_weighted_native_masks_and_empty_samples(shape):
    p = torch.ones(2, 64, requires_grad=True)
    valid = torch.ones(shape)
    valid[..., 9:18] = 0
    loss, metrics = whole_action_flow_loss(
        pred=[p, p], target=[torch.zeros_like(p)] * 2,
        condition_mask=[torch.zeros(2), torch.ones(2)],
        visibility=[torch.ones(2, 2)] * 2, valid_mask=[valid, valid],
        channel_weights=AR_V03_ACTION_CHANNEL_WEIGHTS,
    )
    assert loss.item() == 1
    assert metrics["active_samples"].tolist() == [True, False]
    assert metrics["valid_coordinates"].tolist() == [96, 0]
    loss.backward()
    assert torch.count_nonzero(p.grad[:, 9:18]) == 0


def test_small_weights_use_actual_denominator_and_weights_are_detached():
    p = torch.full((1, 57), 2.0, requires_grad=True)
    weights = torch.full((57,), 0.001, requires_grad=True)
    loss, _ = whole_action_flow_loss(
        pred=[p], target=[torch.zeros_like(p)], condition_mask=[torch.zeros(1)],
        visibility=[torch.ones(1, 2)], channel_weights=weights,
    )
    torch.testing.assert_close(loss, torch.tensor(4.0))
    loss.backward()
    assert weights.grad is None


@pytest.mark.parametrize("weights", [
    [1.0] * 56, [1.0] * 58, [0.0] * 57,
    [-1.0] + [1.0] * 56, [float("nan")] + [1.0] * 56,
    [float("inf")] + [1.0] * 56,
])
def test_invalid_channel_weights_fail(weights):
    with pytest.raises(ValueError, match="57 finite positive"):
        whole_action_flow_loss(channel_weights=weights, **inputs())


@pytest.mark.skipif(os.environ.get("RUN_CUDA_TESTS") != "1", reason="requires explicit reserved GPU")
def test_gpu_all_one_and_weighted_raw_fields():
    kwargs = inputs(device="cuda")
    old, old_metrics = v02_loss(**kwargs)
    one, one_metrics = whole_action_flow_loss(channel_weights=torch.ones(57), **kwargs)
    assert torch.equal(old, one)
    assert_metrics_equal(one_metrics, old_metrics)
    _, weighted = whole_action_flow_loss(
        channel_weights=AR_V03_ACTION_CHANNEL_WEIGHTS, **kwargs
    )
    assert_metrics_equal(weighted["field_per_sample_losses"], old_metrics["field_per_sample_losses"])
