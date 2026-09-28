import pytest
import torch

from cosmos3_joint_video_hand_pose.src.loss import whole_action_flow_loss


def test_denominator_counts_valid_coordinates_not_mean_of_blocks():
    pred = torch.zeros(3, 64, requires_grad=True)
    with torch.no_grad():
        pred[:, :9] = 1
        pred[:, 9:33] = 2
        pred[:, 33:57] = 3
        pred[:, 57:] = 999
    loss, metrics = whole_action_flow_loss(
        pred=[pred],
        target=[torch.zeros_like(pred)],
        condition_mask=[torch.tensor([1, 0, 0])],
        visibility=[torch.tensor([[1, 1], [1, 1], [0, 1]])],
    )
    expected = (18 + 24 * 4 + 48 * 9) / (18 + 24 + 48)
    torch.testing.assert_close(loss, torch.tensor(expected))
    assert metrics["valid_coordinates"].tolist() == [90]
    loss.backward()
    assert torch.count_nonzero(pred.grad[0]) == 0
    assert torch.count_nonzero(pred.grad[2, 9:33]) == 0
    assert torch.count_nonzero(pred.grad[:, 57:]) == 0
    torch.testing.assert_close(pred.grad[1, :9], torch.full((9,), 2 / 90))


def test_masked_target_changes_do_not_change_loss_or_prediction_gradient():
    pred = torch.randn(4, 64, generator=torch.Generator().manual_seed(3), requires_grad=True)
    target = torch.zeros_like(pred)
    valid = torch.ones_like(pred, dtype=torch.bool)
    valid[3] = False  # structural tail padding
    valid[1, 2] = False  # one invalid coordinate
    condition = torch.tensor([1, 0, 0, 0])
    visibility = torch.tensor([[1, 1], [1, 1], [0, 1], [1, 1]])
    kwargs = dict(pred=[pred], condition_mask=[condition], visibility=[visibility], valid_mask=[valid])
    baseline, _ = whole_action_flow_loss(target=[target], **kwargs)
    gradient = torch.autograd.grad(baseline, pred)[0]
    target[0] = 10
    target[2, 9:33] = -10
    target[:, 57:] = 100
    target[3] = 200
    target[1, 2] = -300
    changed, _ = whole_action_flow_loss(target=[target], **kwargs)
    torch.testing.assert_close(changed, baseline)
    torch.testing.assert_close(torch.autograd.grad(changed, pred)[0], gradient)


def test_sample_average_excludes_empty_sample_and_does_not_weight_by_length():
    pred = [torch.full((n, 57), value, requires_grad=True) for n, value in [(2, 1.0), (10, 3.0), (4, 100.0)]]
    loss, metrics = whole_action_flow_loss(
        pred=pred,
        target=[torch.zeros_like(p) for p in pred],
        condition_mask=[torch.zeros(2), torch.zeros(10), torch.ones(4)],
        visibility=[torch.ones(len(p), 2) for p in pred],
    )
    torch.testing.assert_close(loss, torch.tensor(5.0))
    assert metrics["active_samples"].tolist() == [True, True, False]
    loss.backward()
    assert torch.count_nonzero(pred[2].grad) == 0


def test_empty_modality_returns_differentiable_zero():
    pred = torch.ones(2, 64, requires_grad=True)
    loss, metrics = whole_action_flow_loss(
        pred=[pred],
        target=[torch.zeros_like(pred)],
        condition_mask=[torch.ones(2)],
        visibility=[torch.ones(2, 2)],
    )
    assert loss.item() == 0 and not metrics["active_samples"].any()
    loss.backward()
    assert torch.count_nonzero(pred.grad) == 0


def test_nan_in_masked_position_is_not_hidden():
    target = torch.zeros(2, 64)
    target[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        whole_action_flow_loss(
            pred=[torch.zeros_like(target)],
            target=[target],
            condition_mask=[torch.ones(2)],
            visibility=[torch.zeros(2, 2)],
        )


@pytest.mark.parametrize("shape", [(57,), (64,), (1, 57), (1, 64), (2, 57), (2, 64)])
def test_native_channel_and_per_coordinate_masks(shape):
    p = torch.ones(2, 64, requires_grad=True)
    valid = torch.ones(shape)
    valid[..., 0] = 0
    loss, stats = whole_action_flow_loss(
        pred=[p],
        target=[torch.zeros_like(p)],
        condition_mask=[torch.zeros(2)],
        visibility=[torch.ones(2, 2)],
        valid_mask=[valid],
    )
    assert stats["valid_coordinates"].item() == 112
    loss.backward()
    assert torch.count_nonzero(p.grad[:, 0]) == 0


def test_masked_finite_extremes_do_not_overflow_gradients():
    p = torch.ones(2, 64, requires_grad=True)
    target = torch.zeros_like(p)
    p.data[0] = 3e38
    target[0] = -3e38
    loss, _ = whole_action_flow_loss(
        pred=[p], target=[target], condition_mask=[torch.tensor([1, 0])], visibility=[torch.ones(2, 2)]
    )
    loss.backward()
    assert torch.isfinite(p.grad).all()
    assert loss.item() == 1
    assert torch.count_nonzero(p.grad[0]) == 0
