"""Scheme A: wrist-local futures use tracking validity, not palm FOV."""
from types import SimpleNamespace

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.action_representation import FIXED_CAMERA, LEGACY
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout, STATE
from cosmos3_joint_video_hand_pose.src.loss import whole_action_flow_loss
from cosmos3_joint_video_hand_pose.src.model import EgoVerseLossMixin


@pytest.mark.parametrize("groups", [1, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("all_invisible", [False, True])
def test_future_numerator_denominator_and_gradients_with_real_chunk_layout(groups, all_invisible):
    # C=4, including 1/2/3-group partial tails. Every original future row survives.
    layout = JointChunkLayout(groups + 1, 1, 4)
    future = torch.zeros(groups * 8, 64)
    future[:, :9], future[:, 9:33], future[:, 33:57] = 1, 2, 3
    visibility = torch.zeros(len(future), 2, dtype=torch.bool)
    if not all_invisible:
        visibility[::2, 0] = True
        visibility[::3, 1] = True
    saved_visibility = visibility.clone()
    assembled, packed_visibility = layout.assemble_action(future, torch.zeros(groups, 64), visibility)
    roles, _, sources = layout.action_metadata()
    condition = roles == STATE
    pred = assembled.detach().requires_grad_()
    kwargs = dict(pred=[pred], target=[torch.zeros_like(pred)],
                  condition_mask=[condition], visibility=[packed_visibility])
    loss, stats = whole_action_flow_loss(**kwargs, mask_out_of_fov=False)
    count = len(future) * 57
    numerator = len(future) * (9 + 24 * 4 + 24 * 9)
    assert stats["valid_coordinates"].item() == count
    torch.testing.assert_close(loss, torch.tensor(numerator / count))
    torch.testing.assert_close(loss * stats["valid_coordinates"][0], torch.tensor(float(numerator)))
    grad = torch.autograd.grad(loss, pred)[0]
    torch.testing.assert_close(grad[~condition, :57], 2 * future[:, :57] / count)
    assert torch.count_nonzero(grad[condition]) == 0
    assert torch.count_nonzero(grad[:, 57:]) == 0
    assert sources[~condition].tolist() == list(range(1, len(future) + 1))
    assert torch.equal(visibility, saved_visibility)

    # Legacy default retains hand-side FOV masking in BOTH numerator and denominator.
    legacy, old = whole_action_flow_loss(**kwargs)
    old_count = len(future) * 9 + int(visibility.sum()) * 24
    old_sum = len(future) * 9 + int(visibility[:, 0].sum()) * 24 * 4 + int(visibility[:, 1].sum()) * 24 * 9
    assert old["valid_coordinates"].item() == old_count
    torch.testing.assert_close(legacy, torch.tensor(old_sum / old_count))
    old_grad = torch.autograd.grad(legacy, pred)[0][~condition]
    assert torch.count_nonzero(old_grad[~visibility[:, 0], 9:33]) == 0
    assert torch.count_nonzero(old_grad[~visibility[:, 1], 33:57]) == 0

    # Visibility changes cannot affect the new objective or its gradient.
    changed, _ = whole_action_flow_loss(**(kwargs | {"visibility": [~packed_visibility]}),
                                       mask_out_of_fov=False)
    torch.testing.assert_close(changed, loss)
    torch.testing.assert_close(torch.autograd.grad(changed, pred)[0], grad)


def test_scheme_a_still_excludes_structural_padding_and_rejects_corrupt_targets():
    pred = torch.full((3, 64), 2.0, requires_grad=True)
    valid = torch.ones_like(pred, dtype=torch.bool)
    valid[-1] = False  # Structural padding, not an invalid tracked frame.
    target = torch.zeros_like(pred)
    kwargs = dict(pred=[pred], target=[target], condition_mask=[torch.tensor([1, 0, 0])],
                  visibility=[torch.zeros(3, 2)], valid_mask=[valid], mask_out_of_fov=False)
    loss, stats = whole_action_flow_loss(**kwargs)
    assert stats["valid_coordinates"].item() == 57
    assert loss.item() == 4
    loss.backward()
    assert torch.count_nonzero(pred.grad[[0, 2]]) == 0
    assert torch.count_nonzero(pred.grad[:, 57:]) == 0
    target[1, 9] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        whole_action_flow_loss(**kwargs)


class LossHarness(EgoVerseLossMixin, torch.nn.Module):
    def __init__(self, representation):
        torch.nn.Module.__init__(self)
        if representation is not None:
            self.action_representation = representation
        self.whole_action_loss = True
        self.config = SimpleNamespace(
            vision_gen=True, action_gen=True,
            rectified_flow_training_config=SimpleNamespace(loss_scale=1.0, action_loss_weight=1.0),
        )
        self.rectified_flow_video = SimpleNamespace(train_time_weight=lambda t, kw: torch.ones_like(t))
        self.tensor_kwargs_fp32 = dict(device="cpu", dtype=torch.float32)

    def _loss_averaging_group(self):
        return None, 1


@pytest.mark.parametrize("representation", [FIXED_CAMERA, LEGACY, None])
@pytest.mark.parametrize("entry", ["whole", "flow"])
def test_actual_model_loss_entries_dispatch_only_new_representation(representation, entry):
    model = LossHarness(representation)
    p = torch.zeros(2, 64, requires_grad=True)
    with torch.no_grad():
        p[:, :9], p[:, 9:33], p[:, 33:57] = 1, 2, 3
    visibility = torch.zeros(2, 2, dtype=torch.bool)
    model._current_hand_visibility = [visibility]
    condition = [torch.tensor([1, 0])]
    if entry == "whole":
        video = torch.zeros(1, 2, 1, 1, requires_grad=True)
        packed = SimpleNamespace(sample_lens=[1],
            vision=SimpleNamespace(condition_mask=condition),
            action=SimpleNamespace(raw_action_dim=[57], condition_mask=condition, action_valid_mask=None))
        loss, metrics = model._compute_whole_losses(
            {"preds_action": [p], "preds_vision": [video]}, packed,
            SimpleNamespace(vt_target_action=[torch.zeros_like(p)], vt_target_vision=[torch.zeros_like(video)]),
            torch.zeros(1, 2), False)
        torch.testing.assert_close(loss, metrics["flow_matching_loss_action"])
        assert metrics["egoverse_global_action_samples"].item() == 1
    else:
        loss, _ = model._compute_flow_matching_loss(
            [p], [torch.zeros_like(p)], condition, torch.zeros(1, 2), True,
            model.rectified_flow_video, raw_action_dim=[57])
        expected_count = 57 if representation == FIXED_CAMERA else 9
        assert model._last_visibility_loss_metrics["valid_coordinates"].item() == expected_count
    expected = (9 + 24 * 4 + 24 * 9) / 57 if representation == FIXED_CAMERA else 1.0
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()
    assert torch.count_nonzero(p.grad[1, 9:57]).item() == (48 if representation == FIXED_CAMERA else 0)
    assert torch.count_nonzero(p.grad[0]) == 0
    assert torch.count_nonzero(p.grad[:, 57:]) == 0
    assert not visibility.any()
