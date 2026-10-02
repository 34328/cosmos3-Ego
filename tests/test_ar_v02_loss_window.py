"""Whole-modality loss reduction and globally counted update windows."""

from datetime import timedelta
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cosmos3_joint_video_hand_pose.src.ar_v02_contract import GlobalSampleMeanWindow
from cosmos3_joint_video_hand_pose.src.loss import whole_video_flow_loss


def test_video_condition_and_sigma_weight_alignment():
    p = torch.ones(2, 3, 1, 1, requires_grad=True)
    p.data[:, 0] = 1e30
    loss, stats = whole_video_flow_loss(
        pred=[p],
        target=[torch.zeros_like(p)],
        condition_mask=[torch.tensor([1, 0, 0])],
        time_weight=lambda *args: torch.tensor([9.0, 2.0, 4.0]),
    )
    torch.testing.assert_close(loss, torch.tensor(3.0))
    torch.testing.assert_close(stats["per_sample_losses"], torch.tensor([3.0]))
    loss.backward()
    assert torch.count_nonzero(p.grad[:, 0]) == 0
    torch.testing.assert_close(p.grad[:, 1], torch.ones(2, 1, 1))
    torch.testing.assert_close(p.grad[:, 2], torch.full((2, 1, 1), 2.0))


def test_window_uses_update_counts_not_average_of_microbatch_means():
    p = torch.tensor(0.4, requires_grad=True)
    window = GlobalSampleMeanWindow([[1, 0], [3, 2]])
    losses = []
    for vx, ax in (([1.0], []), ([2.0, 3.0, 4.0], [7.0, 8.0])):
        v = (p * torch.tensor(vx)).square()
        a = (p * torch.tensor(ax)).square()
        loss, _ = window.reduce(v, torch.ones(len(v), dtype=torch.bool), a, torch.ones(len(a), dtype=torch.bool))
        losses.append(loss / 2)  # Cosmos trainer division
    grad = torch.autograd.grad(sum(losses), p)[0]
    ref = (p * torch.tensor([1.0, 2.0, 3.0, 4.0])).square().mean() + 1.0 * (
        p * torch.tensor([7.0, 8.0])
    ).square().mean()
    torch.testing.assert_close(sum(losses), ref)
    torch.testing.assert_close(grad, torch.autograd.grad(ref, p)[0])
    assert window.complete
    with pytest.raises(RuntimeError, match="consumed"):
        window.reduce(v, v.bool(), a, a.bool())


def test_empty_modalities_are_differentiable_and_plans_checked():
    p = torch.tensor(3.0, requires_grad=True)
    w = GlobalSampleMeanWindow([[0, 0]])
    loss, _ = w.reduce(p[None] * 0, torch.tensor([False]), p[None] * 0, torch.tensor([False]))
    loss.backward()
    assert p.grad == 0
    w = GlobalSampleMeanWindow([[1, 0]])
    with pytest.raises(ValueError, match="planned"):
        w.reduce(p[None], torch.tensor([False]), p[None], torch.tensor([False]))
    assert w.position == 0


class TwoHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.4))
        self.video = torch.nn.Parameter(torch.tensor(0.2))
        self.action = torch.nn.Parameter(torch.tensor(-0.1))

    def forward(self, v, a):
        return (self.shared * v + self.video).square(), (self.shared * a + self.action).square()


BATCHES = [
    [([1.0], [5.0, 7.0]), ([2.0, 3.0], [8.0]), ([], [9.0])],
    [([4.0, 5.0, 6.0], []), ([7.0], []), ([8.0, 9.0], [])],
]


def _distributed_worker(rank, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        from contextlib import nullcontext

        model = DDP(TwoHead())
        counts = [[len(v), len(a)] for v, a in BATCHES[rank]]
        w = GlobalSampleMeanWindow(counts)
        report = torch.zeros(2)
        for i, (vx, ax) in enumerate(BATCHES[rank]):
            # Keep both heads connected, including empty modalities.
            with model.no_sync() if i < 2 else nullcontext():
                v, a = model(torch.tensor(vx), torch.tensor(ax))
                loss, stats = w.reduce(v, torch.ones(len(v), dtype=torch.bool), a, torch.ones(len(a), dtype=torch.bool))
                (loss / 3).backward()
                report += torch.stack((stats["video_contribution"], stats["action_contribution"]))
        ref = TwoHead()
        all_v = torch.tensor([x for rank_batches in BATCHES for v, a in rank_batches for x in v])
        all_a = torch.tensor([x for rank_batches in BATCHES for v, a in rank_batches for x in a])
        rv, ra = ref(all_v, all_a)
        (rv.mean() + 1.0 * ra.mean()).backward()
        for actual, expected in zip(model.module.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-6)
        dist.all_reduce(report)
        torch.testing.assert_close(report, torch.stack((rv.mean(), ra.mean())).detach())
        # Different window lengths fail identically on both ranks before backward.
        with pytest.raises(ValueError, match="same number"):
            GlobalSampleMeanWindow([[1, 1]] * (rank + 1))
    finally:
        dist.destroy_process_group()


def test_two_rank_dynamic_pack_accumulation_matches_single_process(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_mixin_bypasses_native_scaling_and_passes_coordinate_validity():
    from cosmos3_joint_video_hand_pose.src.model import EgoVerseLossMixin

    class DummyParent(torch.nn.Module):
        def _compute_losses(self, **kwargs):
            raise AssertionError("native loss scaling must not run")

        def _sample_level_loss_scale(self, **kwargs):
            raise AssertionError("native sample scaler must not run")

        def _loss_averaging_group(self):
            return None, 1

        def training_step(self, data, iteration):
            loss, logs = self._compute_losses(**data)
            return logs, loss

    class Model(EgoVerseLossMixin, DummyParent):
        def __init__(self):
            torch.nn.Module.__init__(self)

    model = Model()
    model.whole_action_loss = True
    model.config = SimpleNamespace(
        vision_gen=True,
        action_gen=True,
        rectified_flow_training_config=SimpleNamespace(
            sample_level_loss_averaging=True, loss_scale=1.0, action_loss_weight=1.0, image_loss_scale=None
        ),
    )
    model.rectified_flow_video = SimpleNamespace(
        train_time_weight=lambda t, kw: torch.ones_like(t),
        noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
    )
    model.tensor_kwargs_fp32 = dict(device="cpu", dtype=torch.float32)
    model._current_hand_visibility = [torch.ones(2, 2)] * 2
    v = [torch.ones(1, 2, 1, 1, requires_grad=True)] * 2
    a = [torch.full((2, 64), x, requires_grad=True) for x in (2.0, 100.0)]
    masks = [torch.ones(64, dtype=torch.bool), torch.zeros(64, dtype=torch.bool)]
    packed = SimpleNamespace(
        sample_lens=[1, 1],
        vision=SimpleNamespace(condition_mask=[torch.tensor([1, 0])] * 2),
        action=SimpleNamespace(
            raw_action_dim=[57, 57], condition_mask=[torch.tensor([1, 0])] * 2, action_valid_mask=masks
        ),
    )
    data = dict(
        out_net={"preds_vision": v, "preds_action": a},
        data_batch_packed=packed,
        gen_data_noised=SimpleNamespace(
            vt_target_vision=[torch.zeros_like(x) for x in v], vt_target_action=[torch.zeros_like(x) for x in a]
        ),
        timesteps=torch.tensor([[0.0, 100.0], [0.0, 200.0]]),
        is_image_batch=False,
    )
    output, loss = model.training_step(data, 0)
    torch.testing.assert_close(loss, torch.tensor(5.0))
    torch.testing.assert_close(output["_backward_loss"], loss)
    for key, expected in (("train_objective_numerator", loss.detach()),
                          ("train_objective_denominator", torch.ones_like(loss))):
        assert not output[key].requires_grad
        assert output[key].grad_fn is None
        assert output[key].shape == loss.shape
        assert output[key].dtype == loss.dtype
        assert output[key].device == loss.device
        torch.testing.assert_close(output[key], expected)
    # The logger's detached metadata must not change the training objective or
    # its gradient; compare with the existing two-term objective independently.
    from cosmos3_joint_video_hand_pose.src.loss import whole_action_flow_loss
    reference_video, _ = whole_video_flow_loss(
        pred=v, target=data["gen_data_noised"].vt_target_vision,
        condition_mask=packed.vision.condition_mask,
        time_weight=lambda *args: torch.ones(2),
    )
    reference_action, _ = whole_action_flow_loss(
        pred=a, target=data["gen_data_noised"].vt_target_action,
        condition_mask=packed.action.condition_mask,
        visibility=model._current_hand_visibility, valid_mask=masks,
    )
    reference = reference_video + reference_action
    torch.testing.assert_close(loss, reference)
    expected_grads = torch.autograd.grad(reference, [v[0], *a])
    output["_backward_loss"].backward()
    for tensor, expected in zip([v[0], *a], expected_grads, strict=True):
        torch.testing.assert_close(tensor.grad, expected)
    assert torch.count_nonzero(a[1].grad) == 0
    assert output["egoverse_global_action_samples"] == 1
    model.configure_ar_loss_accumulation(2)
    with pytest.raises(RuntimeError, match="begin_ar_loss_window"):
        model.training_step(data, 1)
    model.begin_ar_loss_window([[2, 1], [2, 1]], device="cpu")
    for i in range(2):
        output, loss = model.training_step(data, i)
        torch.testing.assert_close(output["_backward_loss"] / 2, loss)
        torch.testing.assert_close(output["train_objective_numerator"], loss.detach())
        torch.testing.assert_close(output["train_objective_denominator"], torch.ones_like(loss))
        assert not output["train_objective_numerator"].requires_grad
        assert not output["train_objective_denominator"].requires_grad
    assert model._ar_loss_window.complete
