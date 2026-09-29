"""CPU checks for the native lifecycle and project-only logging extension."""
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos_framework.utils import callback as native
from cosmos3_joint_video_hand_pose.src import wandb_metrics as logging


def output(video=1.0, action=2.0, total=3.0):
    return dict(zip(logging.BASE_LOSS_METRIC_SOURCES.values(),
                    map(torch.tensor, (video, action, total)), strict=True))


def make_callback(tmp_path, monkeypatch, *, rank0=True, online=False, interval=1):
    monkeypatch.setattr(logging.distributed, "is_rank0", lambda: rank0)
    monkeypatch.setattr(logging.wandb, "run", object() if online else None)
    cb = logging.EgoVerseLossWandbCallback()
    cb.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)),
                                trainer=SimpleNamespace(logging_iter=interval))
    return cb


def read_rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "loss_metrics.jsonl").read_text().splitlines()]


def model_with_gradients():
    net = torch.nn.Module()
    for index, name in enumerate(("moe_gen", "time_embedder", "vae2llm", "llm2vae",
                                  "vision_condition_embed", "action2llm", "llm2action",
                                  "action_modality_embed", "action_state_embed")):
        net.register_parameter(name, torch.nn.Parameter(torch.ones(3)))
        getattr(net, name).grad = torch.full((3,), float(index + 1))
    return SimpleNamespace(net=net)


def test_native_lifecycle_is_inherited_and_timer_resets(tmp_path, monkeypatch):
    cb = logging.LossOnlyWandBCallback()
    cb.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=2))
    cb.trainer = SimpleNamespace(training_timer=Mock())
    cb.trainer.training_timer.compute_average_results.return_value = {"step": 0.1}
    init, finish, log = Mock(), Mock(), Mock()
    monkeypatch.setattr(native.wandb_util, "init_wandb", init)
    monkeypatch.setattr(native.wandb, "log", log)
    monkeypatch.setattr(native.wandb, "finish", finish)
    monkeypatch.setattr(native.distributed, "is_rank0", lambda: True)
    monkeypatch.setattr(native.dist, "all_reduce", lambda *a, **k: None)
    for hook in ("on_train_start", "on_training_step_end", "on_train_end",
                 "on_before_optimizer_step", "on_validation_end"):
        assert getattr(logging.LossOnlyWandBCallback, hook) is getattr(native.WandBCallback, hook)
    cb.on_train_start(None)
    for step in range(1, 5):
        cb.on_training_step_batch_end(None, {}, {
            "train_objective_numerator": torch.tensor(3.0),
            "train_objective_denominator": torch.tensor(1.0),
        }, torch.tensor(3.0), step - 1)
        cb.on_training_step_end(None, {}, {}, torch.tensor(3.0), step)
        assert native.wandb.log is log
    cb.on_train_end(None, 4)
    assert cb.trainer.training_timer.reset.call_count == 2
    assert cb._train_objective_numerator is None
    assert {call.kwargs["step"] for call in log.call_args_list} == {2, 4}
    init.assert_called_once()
    finish.assert_called_once()
    assert native.wandb.log is log


@pytest.mark.parametrize("online", [False, True])
def test_every_step_has_three_losses_and_resume_appends(tmp_path, monkeypatch, online):
    cb = make_callback(tmp_path, monkeypatch, online=online, interval=10)
    log, fsync = Mock(), Mock()
    monkeypatch.setattr(logging.wandb, "log", log)
    monkeypatch.setattr(logging.os, "fsync", fsync)
    for step in (1, 2):
        cb.on_training_step_end(None, {}, output(total=float(step)), torch.tensor(0.0), step)
        assert len(read_rows(tmp_path)) == step  # durable before train end
    prefix = (tmp_path / "loss_metrics.jsonl").read_bytes()
    resumed = make_callback(tmp_path, monkeypatch, online=online)
    resumed.on_train_start(None, 2)
    resumed.on_training_step_end(None, {}, output(), torch.tensor(0.0), 3)
    assert (tmp_path / "loss_metrics.jsonl").read_bytes().startswith(prefix)
    rows = read_rows(tmp_path)
    assert [row["step"] for row in rows] == [1, 2, 3]
    for row in rows:
        assert set(logging.BASE_LOSS_METRIC_SOURCES) <= row.keys()
        assert row["loss/video_raw"] == 1.0
        assert row["loss/action_raw"] == 2.0
        assert row["status"] == "ok"
    assert fsync.call_count == 3
    if online:
        assert [call.kwargs["step"] for call in log.call_args_list] == [1, 2, 3]
        assert all(call.kwargs["commit"] is False for call in log.call_args_list)
        for call in log.call_args_list:
            assert {"iteration", "train/loss", "train/loss_avg", "optim/lr", "optim/grad_scale"}.isdisjoint(call.args[0])
            assert not any(key.startswith("timer/") for key in call.args[0])
    else:
        log.assert_not_called()


def test_diagnostics_share_completed_loss_step_and_include_embeddings(tmp_path, monkeypatch):
    cb = make_callback(tmp_path, monkeypatch, online=True, interval=2)
    log = Mock()
    monkeypatch.setattr(logging.wandb, "log", log)
    model = model_with_gradients()
    scheduler = SimpleNamespace(get_last_lr=lambda: [1e-4, 5e-4])
    cb.on_before_optimizer_step(model, None, scheduler, None, iteration=0)
    cb.on_training_step_end(None, {}, output(), torch.tensor(3.0), 1)
    cb.on_before_optimizer_step(model, None, scheduler, None, iteration=1)
    assert log.call_count == 1  # pre-step hook must not advance the W&B step
    cb.on_training_step_end(None, {}, output(), torch.tensor(3.0), 2)
    row = read_rows(tmp_path)[1]
    assert row["grad_norm/video_projection_pre_clip"] == pytest.approx((3*(3**2+4**2+5**2))**0.5)
    assert row["grad_norm/action_projection_pre_clip"] == pytest.approx((3*(6**2+7**2+8**2+9**2))**0.5)
    assert row["grad_nonfinite_elements/selected_numel"] == 27
    assert row["optim/lr_base"] == 1e-4
    assert row["optim/lr_action"] == 5e-4
    assert log.call_args.kwargs["step"] == row["step"] == 2
    assert not cb._pending_diagnostics


@pytest.mark.parametrize("bad", [False, True])
def test_logging_does_not_modify_gradients_or_rng(tmp_path, monkeypatch, bad):
    cb = make_callback(tmp_path, monkeypatch)
    model = model_with_gradients()
    if bad:
        model.net.action_state_embed.grad[:] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    before = {name: p.grad.clone().view(torch.int32) for name, p in model.net.named_parameters()}
    rng = torch.get_rng_state().clone()
    cb.on_before_optimizer_step(model, None, SimpleNamespace(get_last_lr=lambda: [1e-4]), None, 0)
    for name, p in model.net.named_parameters():
        assert torch.equal(before[name], p.grad.view(torch.int32))
    assert torch.equal(rng, torch.get_rng_state())
    cb.on_training_step_end(None, {}, output(), torch.tensor(3.0), 1)
    row = read_rows(tmp_path)[0]
    assert row["grad_nonfinite/all_selected_present"] == float(bad)
    if bad:
        assert row["grad_norm/action_projection_pre_clip"] is None
        assert row["nonfinite_metrics"]["grad_norm/action_projection_pre_clip"] == "nan"
        assert row["grad_nonfinite_elements/nonfinite_count"] == 3
        trace = tmp_path / "grad_nonfinite_trace" / "rank_00000.jsonl"
        assert json.loads(trace.read_text())["fqn"] == "action_state_embed"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_is_explicit_and_never_successful_zero(tmp_path, monkeypatch, bad):
    cb = make_callback(tmp_path, monkeypatch, online=True)
    log = Mock()
    monkeypatch.setattr(logging.wandb, "log", log)
    cb.on_training_step_end(None, {}, output(total=bad), torch.tensor(bad), 1)
    row = read_rows(tmp_path)[0]
    assert row["status"] == "nonfinite_loss"
    assert row["loss/total"] is None
    assert "loss/total" in row["nonfinite_metrics"]
    metrics = log.call_args.args[0]
    assert "loss/total" not in metrics
    assert metrics["loss/nonfinite"] == 1
    assert "NaN" not in (tmp_path / "loss_metrics.jsonl").read_text()


def test_microbatches_are_accumulated_once_and_reset(tmp_path, monkeypatch):
    cb = make_callback(tmp_path, monkeypatch)
    for value in (2.0, 4.0):
        cb.on_training_step_batch_end(None, {}, output(total=value), torch.tensor(value), 0)
    cb.on_training_step_end(None, {}, output(total=4.0), torch.tensor(4.0), 1)
    cb.on_training_step_batch_end(None, {}, output(total=9.0), torch.tensor(9.0), 1)
    cb.on_training_step_end(None, {}, output(total=9.0), torch.tensor(9.0), 2)
    assert [row["loss/total"] for row in read_rows(tmp_path)] == [3.0, 9.0]
    assert not cb._loss_sums and not cb._loss_counts


def test_nonzero_rank_reduces_but_does_not_write(tmp_path, monkeypatch):
    cb = make_callback(tmp_path, monkeypatch, rank0=False, online=True)
    reduce, log = Mock(), Mock()
    monkeypatch.setattr(logging.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(logging.dist, "all_reduce", reduce)
    monkeypatch.setattr(logging.wandb, "log", log)
    cb.on_training_step_end(None, {}, output(), torch.tensor(3.0), 1)
    reduce.assert_called_once()
    log.assert_not_called()
    assert not (tmp_path / "loss_metrics.jsonl").exists()


def test_local_row_survives_online_upload_error(tmp_path, monkeypatch):
    cb = make_callback(tmp_path, monkeypatch, online=True)
    monkeypatch.setattr(logging.wandb, "log", Mock(side_effect=RuntimeError("upload failed")))
    with pytest.raises(RuntimeError, match="upload failed"):
        cb.on_training_step_end(None, {}, output(), torch.tensor(3.0), 1)
    assert read_rows(tmp_path)[0]["loss/total"] == 3.0


def _native_rank_mean_worker(rank, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        cb = logging.LossOnlyWandBCallback()
        cb.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=1))
        cb.trainer = SimpleNamespace(training_timer=Mock())
        cb.trainer.training_timer.compute_average_results.return_value = {}
        local_count = (1, 3)[rank]
        parameter = torch.tensor(2.0, requires_grad=True)
        loss = parameter * (1.0, 5.0)[rank]
        expected_grad = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
        metadata = {
            "train_objective_numerator": loss.detach(),
            "train_objective_denominator": torch.ones_like(loss.detach()),
        }
        saved_metadata = {key: value.clone() for key, value in metadata.items()}
        with patch.object(native.wandb_util, "init_wandb"), \
             patch.object(native.wandb, "log") as log, \
             patch.object(native.distributed, "is_rank0", return_value=rank == 0), \
             patch.object(native.misc, "get_data_batch_size", side_effect=AssertionError("double weighting")):
            cb.on_train_start(None)
            # Check repeated logging resets, and more than one microbatch.
            for step, microsteps in ((1, 1), (2, 2)):
                for _ in range(microsteps):
                    cb.on_training_step_batch_end(
                        None, {"video": torch.zeros(local_count, 1)}, metadata, loss, step - 1,
                    )
                cb.on_training_step_end(None, {"video": torch.zeros(local_count, 1)}, metadata, loss, step)
                assert cb._train_objective_numerator is None
                assert cb._train_objective_denominator is None
            if rank == 0:
                rows = [call.args[0] for call in log.call_args_list if "train/loss_avg" in call.args[0]]
                assert len(rows) == 2
                for row in rows:
                    # SUM(2, 10) / SUM(1, 1), not (2*1 + 10*3)/(1+3).
                    assert row["train/loss_avg"] == pytest.approx(6.0)
                    assert row["train/loss"] == pytest.approx(6.0)
                    assert row["train/loss_avg"] != pytest.approx(8.0)
            else:
                log.assert_not_called()
        for key, value in metadata.items():
            torch.testing.assert_close(value, saved_metadata[key])
        torch.testing.assert_close(loss, torch.tensor((2.0, 10.0)[rank]))
        loss.backward()
        torch.testing.assert_close(parameter.grad, expected_grad)
    finally:
        dist.destroy_process_group()


def test_native_logger_rank_mean_ignores_unequal_local_pack_sizes(tmp_path):
    mp.spawn(_native_rank_mean_worker, args=(str(tmp_path / "logger_gloo"),), nprocs=2, join=True)
