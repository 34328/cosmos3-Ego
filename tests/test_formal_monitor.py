from types import SimpleNamespace
import pytest
import torch

from cosmos3_joint_video_hand_pose.src.formal_monitor import StopPolicy, optimizer_lr_receipt, FormalTrainingMonitor


def test_clipping_stop_includes_warmup_and_exact_boundary():
    p = StopPolicy()
    for i in range(9):
        assert p.update(i != 0, [1., 2., 3.], 30.) is None
    assert "9 of" in p.update(True, [1., 2., 3.], 30.)
    p = StopPolicy()
    for i in range(10):
        assert p.update(i >= 2, [1., 2., 3.], 30.) is None


def test_divergence_and_nonfinite_stop():
    p = StopPolicy()
    for _ in range(10):
        assert p.update(False, [1., 2., 3.], 30.) is None
    assert "3x" in p.update(False, [30., 2., 32.], 30.)
    assert "nonfinite" in StopPolicy().update(False, [float("nan")], 30.)


def test_memory_growth_distinguished_from_packing_fluctuation():
    p = StopPolicy()
    for i in range(9):
        assert p.update(False, [1.], 30.+i*.3) is None
    assert "memory" in p.update(False, [1.], 32.7)
    p = StopPolicy()
    for i in range(30):
        assert p.update(False, [1.], 30.+i%3) is None


def test_native_optimizer_group_receipt_uses_parameter_identity():
    net = torch.nn.ModuleDict({"action2llm": torch.nn.Linear(2, 2), "other": torch.nn.Linear(2, 2)})
    opt = torch.optim.Adam([{"params": net["action2llm"].parameters(), "lr": 1e-4},
                            {"params": net["other"].parameters(), "lr": 2e-5}])
    torch.optim.lr_scheduler.LambdaLR(opt, lambda step: step/100)
    cfg = dict(lr=2e-5, lr_multipliers={"action2llm": 5})
    rows = optimizer_lr_receipt(net, SimpleNamespace(optimizers=[opt]), cfg)
    assert [x["initial_lr"] for x in rows] == [1e-4, 2e-5]
    assert all(x["actual_lr"] == 0 for x in rows)
    opt.param_groups[0]["initial_lr"] = 2e-5
    with pytest.raises(ValueError, match="LR mismatch"):
        optimizer_lr_receipt(net, opt, cfg)


def test_nonfinite_loss_fails_before_backward():
    monitor = FormalTrainingMonitor()
    monitor.on_before_backward(None, torch.tensor(1.))
    with pytest.raises(FloatingPointError):
        monitor.on_before_backward(None, torch.tensor(float("nan")))


def test_formal_recipe_composes_without_cli_overrides():
    from cosmos3_joint_video_hand_pose.src import config as project
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    c = load_experiment_from_toml(project.COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_2_fixed_camera.toml", [])
    assert c.model.config.parallelism.data_parallel_shard_degree == 8
    assert c.model.config.parallelism.data_parallel_replicate_degree == 2
    assert c.trainer.max_iter == 1000 and c.checkpoint.save_iter == 500
    assert list(c.scheduler.warm_up_steps) == [100] and list(c.scheduler.cycle_lengths) == [1000]
    ds = c.dataloader_train.dataloader.datasets.egoverse.dataset
    assert list(ds.clip_frame_tiers) == [273, 257, 129, 65, 33]
    assert "_t273/" in ds.future_normalizer
    assert c.job.wandb_mode == "online"
    assert c.trainer.callbacks.formal_monitor["_target_"] is FormalTrainingMonitor


def test_callback_records_fields_then_stops_without_changing_loss(monkeypatch, tmp_path):
    import json
    import cosmos3_joint_video_hand_pose.src.formal_monitor as module
    from cosmos_framework.callbacks.grad_clip import GradClip
    monkeypatch.setattr(module.wandb, "run", None)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("memory_allocated", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 30 * 2**30)
    clip = GradClip(clip_norm=1.)
    clip._last_global_norm[""] = torch.tensor(2.)
    m = FormalTrainingMonitor()
    m.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    m.trainer = SimpleNamespace(callbacks=SimpleNamespace(_callbacks=[clip]))
    model = SimpleNamespace(_optimizer_lr_receipt=[])
    m.on_train_start(model)
    output = {k: torch.tensor(1.) for k in module.LOSS_METRIC_SOURCES.values()}
    loss = torch.tensor(2., requires_grad=True)
    for i in range(10):
        m.on_training_step_start(model, {"video": [None]*3}, iteration=i)
        m.on_before_backward(model, loss, iteration=i)
        m.on_training_step_batch_end(model, {}, output, loss, iteration=i)
        if i < 9:
            m.on_training_step_end(model, {}, output, loss, iteration=i+1)
        else:
            with pytest.raises(RuntimeError, match="9 of"):
                m.on_training_step_end(model, {}, output, loss, iteration=i+1)
    row = json.loads((tmp_path / "STOPPED.json").read_text())
    assert row["step"] == 10 and row["global_batch"] == 3 and row["clips_per_rank"] == [3]
    assert set(module.LOSS_METRIC_SOURCES) <= row.keys()
    assert loss.item() == 2. and loss.grad is None
    resumed = FormalTrainingMonitor()
    resumed.config, resumed.trainer = m.config, m.trainer
    resumed.on_train_start(model, iteration=9)
    assert list(resumed.policy.clips) == [True]*9
