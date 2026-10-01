import math
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from cosmos3_joint_video_hand_pose.src.ar_v03_config import (
    AR_V03_CONFIG_NAME, AR_V03_LR_MULTIPLIERS,
    _ar_v03_experiment, learning_rate_receipt,
)
from cosmos3_joint_video_hand_pose.src.ar_v03_loss import AR_V03_ACTION_CHANNEL_WEIGHTS
from cosmos3_joint_video_hand_pose.src.ar_v03_model import EgoVerseARV03Model
from cosmos3_joint_video_hand_pose.src.config import (
    COSMOS_REPO_ROOT, _ar_v02_fixed_camera_experiment,
)
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

TOML = COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_3.toml"


def load(overrides=()):
    return load_experiment_from_toml(TOML, list(overrides))


def test_versioned_recipe_preserves_data_contract_and_native_lifecycle():
    old = _ar_v02_fixed_camera_experiment()
    new = _ar_v03_experiment()
    assert new["model"]["_target_"] is EgoVerseARV03Model
    assert new["dataloader_train"] == old["dataloader_train"]
    assert new["optimizer"]["keys_to_select"] == old["optimizer"]["keys_to_select"]
    assert new["trainer"]["callbacks"]["formal_monitor"] == old["trainer"]["callbacks"]["formal_monitor"]
    assert new["trainer"]["callbacks"]["ar_v02_contract"] == old["trainer"]["callbacks"]["ar_v02_contract"]
    assert new["trainer"]["cudnn"] == old["trainer"]["cudnn"]
    assert new["model"]["config"].max_num_tokens_after_packing == 60000
    assert new["model"]["config"].action_tokens_per_latent == 8
    assert new["model"]["config"].teacher_forcing_frames_per_chunk == 4
    assert new["model"]["config"].causal_training_strategy == "diffusion_forcing"
    assert new["model"]["config"].sigma_small == 0.02
    assert new["model"]["config"].prefix_low_noise_enabled is True
    assert new["model"]["config"].sigma_hist_max == 0.1
    assert list(new["model"]["config"].action_channel_weights) == list(AR_V03_ACTION_CHANNEL_WEIGHTS)
    assert old == _ar_v02_fixed_camera_experiment()


def test_toml_composes_using_native_trainer_optimizer_and_scheduler():
    from cosmos_framework.trainer import ImaginaireTrainer
    from cosmos_framework.utils.generator.optimizer import build_optimizer, build_lr_scheduler
    config = load()
    assert config.job.group == "ar_v0_3"
    assert config.job.wandb_mode == "online"
    assert config.model._target_ is EgoVerseARV03Model
    assert config.trainer.type is ImaginaireTrainer
    assert config.optimizer._target_ is build_optimizer
    assert config.scheduler._target_ is build_lr_scheduler
    assert config.trainer.max_iter == 3000
    assert config.checkpoint.save_iter == 500
    assert dict(config.optimizer.lr_multipliers) == AR_V03_LR_MULTIPLIERS
    assert config.model.config.parallelism.data_parallel_shard_degree == 8
    assert config.model.config.parallelism.data_parallel_replicate_degree == 2
    assert config.model.config.parallelism.context_parallel_shard_degree == 1
    assert "ar_v03_lr_receipt" in config.trainer.callbacks
    assert config.model.history_video_noise_prob == 0
    assert config.model.config.prefix_low_noise_enabled is True
    assert config.model.config.sigma_hist_max == 0.1


def test_native_cli_overrides_prefix_fields_after_registered_defaults():
    config = load(["model.config.prefix_low_noise_enabled=false", "model.config.sigma_hist_max=0.05"])
    assert config.model.config.prefix_low_noise_enabled is False
    assert config.model.config.sigma_hist_max == 0.05
    assert config.model.config.sigma_small == 0.02
    assert config.model.config.max_num_tokens_after_packing == 60000
    assert config.dataloader_train == load().dataloader_train


def test_receipt_uses_official_total_cycle_including_warmup():
    receipt = learning_rate_receipt(load())
    assert [row["step"] for row in receipt["theoretical_lr"]] == [0, 100, 1500, 3000]
    expected = [0, 1e-4, 1e-4 * (0.3 + 0.35 * (1 + math.cos(math.pi * 1400 / 2900))), 3e-5]
    assert [row["lr"] for row in receipt["theoretical_lr"]] == pytest.approx(expected, rel=1e-12)
    assert receipt["max_iter"] == 3000 and receipt["save_iter"] == 500
    smoke = learning_rate_receipt(load(["trainer.max_iter=20", "checkpoint.save_iter=20"]))
    assert smoke["theoretical_lr"] == receipt["theoretical_lr"]
    assert smoke["max_iter"] == 20 and smoke["save_iter"] == 20


def test_native_optimizer_init_hook_prints_and_persists_final_config(tmp_path, capsys):
    import json
    from cosmos3_joint_video_hand_pose.src.ar_v03_config import ARV03LearningRateReceiptCallback
    config = load(["trainer.max_iter=20", "checkpoint.save_iter=20"])
    callback = ARV03LearningRateReceiptCallback()
    callback.config = SimpleNamespace(
        optimizer=config.optimizer, scheduler=config.scheduler,
        trainer=config.trainer, checkpoint=config.checkpoint,
        job=SimpleNamespace(path_local=str(tmp_path)),
    )
    callback.on_optimizer_init_start()
    saved = json.loads((tmp_path / "learning_rate_receipt.json").read_text())
    printed = capsys.readouterr().out
    assert printed.startswith("AR_V03_LEARNING_RATE_RECEIPT ")
    assert json.loads(printed.split(" ", 1)[1]) == saved
    assert saved["max_iter"] == 20


@pytest.mark.parametrize("override", [
    "optimizer.lr=2e-5", "optimizer.lr_multipliers.action2llm=5",
    "scheduler.cycle_lengths=[1000]", "scheduler.f_min=[0.1]",
    "scheduler.f_start=[1e-6]", "scheduler.lr_scheduler_type=LambdaLinear",
])
def test_receipt_rejects_drift_after_final_cli_override(override):
    with pytest.raises(ValueError, match="V0.3"):
        learning_rate_receipt(load([override]))


def test_entrypoint_delegates_without_replacing_native_cli(monkeypatch):
    from cosmos3_joint_video_hand_pose.src import train_ar_v03
    calls = []
    monkeypatch.setattr(train_ar_v03.runpy, "run_module", lambda *a, **kw: calls.append((a, kw)))
    train_ar_v03.main()
    assert calls == [(("cosmos_framework.scripts.train",), {"run_name": "__main__"})]
