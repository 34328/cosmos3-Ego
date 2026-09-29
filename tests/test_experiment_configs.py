from types import SimpleNamespace

import torch

from cosmos3_joint_video_hand_pose.src import config as experiment_config
from cosmos3_joint_video_hand_pose.src import model as model_module


CONFIG_DIR = experiment_config.COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/configs"
CURRENT_NAMES = ("ar_v0_2", "ar_v0_2_c", "ar_v0_2_multitask")
NATIVE_CALLBACKS = (
    "wandb", "wandb_2x", "iter_speed", "manual_gc", "load_pretrained",
    "param_count", "sequence_packing_padding",
)


def test_only_current_experiments_are_registered():
    from hydra.core.config_store import ConfigStore

    names = {
        name for name in ConfigStore.instance().list("experiment")
        if name.startswith("egoverse_joint_video_hand_pose_")
    }
    assert names == {"egoverse_joint_video_hand_pose_" + name + ".yaml" for name in CURRENT_NAMES}
    assert not any("overfit" in name or "ar_v0_1" in name for name in vars(experiment_config))
    assert {p.stem for p in CONFIG_DIR.glob("*.toml")} == {"ar_v0_1", "ar_v0_2", "ar_v0_2_fixed_camera"}


def test_native_callbacks_are_copied_without_replacement():
    from cosmos_framework.configs.base.defaults.callbacks import BASIC_CALLBACKS

    callbacks = experiment_config._EGOVERSE_BASIC_CALLBACKS
    assert set(callbacks) == {*NATIVE_CALLBACKS, "egoverse_loss_wandb"}
    for name in NATIVE_CALLBACKS:
        assert callbacks[name] == BASIC_CALLBACKS[name]
        assert callbacks[name] is not BASIC_CALLBACKS[name]


def test_current_recipe_has_no_historical_lr_or_data_inheritance():
    from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout

    experiment = experiment_config._ar_v02_experiment(True)
    model = experiment["model"]["config"]
    assert experiment["optimizer"]["lr_multipliers"] == {}
    assert not any(
        name in experiment["checkpoint"]["keys_to_skip_loading"]
        for name in ("action2llm", "llm2action", "action_modality_embed")
    )
    assert experiment["optimizer"]["lr"] == 2.0e-5
    assert experiment["scheduler"]["lr_scheduler_type"] == "LambdaCosine"
    assert model.max_num_tokens_after_packing == 60000
    assert model.parallelism.context_parallel_shard_degree == 1
    assert model.parallelism.data_parallel_shard_degree == 8
    assert model.action_tokens_per_latent == 8
    assert model.teacher_forcing_frames_per_chunk == 4
    assert JointChunkLayout(num_frames=2, vision_tokens=1, chunk_size=1).history_chunks == 15
    assert model.teacher_forcing_detach_clean_kv is False
    assert model.diffusion_expert_config.enable_action_state_embedding
    assert model.diffusion_expert_config.enable_vision_condition_embedding
    rf = model.rectified_flow_training_config
    assert rf.loss_scale == 1.0 and rf.action_loss_weight == 1.0
    assert rf.normalize_loss_by_active and rf.independent_action_schedule
    assert rf.shift_action == 5
    assert experiment["checkpoint"]["save_iter"] == 600
    assert experiment["job"]["wandb_mode"] == "online"
    assert experiment["trainer"]["callbacks"]["manual_gc"] == {
        "every_n": 1, "gc_level": 2, "warm_up": 0,
    }
    loader = experiment["dataloader_train"]
    assert loader["joint_max_samples"] == 4
    assert loader["dataloader"]["stateful"] and loader["dataloader"]["in_order"]
    dataset = loader["dataloader"]["datasets"]["egoverse"]["dataset"]
    assert dataset["prompt_mode"] == "segment_only"
    assert dataset["cfg_dropout_rate"] == 0.1
    assert dataset["frame_stride"] == 2
    assert list(dataset["clip_frame_tiers"]) == [129, 65, 33]
    root = experiment_config.COSMOS_REPO_ROOT / "outputs/data_expansion_20260928"
    assert dataset["episodes_manifest"] == str(root / "episodes.csv")
    assert dataset["segments_manifest"] == str(root / "segments.csv")
    contract = experiment["trainer"]["callbacks"]["ar_v02_contract"]
    for key, filename, contract_key in (
        ("chunk_state_normalizer", "chunk_state_normalizer.json", "state_normalizer"),
        ("future_normalizer", "future_frame_delta_normalizer.json", "action_normalizer"),
        ("valid_windows_manifest", "valid_windows.json", "valid_windows_manifest"),
    ):
        assert dataset[key] == str(root / "prepared_v02" / filename)
        assert contract[contract_key] == dataset[key]


def test_every_toml_composes_through_native_cosmos_loader():
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils.callback import WandBCallback


    for name in ("ar_v0_2", "ar_v0_2_fixed_camera"):
        config = load_experiment_from_toml(CONFIG_DIR / (name + ".toml"), [])
        # Native validate() initializes the GPU world communicator; composition is CPU-only.
        from cosmos_framework.trainer import ImaginaireTrainer

        assert config.trainer.type is ImaginaireTrainer
        assert config.job.wandb_mode == "online"
        assert config.model.config.rectified_flow_training_config.action_loss_weight == 1.0
        assert config.checkpoint.save_iter == 600
        assert config.optimizer.lr_multipliers == {}
        assert config.trainer.max_iter == 1200
        assert list(config.scheduler.cycle_lengths) == [1200]
        assert config.model.chunk_state_conditioning
        assert config.trainer.callbacks.wandb["_target_"] is WandBCallback
        assert set(NATIVE_CALLBACKS) <= set(config.trainer.callbacks)
        dataset = config.dataloader_train.dataloader.datasets.egoverse.dataset
        assert "outputs/data_expansion_20260928/" in dataset.episodes_manifest
        assert config.dataloader_train.max_sequence_length == 60000


def test_subblock_losses_are_not_repeated_from_previous_step(monkeypatch):
    """A step without the 57D action loss must not re-log the last step's values."""
    model = object.__new__(model_module.EgoVerseOmniMoTModel)
    torch.nn.Module.__init__(model)
    model._last_visibility_loss_metrics = {}
    model.config = SimpleNamespace(
        rectified_flow_training_config=SimpleNamespace(
            sample_level_loss_averaging=False, image_loss_scale=None, loss_scale=1.0, action_loss_weight=1.0
        ),
        vision_gen=False,
    )
    model.rectified_flow_video = SimpleNamespace(
        noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    )
    writes = iter([{"camera_translation_loss": torch.tensor(3.0)}, None])

    def fake_parent_compute_losses(self, **kwargs):
        metrics = next(writes)
        if metrics is not None:  # the 57D path ran in this step
            self._last_visibility_loss_metrics = metrics
        zero = torch.zeros(())
        return zero, {"flow_matching_loss_vision": zero, "flow_matching_loss_action": zero}

    monkeypatch.setattr(model_module.OmniMoTModel, "_compute_losses", fake_parent_compute_losses)
    kwargs = dict(
        out_net={}, data_batch_packed={}, gen_data_noised=None, timesteps=torch.tensor([500.0]), is_image_batch=False
    )
    _, first = model._compute_losses(**kwargs)
    _, second = model._compute_losses(**kwargs)
    assert first["egoverse_loss_action_camera_translation_raw"] == 3.0
    assert "egoverse_loss_action_camera_translation_raw" not in second
