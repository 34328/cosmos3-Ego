"""The video-LR control changes only LR allocation and run identity."""
from cosmos3_joint_video_hand_pose.src import config as project
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml


def test_video_lr_control_preserves_formal_recipe():
    root = project.COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/configs"
    baseline = load_experiment_from_toml(root / "ar_v0_2_fixed_camera.toml", [])
    control = load_experiment_from_toml(root / "ar_v0_2_video_lr1e4.toml", [])
    for name in ("model", "scheduler", "trainer", "checkpoint", "dataloader_train"):
        assert getattr(control, name) == getattr(baseline, name), name
    assert control.optimizer.lr == 1e-4
    assert dict(control.optimizer.lr_multipliers) == dict.fromkeys(
        ("action2llm", "llm2action", "action_modality_embed", "action_state_embed", "vision_condition_embed"), 1
    )
    for name in baseline.optimizer:
        if name not in ("lr", "lr_multipliers"):
            assert control.optimizer[name] == baseline.optimizer[name], name
    assert control.model.history_video_noise_prob == 0
    assert control.job.wandb_mode == "online"
    assert control.job.name == "video_lr1e4_clean_history"
    assert list(control.scheduler.warm_up_steps) == [100]
    assert list(control.scheduler.cycle_lengths) == [1000]
    assert list(control.scheduler.f_min) == [.1]
    assert control.trainer.max_iter == 1000 and control.checkpoint.save_iter == 500
