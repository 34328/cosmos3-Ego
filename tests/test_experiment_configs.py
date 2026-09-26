from types import SimpleNamespace

import torch

from cosmos3_joint_video_hand_pose.src import config as experiment_config
from cosmos3_joint_video_hand_pose.src import model as model_module


def test_only_supported_experiments_are_defined():
    names = {name for name in vars(experiment_config) if name.startswith("egoverse_joint_video_hand_pose_overfit_")}
    assert names == {
        "egoverse_joint_video_hand_pose_overfit_v0_0",
        "egoverse_joint_video_hand_pose_overfit_v0_2_lr_balanced",
        "egoverse_joint_video_hand_pose_overfit_v0_3_active_norm_independent_action",
        "egoverse_joint_video_hand_pose_overfit_v0_5_frame_delta_b3",
    }


def test_v0_5_derives_from_v0_3_with_b3_only():
    v0_3 = experiment_config.egoverse_joint_video_hand_pose_overfit_v0_3_active_norm_independent_action
    v0_5 = experiment_config.egoverse_joint_video_hand_pose_overfit_v0_5_frame_delta_b3
    assert v0_5["model"]["_target_"] is model_module.EgoVerseOmniMoTModel
    assert v0_5["model"]["config"] == v0_3["model"]["config"]
    dataset = v0_5["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
    assert dataset["rigid_pose_frame_delta"] is True
    assert dataset["future_normalizer"].endswith("v3_frame_delta/normalizers/future_frame_delta_normalizer.json")


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
