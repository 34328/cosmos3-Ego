from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.utils.lazy_config import instantiate

from cosmos3_joint_video_hand_pose.src import config as experiment_config
from cosmos3_joint_video_hand_pose.src import model as model_module
from cosmos3_joint_video_hand_pose.src.dataloader_state import RecoverablePackingDataLoader


REMOVED = {
    "v0_4": experiment_config.egoverse_joint_video_hand_pose_overfit_v0_4_video_first_causal_mask,
    "v0_6": experiment_config.egoverse_joint_video_hand_pose_overfit_v0_6_frame_delta_temporal_mask,
}


@pytest.mark.parametrize("name", sorted(REMOVED))
@pytest.mark.parametrize("part", ["model", "dataloader_train"])
def test_removed_mask_experiments_refuse_to_build(name, part):
    experiment = REMOVED[name]
    # The dataloader must fail before any child dataset is constructed.
    assert experiment["dataloader_train"]["_recursive_"] is False
    with pytest.raises(RuntimeError, match="cf5d68c.*8525625"):
        instantiate(experiment[part])


def test_v0_3_and_v0_5_are_not_disabled():
    for experiment in (
        experiment_config.egoverse_joint_video_hand_pose_overfit_v0_3_active_norm_independent_action,
        experiment_config.egoverse_joint_video_hand_pose_overfit_v0_5_frame_delta_b3,
    ):
        assert experiment["model"]["_target_"] is model_module.EgoVerseOmniMoTModel
        assert experiment["dataloader_train"]["_target_"] is RecoverablePackingDataLoader
        assert "_recursive_" not in experiment["dataloader_train"]


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
