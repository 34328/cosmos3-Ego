"""Continuous training VAE routing and isolated V0.4 checkpoint semantics (CPU)."""
from types import SimpleNamespace
import json

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_model import ARStepContext
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v031_model import EgoVerseARV031Model
from cosmos3_joint_video_hand_pose.src.ar_v04_model import EgoVerseARV04Model
from cosmos3_joint_video_hand_pose.src.ar_v04_codec import VIDEO_LATENT_FORMAT
from cosmos3_joint_video_hand_pose.src.ar_v04_checkpoint import (
    ARV04VideoFormatCallback, FORMAT_FILE, OFFICIAL_NANO, validate_video_checkpoint,
)


@pytest.mark.parametrize("frame_counts", [(33,), (33, 21)])
@pytest.mark.parametrize("balance", [False, True])
def test_full_official_encode_then_boundary_gather(monkeypatch, frame_counts, balance):
    model = object.__new__(EgoVerseARV04Model)
    torch.nn.Module.__init__(model)
    model._ar_step = ARStepContext(4, 15)
    calls = []
    continuous = []
    # A context-dependent encoder stub makes reset-at-boundary semantically
    # different. Patch only the costly VAE; execute the official encode router.
    model.tokenizer_vision_gen = None
    def encode_item(raw, num_views):
        calls.append(raw.shape[2])
        assert num_views == 1
        latent = raw.cumsum(2)[:, :, ::4]
        continuous.append(latent)
        return latent
    model._encode_vision_item = encode_item
    model._vision_encoder = lambda: SimpleNamespace(balancing_available=lambda: False)
    originals = [torch.arange(1, n + 1.).reshape(1, 1, n, 1, 1) for n in frame_counts]
    output = model._encode_vision_x0_tokens(originals, None, [[0]] * len(originals),
                                           balance_vae_encode=balance)
    assert calls == list(frame_counts), "one full encode per sample, never per AR chunk"
    assert model._joint_original_frames == [1 + (n - 1) // 4 for n in frame_counts]
    for packed, latent in zip(output, continuous, strict=True):
        layout = JointChunkLayout(latent.shape[2], 1, 4)
        cursor = 0
        for boundary in layout.boundaries:
            expected = latent[:, :, boundary.latent_start - 1:boundary.latent_stop]
            width = expected.shape[2]
            torch.testing.assert_close(packed[:, :, cursor:cursor+width], expected, atol=0, rtol=0)
            cursor += width
        assert cursor == packed.shape[2]


def test_v031_objective_noising_and_layout_remain_inherited():
    for method in ("_compute_whole_losses", "_add_noise_to_input", "_get_train_noise_level_vision",
                   "_get_train_noise_level_action", "_prepare_training_data", "_pack_input_sequence"):
        assert getattr(EgoVerseARV04Model, method) is getattr(EgoVerseARV031Model, method)


def test_training_encoder_rejects_unaligned_video():
    model = object.__new__(EgoVerseARV04Model)
    torch.nn.Module.__init__(model)
    model._ar_step = ARStepContext(4, 15)
    with pytest.raises(ValueError, match="1\\+4N"):
        model._encode_vision_x0_tokens([torch.zeros(1, 3, 18, 2, 2)], None, None)


def test_checkpoint_guard_rejects_old_joint_dcp_and_wrong_format(tmp_path):
    model_path = tmp_path / "iter_000000500" / "model"
    model_path.mkdir(parents=True)
    with pytest.raises(ValueError, match="lacks V0.4"):
        validate_video_checkpoint(model_path)
    marker = model_path.parent / FORMAT_FILE
    marker.write_text(json.dumps(dict(model_version="ar_v0.3.1", video_latent_format=VIDEO_LATENT_FORMAT)))
    with pytest.raises(ValueError, match="incompatible"):
        validate_video_checkpoint(model_path)
    marker.write_text(json.dumps(dict(model_version="ar_v0.4", video_latent_format="reset_each_chunk")))
    with pytest.raises(ValueError, match="incompatible"):
        validate_video_checkpoint(model_path)


def test_official_nano_is_only_allowed_for_training_initialization():
    assert validate_video_checkpoint(OFFICIAL_NANO, allow_official_nano=True)["initialization"] == "official_nano"
    with pytest.raises(ValueError, match="lacks V0.4"):
        validate_video_checkpoint(OFFICIAL_NANO)


def test_checkpoint_success_writes_format_receipt_and_load_guard(tmp_path, monkeypatch):
    from cosmos3_joint_video_hand_pose.src import ar_v04_checkpoint as module
    monkeypatch.setattr(module.distributed, "is_rank0", lambda: True)
    callback = ARV04VideoFormatCallback()
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    model_path = tmp_path / "checkpoints" / "iter_000000500" / "model"
    model_path.mkdir(parents=True)
    with pytest.raises(ValueError, match="missing its model metadata"):
        callback.on_save_checkpoint_success(iteration=500)
    (model_path / ".metadata").write_bytes(b"test checkpoint metadata")
    callback.on_save_checkpoint_success(iteration=500)
    receipt = validate_video_checkpoint(model_path)
    assert receipt["step"] == 500
    assert receipt["video_latent_format"] == VIDEO_LATENT_FORMAT
    callback.on_load_checkpoint_end(None, 500, str(model_path.parent))
    with pytest.raises(ValueError, match="must initialize"):
        callback.on_load_checkpoint_end(None)


def test_v04_config_preserves_v031_recipe_except_model_identity():
    from omegaconf import OmegaConf
    from cosmos3_joint_video_hand_pose.src.ar_v031_config import _ar_v031_experiment
    from cosmos3_joint_video_hand_pose.src.ar_v04_config import _ar_v04_experiment
    old = OmegaConf.to_container(OmegaConf.create(_ar_v031_experiment()), resolve=False)
    new = OmegaConf.to_container(OmegaConf.create(_ar_v04_experiment()), resolve=False)
    assert new["model"]["config"].pop("video_latent_format") == VIDEO_LATENT_FORMAT
    new["model"]["_target_"] = old["model"]["_target_"]
    new["job"] = old["job"]
    new["trainer"]["callbacks"].pop("ar_v04_video_format")
    assert new == old


def test_v04_toml_composes_through_official_loader():
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos3_joint_video_hand_pose.src import ar_v04_config  # noqa: F401
    from cosmos3_joint_video_hand_pose.src.config import COSMOS_REPO_ROOT
    path = COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_4.toml"
    config = load_experiment_from_toml(path, [])
    assert config.model["_target_"] is EgoVerseARV04Model
    assert config.job.name == "ar_v0_4_continuous_video_v1"
    assert config.job.wandb_mode == "online"
    assert config.model.config.video_latent_format == VIDEO_LATENT_FORMAT
    assert config.model.config.mask_prefix_loss is True
    assert config.optimizer.lr == 1e-4
    assert config.trainer.max_iter == 3000
    assert list(config.scheduler.cycle_lengths) == [3000]
    assert config.checkpoint.save_iter == 500
