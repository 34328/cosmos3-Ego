"""The fixed-camera recipe is explicit and cannot inherit legacy artifacts."""

from hydra.core.config_store import ConfigStore

from cosmos3_joint_video_hand_pose.src import config as experiment_config


def test_fixed_camera_recipe_is_separate_and_fully_bound():
    experiment = experiment_config._ar_v02_fixed_camera_experiment()
    representation = experiment_config.FIXED_CAMERA_ACTION_REPRESENTATION
    model = experiment["model"]
    assert model["action_representation"] == representation
    assert model["config"].teacher_forcing_frames_per_chunk == 4
    assert model["config"].rectified_flow_training_config.action_loss_weight == 1.0

    dataset = experiment["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
    contract = experiment["trainer"]["callbacks"]["ar_v02_contract"]
    assert dataset["action_representation"] == representation
    assert dataset["state_normalizer"] is None
    assert (dataset["right_codec"], dataset["left_codec"]) == (
        contract["right_hand_codec"], contract["left_hand_codec"]
    )
    assert dataset["chunk_state_normalizer"] == contract["state_normalizer"]
    assert dataset["future_normalizer"] == contract["action_normalizer"]
    assert dataset["valid_windows_manifest"] == contract["valid_windows_manifest"]
    assert contract["representation"] == representation
    all_paths = " ".join(
        (dataset["right_codec"], dataset["left_codec"],
         dataset["chunk_state_normalizer"], dataset["future_normalizer"])
    )
    assert "prepared_fixed_camera_wrist_local_delta_latent_v1" in all_paths
    assert "v3_wrist_local_pca15_train744/right_pca15.pt" in dataset["right_codec"]
    assert "v3_wrist_local_pca15_train744/left_pca15.pt" in dataset["left_codec"]
    assert "v2_4" not in all_paths and "cosmos3_action_contract/v2" not in all_paths


def test_legacy_aliases_do_not_silently_switch_representation():
    legacy = experiment_config._ar_v02_experiment(True)
    assert legacy["model"]["action_representation"] == "legacy_local_delta_absolute_hand_v1"
    dataset = legacy["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
    assert dataset["right_codec"] is None and dataset["left_codec"] is None
    assert "prepared_v02" in dataset["chunk_state_normalizer"]


def test_fixed_recipe_has_a_distinct_explicit_config_store_name():
    names = ConfigStore.instance().list("experiment")
    assert experiment_config.FIXED_CAMERA_CONFIG_NAME + ".yaml" in names
    assert "rbs_wam_ar_v0_2_fixed_camera_delta_latent_v1.yaml" not in names


def test_fixed_toml_selects_new_representation_through_official_loader():
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    path = experiment_config.COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_2_fixed_camera.toml"
    config = load_experiment_from_toml(path, [])
    representation = experiment_config.FIXED_CAMERA_ACTION_REPRESENTATION
    assert config.model.action_representation == representation
    assert config.dataloader_train.dataloader.datasets.egoverse.dataset.action_representation == representation
    assert config.trainer.callbacks.ar_v02_contract.representation == representation


def test_runtime_paths_resolve_from_environment_through_official_loader(monkeypatch, tmp_path):
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    base = tmp_path / "base_dcp"
    vae = tmp_path / "vae.pth"
    tokenizer = tmp_path / "text_tokenizer"
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", str(base))
    monkeypatch.setenv("WAN_VAE_PATH", str(vae))
    monkeypatch.setenv("TEXT_TOKENIZER_PATH", str(tokenizer))
    path = experiment_config.COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_2_fixed_camera.toml"
    config = load_experiment_from_toml(path, [])

    resolved = (
        str(config.checkpoint.load_path),
        str(config.model.config.tokenizer.vae_path),
        str(config.model.config.vlm_config.tokenizer.pretrained_model_name),
        str(config.trainer.callbacks.ar_v02_contract.official_checkpoint),
    )
    assert resolved == (str(base), str(vae), str(tokenizer), str(base))
    assert all("${oc.env:" not in value for value in resolved)
