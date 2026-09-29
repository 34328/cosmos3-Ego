"""Fixed-camera artifact/checkpoint bindings; CPU only and no training."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos3_joint_video_hand_pose.src.action_fixed_normalization import (
    REPRESENTATION,
    fit_fixed_normalizer,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_contract import (
    ARTrainingContract,
    ARTrainingContractCallback,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixed_artifacts(tmp_path, monkeypatch):
    manifest = tmp_path / "valid_windows.json"
    manifest.write_text("{}")
    manifest_sha = _sha256(manifest)
    codecs = (tmp_path / "right_mlp15.pt", tmp_path / "left_mlp15.pt")
    codecs[0].write_bytes(b"new fixed camera right codec")
    codecs[1].write_bytes(b"new fixed camera left codec")
    codec_sha = tuple(_sha256(path) for path in codecs)
    state_values = np.zeros((3, 57), dtype=np.float64)
    state_values[:, 9:] = np.arange(48, dtype=np.float64)
    future_values = np.arange(3 * 57, dtype=np.float64).reshape(3, 57)
    state_profile = fit_fixed_normalizer(
        state_values, kind="state", codec_sha256=codec_sha, manifest_sha256=manifest_sha
    )
    future_profile = fit_fixed_normalizer(
        future_values, kind="future", codec_sha256=codec_sha, manifest_sha256=manifest_sha
    )
    state = tmp_path / "state_normalizer.json"
    future = tmp_path / "future_normalizer.json"
    state.write_text(json.dumps(state_profile))
    future.write_text(json.dumps(future_profile))

    class FakeValidatedCodec:
        representation = REPRESENTATION
        input_frame = "current_frame_wrist_local"

        def __init__(self, path, *, expected_sha256):
            self.checkpoint_sha256 = _sha256(path)
            self.metadata = {"side": Path(path).name.split("_", 1)[0]}
            assert self.checkpoint_sha256 == expected_sha256

    from cosmos3_joint_video_hand_pose.src import codec_fixed_camera

    monkeypatch.setattr(codec_fixed_camera, "FrozenFixedCameraHandCodec", FakeValidatedCodec)
    return SimpleNamespace(
        manifest=manifest,
        manifest_sha=manifest_sha,
        codecs=codecs,
        codec_sha=codec_sha,
        state=state,
        future=future,
        state_profile=state_profile,
        future_profile=future_profile,
    )


def _contract(artifacts):
    return ARTrainingContract(
        state_normalizer=artifacts.state,
        action_normalizer=artifacts.future,
        manifest_sha256=artifacts.manifest_sha,
        representation=REPRESENTATION,
        right_hand_codec=artifacts.codecs[0],
        left_hand_codec=artifacts.codecs[1],
    )


def test_fixed_contract_binds_representation_profiles_and_both_codecs(tmp_path, monkeypatch):
    artifacts = _fixed_artifacts(tmp_path, monkeypatch)
    contract = _contract(artifacts)
    payload = contract.get_extra_state()
    assert payload["schema"] == "ar_v02_training_contract_v2"
    assert payload["representation"] == REPRESENTATION
    assert payload["codec_sha256"] == list(artifacts.codec_sha)
    assert payload["normalizer_profile_sha256"] == {
        "state": artifacts.state_profile["profile_sha256"],
        "future": artifacts.future_profile["profile_sha256"],
    }
    assert payload["artifacts"]["right_hand_codec"]["sha256"] == artifacts.codec_sha[0]
    assert payload["artifacts"]["left_hand_codec"]["sha256"] == artifacts.codec_sha[1]

    source = torch.nn.Module()
    source.add_module("ar_training_contract", contract)
    target = torch.nn.Module()
    target.add_module("ar_training_contract", _contract(artifacts))
    target.load_state_dict(source.state_dict())


def test_fixed_contract_rejects_missing_or_mismatched_new_artifacts(tmp_path, monkeypatch):
    artifacts = _fixed_artifacts(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicit right/left"):
        ARTrainingContract(
            state_normalizer=artifacts.state,
            action_normalizer=artifacts.future,
            manifest_sha256=artifacts.manifest_sha,
            representation=REPRESENTATION,
        )
    artifacts.codecs[0].write_bytes(b"changed after normalizer fit")
    with pytest.raises(ValueError, match="SHA256 identities differ"):
        _contract(artifacts)


def test_fixed_callback_requires_all_batch_artifact_identities(tmp_path, monkeypatch):
    artifacts = _fixed_artifacts(tmp_path, monkeypatch)
    callback = ARTrainingContractCallback(
        state_normalizer=artifacts.state,
        action_normalizer=artifacts.future,
        valid_windows_manifest=artifacts.manifest,
        official_checkpoint=tmp_path / "official",
        representation=REPRESENTATION,
        right_hand_codec=artifacts.codecs[0],
        left_hand_codec=artifacts.codecs[1],
    )
    callback._ready = True
    expected = callback.contract.get_extra_state()
    batch = {
        "ar_layout_version": [expected["layout"]],
        "ar_action_representation": [REPRESENTATION],
        "ar_state_normalizer_sha256": [expected["artifacts"]["state_normalizer"]["sha256"]],
        "ar_future_normalizer_sha256": [expected["artifacts"]["action_normalizer"]["sha256"]],
        "ar_right_hand_codec_sha256": [artifacts.codec_sha[0]],
        "ar_left_hand_codec_sha256": [artifacts.codec_sha[1]],
        "ar_valid_windows_sha256": [artifacts.manifest_sha],
    }
    callback.on_training_step_batch_start(None, batch)
    del batch["ar_future_normalizer_sha256"]
    with pytest.raises(ValueError, match="ar_future_normalizer_sha256"):
        callback.on_training_step_batch_start(None, batch)
def test_callback_rejects_model_representation_mismatch_before_load(tmp_path, monkeypatch):
    artifacts = _fixed_artifacts(tmp_path, monkeypatch)
    callback = ARTrainingContractCallback(
        state_normalizer=artifacts.state,
        action_normalizer=artifacts.future,
        valid_windows_manifest=artifacts.manifest,
        official_checkpoint=tmp_path / "official",
        representation=REPRESENTATION,
        right_hand_codec=artifacts.codecs[0],
        left_hand_codec=artifacts.codecs[1],
    )
    model = SimpleNamespace(whole_action_loss=True, action_representation="legacy_local_delta_absolute_hand_v1")
    with pytest.raises(ValueError, match="representations differ"):
        callback.on_load_checkpoint_start(model)
