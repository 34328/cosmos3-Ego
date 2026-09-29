"""CPU regressions for opt-in codec sidecars; no runtime contract mutation."""
import copy
import json

import pytest
import torch
import torch.distributed.checkpoint as dcp

from scripts.verify_ar_v02_codec_sidecar import (
    ROOT, BINDING, sha256,
    validate_payload, verify_codecs, verify_checkpoint, verify_reference,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_contract import ARTrainingContract

MANIFEST = ROOT / "cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v2_4/manifest.json"


def payload():
    return torch.load(MANIFEST.parent / "option_b_mlp15/left_mlp15_primary.pt",
                      map_location="cpu", weights_only=True)


def test_real_artifact_and_runtime_defaults():
    result = verify_codecs(MANIFEST)
    assert result["codecs"]["left"]["sha256"] != result["codecs"]["right"]["sha256"]
    assert result["selection"] == "primary"
    assert not result["wrist_local_v1_reuse"]["approved"]
    assert all(not side["episode_provenance_present"] for side in result["codecs"].values())


@pytest.mark.parametrize("mutation", ["architecture", "input", "shape", "nan", "scale", "extra"])
def test_invalid_codec_payload_rejected(mutation):
    value = payload()
    if mutation == "architecture":
        value["architecture"] = "different"
    elif mutation == "input":
        value["input_contract"] = "world coordinates"
    elif mutation == "shape":
        value["latent_mean"] = torch.zeros(14)
    elif mutation == "nan":
        value["state_dict"]["encoder.0.weight"][0, 0] = float("nan")
    elif mutation == "scale":
        value["latent_std"][0] = 0
    else:
        value["state_dict"]["unexpected"] = torch.zeros(1)
    with pytest.raises(ValueError):
        validate_payload(value)


def test_swapped_side_hash_rejected(tmp_path, monkeypatch):
    import inspect
    import shutil
    from scripts import verify_ar_v02_codec_sidecar as validator

    manifest = json.loads(MANIFEST.read_text())
    signature = inspect.signature(validator.Action57Builder)
    parameters = dict(signature.parameters)
    for side in ("left", "right"):
        entry = manifest["option_b_mlp15"]["primary"][side]
        destination = tmp_path / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MANIFEST.parent / entry["path"], destination)
        parameters[f"{side}_codec"] = parameters[f"{side}_codec"].replace(default=destination)
    fake_signature = signature.replace(parameters=list(parameters.values()))
    monkeypatch.setattr(validator.inspect, "signature", lambda _: fake_signature)
    manifest["option_b_mlp15"]["primary"]["left"]["sha256"] = manifest["option_b_mlp15"]["primary"]["right"]["sha256"]
    fake = tmp_path / "manifest.json"
    fake.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="left codec SHA256 mismatch"):
        verify_codecs(fake)


def fixture_checkpoint(tmp_path, missing=False):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "valid_windows.json").write_text("{}")
    digest = sha256(prepared / "valid_windows.json")
    (prepared / "chunk_state_normalizer.json").write_text(json.dumps(
        dict(schema="chunk_camera_test", frozen=True, split="train", manifest_sha256=digest)))
    (prepared / "future_frame_delta_normalizer.json").write_text('{"test": 1}')
    contract = ARTrainingContract(
        state_normalizer=prepared / "chunk_state_normalizer.json",
        action_normalizer=prepared / "future_frame_delta_normalizer.json", manifest_sha256=digest)
    state = contract.get_extra_state()
    if missing:
        del state["artifacts"]["action_normalizer"]["sha256"]
    checkpoint = tmp_path / "checkpoint"
    dcp.save({BINDING: state}, checkpoint_id=checkpoint / "model", no_dist=True)
    return checkpoint, prepared


def test_existing_v1_binding_passes_without_codec_migration(tmp_path):
    checkpoint, prepared = fixture_checkpoint(tmp_path)
    before = sha256(checkpoint / "model/.metadata")
    result = verify_checkpoint(checkpoint, prepared)
    assert result["strict_existing_contract_passed"]
    assert not result["codec_identity_embedded"]
    assert sha256(checkpoint / "model/.metadata") == before


def test_changed_normalizer_rejected(tmp_path):
    checkpoint, prepared = fixture_checkpoint(tmp_path)
    (prepared / "future_frame_delta_normalizer.json").write_text('{"test": 2}')
    with pytest.raises(ValueError, match="contract mismatch"):
        verify_checkpoint(checkpoint, prepared)


def test_missing_binding_leaf_rejected(tmp_path):
    checkpoint, prepared = fixture_checkpoint(tmp_path, missing=True)
    # PyTorch wraps planner errors in CheckpointException (BaseException).
    from torch.distributed.checkpoint.api import CheckpointException
    with pytest.raises((ValueError, CheckpointException)):
        verify_checkpoint(checkpoint, prepared)


@pytest.mark.parametrize("field", ["codec_identity", "checkpoint"])
def test_sidecar_detects_future_drift(field):
    report = dict(codec_identity={"sha": "a"}, checkpoint={"sha": "b"})
    reference = copy.deepcopy(report)
    reference[field]["sha"] = "changed"
    with pytest.raises(ValueError, match="identity drift"):
        verify_reference(report, reference)
