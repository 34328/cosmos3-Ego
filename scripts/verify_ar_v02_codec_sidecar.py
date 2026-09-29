"""CPU-only sidecar verification; never changes a checkpoint or runtime contract.

Historical codec identity is NOT recoverable from a v1 contract without codec
hashes. This verifier records present compatibility and can fail on future drift
using --reference. Inputs must be trusted local project artifacts (DCP is pickle).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from cosmos3_joint_video_hand_pose.src.action import Action57Builder
from cosmos3_joint_video_hand_pose.src.codec import FrozenHandMLPAE15
from cosmos3_joint_video_hand_pose.src.ar_v02_contract import ARTrainingContract

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = "60-64-SiLU-32-SiLU-15 / 15-32-SiLU-64-SiLU-60"
INPUT_CONTRACT = "20 non-wrist points in current-frame wrist-local coordinates, flattened to 60D"
BINDING = "net.ar_training_contract._extra_state"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_payload(payload):
    require(payload.get("architecture") == ARCHITECTURE, "codec architecture mismatch")
    require(payload.get("input_contract") == INPUT_CONTRACT, "codec input contract mismatch")
    shapes = {"mean": (60,), "std": (60,)}
    for name, dims in (("encoder", (60, 64, 32, 15)), ("decoder", (15, 32, 64, 60))):
        for layer, (din, dout) in zip((0, 2, 4), zip(dims, dims[1:])):
            shapes[f"{name}.{layer}.weight"] = (dout, din)
            shapes[f"{name}.{layer}.bias"] = (dout,)
    state = payload["state_dict"]
    require(set(state) == set(shapes), "codec tensor key mismatch")
    tensors = dict(state, latent_mean=payload["latent_mean"], latent_std=payload["latent_std"])
    shapes.update(latent_mean=(15,), latent_std=(15,))
    for name, shape in shapes.items():
        value = tensors[name]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape, f"codec shape mismatch: {name}")
        require(value.is_floating_point() and torch.isfinite(value).all().item(), f"non-finite codec tensor: {name}")
    for name in ("std", "latent_std"):
        require((tensors[name] > 0).all().item(), f"non-positive codec scale: {name}")
    return sum(value.numel() for name, value in state.items() if "." in name)


def verify_codecs(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["contract"]
    require(contract["input_shape_per_hand"] == [20, 3] and contract["latent_dim_per_hand"] == 15,
            "manifest dimensions mismatch")
    require(contract["sides_are_independent"] is True and contract["same_codec_for_initial_and_future"] is True,
            "manifest codec selection contract mismatch")
    selected = manifest["option_b_mlp15"]["primary"]
    defaults = inspect.signature(Action57Builder).parameters
    codecs = {}
    for side in ("right", "left"):
        entry = selected[side]
        path = (manifest_path.parent / entry["path"]).resolve()
        require(path.is_relative_to(manifest_path.parent), "codec path escapes artifact root")
        actual = sha256(path)
        require(actual == entry["sha256"], f"{side} codec SHA256 mismatch")
        require(path == Path(defaults[f"{side}_codec"].default).resolve(), f"{side} runtime default path mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        count = validate_payload(payload)
        require(count == manifest["option_b_mlp15"]["trainable_parameter_count_per_side"], "parameter count mismatch")
        codec = FrozenHandMLPAE15(path)
        points = torch.linspace(-0.1, 0.1, 120).reshape(2, 20, 3)
        latent = codec.encode(points)
        decoded = codec.decode(latent)
        require(tuple(latent.shape) == (2, 15) and tuple(decoded.shape) == (2, 20, 3), "codec smoke shape mismatch")
        require(torch.isfinite(latent).all().item() and torch.isfinite(decoded).all().item(), "codec smoke non-finite")
        require(not codec.training and not any(p.requires_grad for p in codec.parameters()), "codec not frozen")
        require(sha256(path) == actual, "codec changed during verification")
        codecs[side] = dict(path=str(path), sha256=actual, parameter_count=count,
                            architecture=payload["architecture"], cpu_encode_decode_finite=True,
                            episode_provenance_present=bool(payload.get("fit", {}).get("episode_ids")))
    require(sha256(manifest_path) == digest, "manifest changed during verification")
    return dict(manifest_path=str(manifest_path), manifest_sha256=digest,
                option="option_b_mlp15", selection="primary", codecs=codecs,
                provenance=selected["selection_note"],
                wrist_local_v1_reuse=dict(approved=False,
                    reason="Legacy verification does not prove training episodes are disjoint from current heldout; do not reuse without independently traced provenance and new quality gates."))


def verify_checkpoint(checkpoint, prepared):
    checkpoint, prepared = Path(checkpoint).resolve(), Path(prepared).resolve()
    metadata_path = checkpoint / "model/.metadata"
    metadata_sha = sha256(metadata_path)
    files = dict(state_normalizer=prepared / "chunk_state_normalizer.json",
                 action_normalizer=prepared / "future_frame_delta_normalizer.json",
                 manifest=prepared / "valid_windows.json")
    hashes = {name: sha256(path) for name, path in files.items()}
    contract = ARTrainingContract(state_normalizer=files["state_normalizer"],
                                  action_normalizer=files["action_normalizer"],
                                  manifest_sha256=hashes["manifest"])
    reader = dcp.FileSystemReader(checkpoint / "model")
    metadata = reader.read_metadata()
    keys = [key for key in metadata.state_dict_metadata if key.startswith(BINDING + ".")]
    require(bool(keys), "checkpoint missing v1 training contract")
    expected = contract.get_extra_state()
    # This existing contract owns strict comparison; do not add fields to it.
    state = {BINDING: expected}
    dcp.load(state, storage_reader=reader, no_dist=True)
    contract.set_extra_state(state[BINDING])
    require(not any("codec" in key for key in keys), "codec-aware checkpoint requires a versioned verifier")
    require(sha256(metadata_path) == metadata_sha, "checkpoint metadata changed during verification")
    require(all(sha256(files[name]) == digest for name, digest in hashes.items()), "data artifacts changed during verification")
    # Fingerprint the actual serialized binding bytes, not the huge model shards.
    byte_hash = hashlib.sha256()
    for index, info in sorted(metadata.storage_data.items(), key=lambda item: item[0].fqn):
        if not index.fqn.startswith(BINDING + "."):
            continue
        with (checkpoint / "model" / info.relative_path).open("rb") as stream:
            stream.seek(info.offset)
            data = stream.read(info.length)
        require(len(data) == info.length, "truncated checkpoint binding")
        byte_hash.update(index.fqn.encode() + b"\0" + data)
    return dict(path=str(checkpoint), metadata_sha256=metadata_sha,
                contract_bytes_sha256=byte_hash.hexdigest(), contract_schema=expected["schema"],
                contract_leaf_count=len(keys), strict_existing_contract_passed=True,
                data_hashes=hashes, codec_identity_embedded=False,
                model_tensor_integrity_checked=False)


def verify_reference(report, reference):
    for field in ("codec_identity", "checkpoint"):
        require(report[field] == reference[field], f"sidecar identity drift: {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--codec-manifest", type=Path, default=ROOT / "cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v2_4/manifest.json")
    parser.add_argument("--run-snapshot", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Reserve before work: never overwrite previous verification evidence.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        report = dict(schema="ar_v02_codec_sidecar_v1", checked_at=datetime.now(timezone.utc).isoformat())
        try:
            torch.set_num_threads(1)
            report["codec_identity"] = verify_codecs(args.codec_manifest)
            report["checkpoint"] = verify_checkpoint(args.checkpoint, args.prepared)
            source_path = args.run_snapshot / "source_manifest.json"
            sources = json.loads(source_path.read_text())
            source_checks = {}
            for name in ("action.py", "codec.py"):
                path = Path("cosmos3_joint_video_hand_pose/src") / name
                source_checks[str(path)] = sha256(ROOT / path) == sources.get(str(path))
            require(all(source_checks.values()), "codec source differs from training snapshot")
            recorded_data_path = args.run_snapshot / "data_hashes.json"
            recorded_data = json.loads(recorded_data_path.read_text())
            for filename in ("chunk_state_normalizer.json", "future_frame_delta_normalizer.json", "valid_windows.json"):
                path = (args.prepared / filename).resolve()
                require(recorded_data.get(str(path.relative_to(ROOT))) == sha256(path), "training snapshot data hash mismatch")
            report["run_snapshot"] = dict(path=str(args.run_snapshot.resolve()),
                source_manifest_sha256=sha256(source_path), data_hashes_sha256=sha256(recorded_data_path),
                codec_source_matches=source_checks,
                historical_weight_hashes_recorded=any("mlp15_primary.pt" in k for k in sources))
            report["historical_codec_identity_proven"] = False
            report["limitations"] = [
                "v1 checkpoint and run snapshot do not bind codec weight hashes; current compatibility only",
                "sidecar is opt-in preflight, not automatically enforced by training or inference",
                "metadata and contract bytes fingerprinted; model tensor shards are not hashed",
                "artifact manifest is local provenance, not a signed historical attestation"]
            if args.reference:
                verify_reference(report, json.loads(args.reference.read_text()))
            report["status"] = "compatible_current_artifacts"
        except (Exception, CheckpointException) as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(dict(status=report["status"], output=str(args.output), error=report.get("error"))))
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
