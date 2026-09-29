"""The AR V0.2 launch path delegates to the native Cosmos CLI and Trainer."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_2.sh"
COMMON = REPO_ROOT / "packages/cosmos3/examples/_sft_launcher_common.sh"
VERIFIER = REPO_ROOT / "scripts/verify_ar_v02_native_training.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_ar_v02_native_training", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_launch_scripts_have_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(COMMON)], check=True)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_fixed_recipe_uses_verified_cudnn_settings_without_changing_legacy():
    from cosmos3_joint_video_hand_pose.src.config import _ar_v02_experiment, _ar_v02_fixed_camera_experiment
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    fixed = _ar_v02_fixed_camera_experiment()
    legacy = _ar_v02_experiment(True)
    assert fixed.trainer.cudnn == {"benchmark": False, "deterministic": True}
    assert legacy.trainer.cudnn == {"benchmark": True, "deterministic": False}
    composed = load_experiment_from_toml(REPO_ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_2_fixed_camera.toml", [])
    assert composed.trainer.cudnn == fixed.trainer.cudnn


def test_project_launcher_uses_common_plumbing_and_registered_cli(tmp_path):
    base = tmp_path / "base_dcp"
    tokenizer = tmp_path / "text_tokenizer"
    base.mkdir()
    tokenizer.mkdir()
    vae = tmp_path / "vae.pth"
    vae.write_bytes(b"vae")
    (tokenizer / "tokenizer_config.json").write_text('{"tokenizer_class":"TestTokenizer"}')
    (tokenizer / "tokenizer.json").write_text("{}")

    capture = tmp_path / "capture.txt"
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    torchrun = binary_dir / "torchrun"
    torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "{\n"
        "  printf 'PWD=%s\\n' \"$PWD\"\n"
        "  printf 'PYTHONPATH=%s\\n' \"$PYTHONPATH\"\n"
        "  printf 'BASE_CHECKPOINT_PATH=%s\\n' \"$BASE_CHECKPOINT_PATH\"\n"
        "  printf 'WAN_VAE_PATH=%s\\n' \"$WAN_VAE_PATH\"\n"
        "  printf 'TEXT_TOKENIZER_PATH=%s\\n' \"$TEXT_TOKENIZER_PATH\"\n"
        "  printf 'ARG=%s\\n' \"$@\"\n"
        "} > \"$CAPTURE_FILE\"\n"
    )
    torchrun.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        PATH=f"{binary_dir}:{environment['PATH']}",
        CAPTURE_FILE=str(capture),
        BASE_CHECKPOINT_PATH=str(base),
        WAN_VAE_PATH=str(vae),
        TEXT_TOKENIZER_PATH=str(tokenizer),
        OUTPUT_ROOT=str(tmp_path / "output"),
        IMAGINAIRE_OUTPUT_ROOT=str(tmp_path / "output"),
        NPROC_PER_NODE="1",
    )
    subprocess.run(["bash", str(LAUNCHER)], cwd=tmp_path, env=environment, check=True)
    observed = capture.read_text()
    assert f"PWD={REPO_ROOT}" in observed
    assert f"PYTHONPATH={REPO_ROOT}:{REPO_ROOT / 'packages/cosmos3'}" in observed
    assert "ARG=-m\nARG=cosmos3_joint_video_hand_pose.src.train" in observed
    assert f"ARG=--sft-toml={REPO_ROOT}/cosmos3_joint_video_hand_pose/configs/ar_v0_2_fixed_camera.toml" in observed
    assert f"BASE_CHECKPOINT_PATH={base}" in observed
    assert f"WAN_VAE_PATH={vae}" in observed
    assert f"TEXT_TOKENIZER_PATH={tokenizer}" in observed


def test_native_lifecycle_verifier_only_builds_official_short_step_phases(tmp_path):
    verifier = _load_verifier()
    assert verifier.phase_overrides(4).split() == [
        "trainer.max_iter=4",
        "checkpoint.save_iter=2",
        "job.group=ar_v0_2_native_lifecycle",
        "job.name=save_resume_smoke",
        "job.wandb_mode=online",
    ]
    assert verifier.phase_overrides(6).startswith("trainer.max_iter=6 ")
    assert verifier.preflight_status([]) == "inputs_present_unverified"
    assert verifier.preflight_status(["missing"]) == "blocked"
    source = VERIFIER.read_text()
    assert "training_step" not in source and "optimizer.step" not in source

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    missing = verifier.missing_training_inputs(
        base_checkpoint=tmp_path / "missing_dcp",
        vae=tmp_path / "missing_vae",
        text_tokenizer=tmp_path / "missing_tokenizer",
        prepared=prepared,
    )
    assert any("missing_dcp" in item for item in missing)
    assert all(any(name in item for item in missing) for name in verifier.REQUIRED_PREPARED_FILES)
    codecs = tmp_path / "separate_codec_artifacts"
    codecs.mkdir()
    paths = tuple(codecs / (side + "_pca15.pt") for side in ("right", "left"))
    for path in paths:
        path.write_bytes(b"presence check only")
        path.with_suffix(".validation.json").write_text("{}")
    missing = verifier.missing_training_inputs(
        base_checkpoint=tmp_path / "missing_dcp", vae=tmp_path / "missing_vae",
        text_tokenizer=tmp_path / "missing_tokenizer", prepared=prepared, codec_paths=paths)
    assert not any("pca15" in item or "mlp15" in item for item in missing)
    paths[0].with_suffix(".validation.json").unlink()
    missing = verifier.missing_training_inputs(
        base_checkpoint=tmp_path / "missing_dcp", vae=tmp_path / "missing_vae",
        text_tokenizer=tmp_path / "missing_tokenizer", prepared=prepared, codec_paths=paths)
    assert str(paths[0].with_suffix(".validation.json")) in missing


def test_lifecycle_verification_requires_fresh_output_and_checkpoint_evidence(tmp_path, monkeypatch):
    verifier = _load_verifier()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="new path"):
        verifier.require_new_output_root(existing)
    verifier.require_new_output_root(tmp_path / "new")
    verifier.require_single_node_environment({})
    verifier.require_single_node_environment({"NNODES": "", "NODE_RANK": ""})
    verifier.require_single_node_environment({"NNODES": "1", "NODE_RANK": "0"})
    with pytest.raises(RuntimeError, match="single-node only"):
        verifier.require_single_node_environment({"NNODES": "2", "NODE_RANK": "0"})
    with pytest.raises(RuntimeError, match="single-node only"):
        verifier.require_single_node_environment({"NNODES": "1", "NODE_RANK": "1"})

    output = tmp_path / "output"
    checkpoints = verifier._run_directory(output) / "checkpoints"
    saved = checkpoints / "iter_000000004"
    for name in ("model", "optim", "scheduler", "trainer"):
        component = saved / name
        component.mkdir(parents=True)
        (component / ".metadata").write_bytes(b"metadata")
        (component / "__0_0.distcp").write_bytes(b"state")
    for rank in range(2):
        loader = saved / "dataloader" / f"rank_{rank}.pkl"
        loader.parent.mkdir(parents=True, exist_ok=True)
        loader.write_bytes(b"loader")
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "latest_checkpoint.txt").write_text("iter_000000004\n")
    monkeypatch.setattr(verifier, "_read_dcp_metadata", lambda _path: 1)
    monkeypatch.setattr(verifier, "_read_trainer_iteration", lambda _path: 4)
    verified = verifier.verify_checkpoint(output, iteration=4, nproc_per_node=2)
    assert verified["iteration"] == 4 and verified["dataloader_ranks"] == 2

    (saved / "optim" / "__0_0.distcp").unlink()
    with pytest.raises(RuntimeError, match="incomplete optim DCP"):
        verifier.verify_checkpoint(output, iteration=4, nproc_per_node=2)


def test_resume_log_must_name_the_exact_step_four_checkpoint(tmp_path):
    verifier = _load_verifier()
    source = verifier._run_directory(tmp_path) / "checkpoints/iter_000000004"
    log = tmp_path / "logs/native_trainer_to_6.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        f"Resuming ckpt {source} (same-job, local) with keys: "
        "['dataloader', 'model', 'optim', 'scheduler', 'trainer']\n"
        f"Loaded checkpoint from {source} (same-job, local) in iteration 4\n"
    )
    evidence = verifier.verify_resume_log(tmp_path, source_iteration=4)
    assert evidence["log"] == str(log)
    log.write_text("process exited zero without resume evidence\n")
    with pytest.raises(RuntimeError, match="did not prove same-job resume"):
        verifier.verify_resume_log(tmp_path, source_iteration=4)
