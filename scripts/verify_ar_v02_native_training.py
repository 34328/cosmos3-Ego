#!/usr/bin/env python3
"""Preflight and verify an optional official-Trainer save/resume smoke for AR V0.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
COSMOS_ROOT = REPO_ROOT / "packages/cosmos3"
LAUNCHER = REPO_ROOT / "cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_2.sh"
TOML_PATH = REPO_ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_2_fixed_camera.toml"
DEFAULT_BASE_CHECKPOINT = Path("/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp")
DEFAULT_VAE = Path("/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
DEFAULT_TEXT_TOKENIZER = Path("/mnt/checkpoints/Cosmos3-Nano/text_tokenizer")
REQUIRED_PREPARED_FILES = (
    "chunk_state_normalizer.json",
    "future_normalizer.json",
    "valid_windows.json",
)
RUN_PROJECT = "joint_video_hand_pose"
RUN_GROUP = "ar_v0_2_native_lifecycle"
RUN_NAME = "save_resume_smoke"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(path: Path) -> dict[str, object]:
    """Read identity-bearing tokenizer files without judging the directory name."""
    config_path = path / "tokenizer_config.json"
    vocabulary_candidates = (
        path / "tokenizer.json",
        path / "tokenizer.model",
        path / "vocab.json",
    )
    if not path.is_dir() or not config_path.is_file():
        raise FileNotFoundError(f"text tokenizer identity is incomplete: {path}")
    vocabulary = next((candidate for candidate in vocabulary_candidates if candidate.is_file()), None)
    if vocabulary is None:
        raise FileNotFoundError(f"text tokenizer vocabulary is missing: {path}")
    payload = json.loads(config_path.read_text())
    files = [config_path, vocabulary]
    for name in ("special_tokens_map.json", "added_tokens.json", "merges.txt"):
        candidate = path / name
        if candidate.is_file():
            files.append(candidate)
    return {
        "path": str(path.resolve()),
        "tokenizer_class": payload.get("tokenizer_class"),
        "model_max_length": payload.get("model_max_length"),
        "special_tokens": {
            key: payload.get(key)
            for key in ("bos_token", "eos_token", "pad_token", "unk_token")
            if key in payload
        },
        "sha256": {item.name: _sha256(item) for item in files},
    }


def tokenizer_compatibility(path: Path, base_checkpoint: Path) -> dict[str, object]:
    """Cross-check tokenizer semantics against the declared source and DCP tensors.

    The DCP records a model name but no immutable source revision.  This proves
    structural/token-ID compatibility, while reporting that exact provenance is
    not cryptographically established.
    """
    dcp_config_path = base_checkpoint / "model/config.json"
    model_config_path = (
        COSMOS_ROOT
        / "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    )
    dcp_config = json.loads(dcp_config_path.read_text())
    model_config = json.loads(model_config_path.read_text())
    vlm = dcp_config["model"]["config"]["vlm_config"]
    declared_source = vlm["tokenizer"]["pretrained_model_name"]
    expected = {
        "bos_token_id": model_config["text_config"]["bos_token_id"],
        "eos_token_id": model_config["text_config"]["eos_token_id"],
        "vision_start_token_id": model_config["vision_start_token_id"],
        "vision_end_token_id": model_config["vision_end_token_id"],
        "image_token_id": model_config["image_token_id"],
        "video_token_id": model_config["video_token_id"],
    }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokens = {
        "bos_token_id": "<|endoftext|>",
        "eos_token_id": "<|im_end|>",
        "vision_start_token_id": "<|vision_start|>",
        "vision_end_token_id": "<|vision_end|>",
        "image_token_id": "<|image_pad|>",
        "video_token_id": "<|video_pad|>",
    }
    actual = {key: tokenizer.convert_tokens_to_ids(token) for key, token in tokens.items()}

    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(str(base_checkpoint / "model")).read_metadata().state_dict_metadata
    embedding_shape = tuple(metadata["net.language_model.model.embed_tokens.weight"].size)
    head_shape = tuple(metadata["net.language_model.lm_head.weight"].size)
    model_vocab_size = model_config["text_config"]["vocab_size"]
    model_hidden_size = model_config["text_config"]["hidden_size"]
    packaged_root = path.parent
    packaged_vocab = packaged_root / "vocab.json"
    same_packaged_vocab = packaged_vocab.is_file() and (path / "vocab.json").is_file()
    if same_packaged_vocab:
        same_packaged_vocab = _sha256(packaged_vocab) == _sha256(path / "vocab.json")
    compatible = (
        declared_source == "Qwen/Qwen3-VL-8B-Instruct"
        and expected == actual
        and embedding_shape == (model_vocab_size, model_hidden_size)
        and head_shape == embedding_shape
        and len(tokenizer) <= model_vocab_size
        and same_packaged_vocab
    )
    return {
        "declared_source": declared_source,
        "declared_source_revision": None,
        "exact_source_revision_verified": False,
        "compatible_with_declared_source": compatible,
        "tokenizer_class": type(tokenizer).__name__,
        "base_vocab_size": tokenizer.vocab_size,
        "tokenizer_length_with_added_tokens": len(tokenizer),
        "model_vocab_size": model_vocab_size,
        "model_hidden_size": model_hidden_size,
        "expected_token_ids": expected,
        "actual_token_ids": actual,
        "dcp_embedding_shape": embedding_shape,
        "dcp_lm_head_shape": head_shape,
        "vocab_matches_packaged_cosmos3_nano": same_packaged_vocab,
        "provenance_limit": "DCP config has no immutable tokenizer revision/hash",
    }


def missing_training_inputs(
    *, base_checkpoint: Path, vae: Path, text_tokenizer: Path, prepared: Path,
    codec_paths: tuple[Path, Path] | None = None,
) -> list[str]:
    missing = []
    if not base_checkpoint.is_dir():
        missing.append(str(base_checkpoint))
    if not vae.is_file():
        missing.append(str(vae))
    try:
        tokenizer_identity(text_tokenizer)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        missing.append(str(error))
    missing.extend(str(prepared / name) for name in REQUIRED_PREPARED_FILES if not (prepared / name).is_file())
    # New codecs live in their versioned artifact directory, not the statistics directory.
    if codec_paths is None:
        codec_paths = (prepared / "right_mlp15.pt", prepared / "left_mlp15.pt")
    for path in codec_paths:
        missing.extend(str(item) for item in (path, path.with_suffix(".validation.json")) if not item.is_file())
    return missing


def resolved_config_paths() -> dict[str, str]:
    """Compose through the official TOML loader and force interpolation resolution."""
    for path in (str(REPO_ROOT), str(COSMOS_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from cosmos3_joint_video_hand_pose.src import config as _registration  # noqa: F401
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    config = load_experiment_from_toml(TOML_PATH, [])
    dataset = config.dataloader_train.dataloader.datasets.egoverse.dataset
    artifact_paths = (
        Path(str(dataset.chunk_state_normalizer)),
        Path(str(dataset.future_normalizer)),
        Path(str(dataset.valid_windows_manifest)),
    )
    artifact_parents = {path.parent.resolve() for path in artifact_paths}
    if len(artifact_parents) != 1:
        raise RuntimeError(f"fixed-camera statistics do not share one prepared directory: {artifact_paths}")
    result = {
        "base_checkpoint": str(config.checkpoint.load_path),
        "vae": str(config.model.config.tokenizer.vae_path),
        "text_tokenizer": str(config.model.config.vlm_config.tokenizer.pretrained_model_name),
        "contract_checkpoint": str(config.trainer.callbacks.ar_v02_contract.official_checkpoint),
        "prepared": str(artifact_parents.pop()),
        "right_codec": str(dataset.right_codec),
        "left_codec": str(dataset.left_codec),
    }
    unresolved = {key: value for key, value in result.items() if "${oc.env:" in value}
    if unresolved:
        raise RuntimeError(f"unresolved path interpolation: {unresolved}")
    return result


def phase_overrides(max_iter: int) -> str:
    return " ".join(
        (
            f"trainer.max_iter={max_iter}",
            "checkpoint.save_iter=2",
            f"job.group={RUN_GROUP}",
            f"job.name={RUN_NAME}",
            "job.wandb_mode=online",
        )
    )


def preflight_status(missing: list[str]) -> str:
    return "blocked" if missing else "inputs_present_unverified"


def require_new_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"--output-root must be a new path: {output_root}")


def require_single_node_environment(environment: dict[str, str] | os._Environ[str]) -> None:
    nnodes = environment.get("NNODES") or "1"
    node_rank = environment.get("NODE_RANK") or "0"
    if nnodes != "1" or node_rank != "0":
        raise RuntimeError(
            "native lifecycle verification is single-node only; "
            f"got NNODES={nnodes!r}, NODE_RANK={node_rank!r}"
        )


def _run_phase(*, max_iter: int, output_root: Path, nproc_per_node: int) -> None:
    environment = os.environ.copy()
    environment.update(
        IMAGINAIRE_OUTPUT_ROOT=str(output_root),
        OUTPUT_ROOT=str(output_root),
        LOG_FILENAME=f"native_trainer_to_{max_iter}.log",
        NPROC_PER_NODE=str(nproc_per_node),
        EXTRA_TAIL_OVERRIDES=phase_overrides(max_iter),
    )
    subprocess.run(["bash", str(LAUNCHER)], cwd=REPO_ROOT, env=environment, check=True)


def _run_directory(output_root: Path) -> Path:
    return output_root / RUN_PROJECT / RUN_GROUP / RUN_NAME


def _read_dcp_metadata(component: Path) -> int:
    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(str(component)).read_metadata().state_dict_metadata
    if not metadata:
        raise RuntimeError(f"empty DCP metadata: {component}")
    return len(metadata)


def _read_trainer_iteration(trainer: Path) -> int:
    import torch.distributed.checkpoint as dcp

    state = {"iteration": 0}
    dcp.load(state, storage_reader=dcp.FileSystemReader(str(trainer)))
    return int(state["iteration"])


def verify_checkpoint(
    output_root: Path, *, iteration: int, nproc_per_node: int, require_latest: bool = True
) -> dict[str, object]:
    checkpoints = _run_directory(output_root) / "checkpoints"
    name = f"iter_{iteration:09d}"
    saved = checkpoints / name
    if require_latest:
        latest = checkpoints / "latest_checkpoint.txt"
        if not latest.is_file() or latest.read_text().strip() != name:
            raise RuntimeError(f"latest checkpoint is not {name}: {latest}")
    metadata_entries = {}
    for component_name in ("model", "optim", "scheduler", "trainer"):
        component = saved / component_name
        if not (component / ".metadata").is_file() or not any(component.glob("*.distcp")):
            raise RuntimeError(f"incomplete {component_name} DCP: {component}")
        metadata_entries[component_name] = _read_dcp_metadata(component)
    missing_ranks = [
        rank for rank in range(nproc_per_node)
        if not (saved / "dataloader" / f"rank_{rank}.pkl").is_file()
    ]
    if missing_ranks:
        raise RuntimeError(f"checkpoint missing dataloader ranks {missing_ranks}: {saved}")
    restored_iteration = _read_trainer_iteration(saved / "trainer")
    if restored_iteration != iteration:
        raise RuntimeError(
            f"trainer iteration mismatch at {saved}: expected {iteration}, got {restored_iteration}"
        )
    return {
        "path": str(saved),
        "iteration": restored_iteration,
        "metadata_entries": metadata_entries,
        "dataloader_ranks": nproc_per_node,
    }


def verify_resume_log(output_root: Path, *, source_iteration: int) -> dict[str, str]:
    source = _run_directory(output_root) / "checkpoints" / f"iter_{source_iteration:09d}"
    log_path = output_root / "logs/native_trainer_to_6.log"
    text = log_path.read_text()
    selected = f"Resuming ckpt {source} (same-job, local) with keys:"
    loaded = f"Loaded checkpoint from {source} (same-job, local) in iteration {source_iteration}"
    if selected not in text or loaded not in text:
        raise RuntimeError(f"stage 2 did not prove same-job resume from iteration {source_iteration}: {log_path}")
    return {"log": str(log_path), "selected": selected, "loaded": loaded}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true",
        help="use a new output root, run 4-step save/resume-to-6, then verify DCP and resume evidence",
    )
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "outputs/validation/ar_v02_native_lifecycle",
    )
    parser.add_argument(
        "--base-checkpoint", type=Path,
        default=Path(os.environ.get("BASE_CHECKPOINT_PATH", DEFAULT_BASE_CHECKPOINT)),
    )
    parser.add_argument(
        "--vae", type=Path,
        default=Path(os.environ.get("WAN_VAE_PATH", DEFAULT_VAE)),
    )
    parser.add_argument(
        "--text-tokenizer", type=Path,
        default=Path(os.environ.get("TEXT_TOKENIZER_PATH", DEFAULT_TEXT_TOKENIZER)),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    require_single_node_environment(os.environ)
    os.environ.update(
        BASE_CHECKPOINT_PATH=str(args.base_checkpoint),
        WAN_VAE_PATH=str(args.vae),
        TEXT_TOKENIZER_PATH=str(args.text_tokenizer),
    )
    resolved = resolved_config_paths()
    expected = {
        "base_checkpoint": str(args.base_checkpoint),
        "vae": str(args.vae),
        "text_tokenizer": str(args.text_tokenizer),
        "contract_checkpoint": str(args.base_checkpoint),
    }
    if {key: resolved[key] for key in expected} != expected:
        raise RuntimeError(f"effective paths differ from requested paths: resolved={resolved}, expected={expected}")

    prepared = Path(resolved["prepared"])

    missing = missing_training_inputs(
        base_checkpoint=args.base_checkpoint,
        vae=args.vae,
        text_tokenizer=args.text_tokenizer,
        prepared=prepared,
        codec_paths=(Path(resolved["right_codec"]), Path(resolved["left_codec"])),
    )
    try:
        identity: dict[str, object] = tokenizer_identity(args.text_tokenizer)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        identity = {"path": str(args.text_tokenizer), "error": str(error)}
    try:
        compatibility: dict[str, object] = tokenizer_compatibility(
            args.text_tokenizer, args.base_checkpoint
        )
        if not compatibility["compatible_with_declared_source"]:
            missing.append("text tokenizer is incompatible with the DCP-declared source")
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        compatibility = {"compatible_with_declared_source": False, "error": str(error)}
        missing.append(f"text tokenizer compatibility check failed: {error}")
    report = {
        "status": preflight_status(missing),
        "lifecycle_verified": False,
        "lifecycle_scope": "single_node_only",
        "multi_node_lifecycle_verified": False,
        "official_cli": "cosmos3_joint_video_hand_pose.src.train -> cosmos_framework.scripts.train",
        "resolved_paths": resolved,
        "tokenizer_identity": identity,
        "tokenizer_compatibility": compatibility,
        "missing": missing,
        "phases": [phase_overrides(4), phase_overrides(6)],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if missing:
        return 2
    if args.execute:
        require_new_output_root(args.output_root)
        _run_phase(max_iter=4, output_root=args.output_root, nproc_per_node=args.nproc_per_node)
        first = verify_checkpoint(
            args.output_root, iteration=4, nproc_per_node=args.nproc_per_node
        )
        _run_phase(max_iter=6, output_root=args.output_root, nproc_per_node=args.nproc_per_node)
        resume = verify_resume_log(args.output_root, source_iteration=4)
        second = verify_checkpoint(
            args.output_root, iteration=6, nproc_per_node=args.nproc_per_node
        )
        lifecycle = {
            **report,
            "status": "lifecycle_verified",
            "lifecycle_verified": True,
            "first_checkpoint": first,
            "resume_evidence": resume,
            "final_checkpoint": second,
        }
        summary = args.output_root / "native_lifecycle_verification.json"
        summary.write_text(json.dumps(lifecycle, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(lifecycle, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
