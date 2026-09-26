#!/usr/bin/env python3
r"""Check that DCP checkpoints load into EgoVerseOmniMoTModel on cosmos-framework cf5d68c.

Builds the experiment model exactly as the training config does (single rank,
no sharding), compares its state-dict keys/shapes with the checkpoint's DCP
metadata, then loads the weights through the framework's inference loader
(``utils.generator.model_loader._load_model``, the same ``CustomLoadPlanner``
used by training). Optionally runs one no-grad ``training_step`` on a real
packed batch.

Typical targets are the Cosmos3-Nano SFT DCP checkpoint (``.../iter_XXXXXXXXX/model``;
its ``net_ema.*`` tensors are reported as unexpected keys and skipped via
``checkpoint.keys_to_skip_loading``) and a v0.6 training checkpoint
(``outputs/.../checkpoints/iter_XXXXXXXXX/model``). ``--toml`` selects the
experiment used to build the model; it defaults to the v0.6 smoke-train TOML.

With ``--forward``, ``egoverse_action_override_used`` in the result is true only
if that forward pass reached the EgoVerse visibility-weighted action loss
(``EgoVerseOmniMoTModel._compute_flow_matching_loss``) rather than the native
flow-matching loss.

Run with one GPU from the repository root:
    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD:$PWD/packages/cosmos3 \
        torchrun --nproc_per_node=1 cosmos3_joint_video_hand_pose/scripts/check_ckpt_load_cf5d68c.py \
        --ckpt <iter_dir>/model --output <result.json> [--forward]
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from torch.distributed.checkpoint import FileSystemReader

from cosmos_framework.checkpoint.dcp import ModelWrapper
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.utils import distributed, misc
from cosmos_framework.utils.generator.model_loader import _install_pathlib_pickle_compat, _load_model
from cosmos_framework.utils.lazy_config import instantiate

from cosmos3_joint_video_hand_pose.src import config as _config  # noqa: F401
from cosmos3_joint_video_hand_pose.src.smoke_train import DEFAULT_TOML


def _prefix(key: str, depth: int) -> str:
    return ".".join(key.split(".")[:depth])


def _summarize(keys: list[str], depth: int = 3, head: int = 20) -> dict:
    return {
        "count": len(keys),
        "by_prefix": dict(Counter(_prefix(k, depth) for k in keys).most_common()),
        "first": sorted(keys)[:head],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="DCP model directory (contains .metadata)")
    parser.add_argument(
        "--toml", type=Path, default=DEFAULT_TOML, help="experiment TOML used to build the model (default: %(default)s)"
    )
    parser.add_argument("--output", type=Path, required=True, help="path of the JSON result file to write")
    parser.add_argument("--forward", action="store_true", help="run one no-grad training_step on a real batch")
    parser.add_argument("--max-tokens", type=int, default=None, help="packing cap for the forward batch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed.init()
    started = time.time()
    overrides = [
        "job.wandb_mode=disabled",
        "model.config.parallelism.data_parallel_shard_degree=1",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "model.config.parallelism.context_parallel_shard_degree=1",
        "model.config.activation_checkpointing.mode=none",
        "dataloader_train.dataloader.num_workers=0",
        "dataloader_train.dataloader.persistent_workers=False",
        "dataloader_train.dataloader.prefetch_factor=null",
    ]
    if args.max_tokens is not None:
        overrides += [
            f"model.config.max_num_tokens_after_packing={args.max_tokens}",
            f"dataloader_train.max_sequence_length={args.max_tokens}",
        ]
    config = load_experiment_from_toml(args.toml, overrides)
    config.validate()
    config.freeze()
    ckpt_cfg = config.checkpoint
    keys_to_skip = list(ckpt_cfg.keys_to_skip_loading or [])

    model = instantiate(config.model).cuda()
    model.on_train_start()
    built_at = time.time()

    state_dict = ModelWrapper(model).state_dict()
    model_shapes = {k: tuple(v.shape) for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    _install_pathlib_pickle_compat()
    metadata = FileSystemReader(args.ckpt).read_metadata()
    ckpt_shapes = {k: tuple(v.size) for k, v in metadata.state_dict_metadata.items() if hasattr(v, "size")}
    skipped = [k for k in model_shapes if any(s in k for s in keys_to_skip)]
    wanted = {k: s for k, s in model_shapes.items() if k not in skipped}
    missing = [k for k in wanted if k not in ckpt_shapes]
    unexpected = [k for k in ckpt_shapes if k not in model_shapes]
    unexpected_skipped = [k for k in unexpected if any(s in k for s in keys_to_skip)]
    mismatched = {
        k: {"model": wanted[k], "ckpt": ckpt_shapes[k]}
        for k in wanted
        if k in ckpt_shapes and wanted[k] != ckpt_shapes[k]
    }

    result: dict = {
        "ckpt": args.ckpt,
        "experiment": config.job.name,
        "model_class": type(model).__name__,
        "keys_to_skip_loading": keys_to_skip,
        "load_ema_to_reg": bool(ckpt_cfg.load_ema_to_reg),
        "strict_resume": bool(ckpt_cfg.strict_resume),
        "model_tensor_keys": len(model_shapes),
        "ckpt_tensor_keys": len(ckpt_shapes),
        "model_keys_skipped_by_config": _summarize(skipped),
        "missing_keys": _summarize(missing),
        "unexpected_keys": _summarize(unexpected),
        "unexpected_keys_matching_skip_patterns": len(unexpected_skipped),
        "shape_mismatch_count": len(mismatched),
        "shape_mismatches": dict(list(mismatched.items())[:20]),
    }

    loaded = False
    if not missing and not mismatched:
        probe_name, probe = next((n, p) for n, p in model.net.named_parameters() if "llm2action" in n)
        before = probe.detach().float().norm().item()
        _load_model(model, checkpoint_path=args.ckpt, credential_path=None, keys_to_skip_loading=keys_to_skip)
        after = probe.detach().float().norm().item()
        result["load"] = {"status": "ok", "probe": probe_name, "probe_norm_before": before, "probe_norm_after": after}
        loaded = True
    else:
        result["load"] = {"status": "skipped", "reason": "missing keys or shape mismatches under strict loading"}
    loaded_at = time.time()

    if args.forward and loaded:
        dataloader = instantiate(config.dataloader_train)
        batch = next(iter(dataloader))
        indices = [int(x.reshape(-1)[0]) if torch.is_tensor(x) else int(x) for x in batch["dataset_index"]]
        batch = misc.to(batch, device="cuda")
        model.train()
        # Only a value written by this forward pass may mark the EgoVerse loss as used.
        model._last_visibility_loss_metrics = None
        with torch.no_grad():
            output, loss = model.training_step(batch, 0)
        visibility = getattr(model, "_last_visibility_loss_metrics", None)
        result["forward"] = {
            "samples": len(indices),
            "dataset_index": indices,
            "loss": float(loss.float().cpu()),
            "flow_matching_loss_vision": float(output["flow_matching_loss_vision"].float().cpu()),
            "flow_matching_loss_action": float(output["flow_matching_loss_action"].float().cpu()),
            "egoverse_loss_action_raw": float(output["egoverse_loss_action_raw"].float().cpu()),
            "egoverse_action_override_used": visibility is not None,
            "visibility_subblock_losses": {
                k: float(v.float().cpu()) for k, v in (visibility or {}).items() if k.endswith("_loss")
            },
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    result["timing_s"] = {"build": built_at - started, "load": loaded_at - built_at, "total": time.time() - started}

    if distributed.is_rank0():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("CKPT_LOAD_RESULT=" + json.dumps(result), flush=True)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
