#!/usr/bin/env python3
"""Standalone full-Nano bf16 cache acceptance. Never edits production source.

Launch with one torchrun process, LD_LIBRARY_PATH='', PYTHONPATH=.:packages/cosmos3.
Default checkpoint is the verified joint_chunk_cond_v1 iteration-2 checkpoint.
A load-only check is a preflight, never numerical acceptance.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = (
    ROOT
    / "outputs/joint_video_hand_pose/ar_v0_2/resume_2plus1_20260927T180331Z/training/joint_video_hand_pose/ar_v0_2/resume_contract_2plus1/checkpoints/iter_000000002/model"
)


def fingerprint():
    paths = list((ROOT / "cosmos3_joint_video_hand_pose/src").glob("*.py"))
    paths += [
        ROOT / p
        for p in (
            "packages/cosmos3/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py",
            "packages/cosmos3/cosmos_framework/model/generator/mot/causal_attention.py",
        )
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_model(args):
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils import distributed
    from cosmos3_joint_video_hand_pose.src import config as registered_config
    from cosmos3_joint_video_hand_pose.src.ar_v02_eval import _load_bound_model

    distributed.init()
    config = load_experiment_from_toml(
        args.toml,
        [
            "job.wandb_mode=disabled",
            "model.config.parallelism.data_parallel_shard_degree=1",
            "model.config.parallelism.data_parallel_replicate_degree=1",
            "model.config.parallelism.context_parallel_shard_degree=1",
            "model.config.parallelism.enable_inference_mode=true",
            "model.config.activation_checkpointing.mode=none",
        ],
    )
    config.validate()
    config.freeze()
    ds_cfg = config.dataloader_train.dataloader.datasets.egoverse.dataset
    model = _load_bound_model(config, args.ckpt, ds_cfg)
    if model.tensor_kwargs["dtype"] != torch.bfloat16:
        raise AssertionError("full Nano acceptance requires bf16")
    return model, ds_cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--toml", type=Path, default=ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_2.toml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--load-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="C=4, one full plus one partial chunk; not acceptance")
    p.add_argument(
        "--small-spatial",
        action="store_true",
        help="Crop real latents to one spatial patch for full-Nano boundary diagnostics; not full acceptance",
    )
    p.add_argument(
        "--focus-boundaries",
        action="store_true",
        help="Prefill 15 observed GT history chunks, then sample chunks 16/17 and the tail, always 30 steps",
    )
    p.add_argument("--tail", type=int, default=None, help="Select one valid tail for a targeted diagnostic")
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--full-chunks", type=int, default=17)
    p.add_argument("--history", choices=["gt", "pred_history", "generated"], default="gt")
    p.add_argument("--schedule", choices=["shift5", "independent"], default="shift5")
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        checkpoint=str(args.ckpt),
        source_before=fingerprint(),
        torch_version=torch.__version__,
        started=time.time(),
        full_acceptance=False,
    )
    try:
        assert (args.ckpt / ".metadata").is_file(), "DCP metadata missing"
        torch.set_num_threads(1)
        model, ds_cfg = load_model(args)
        # Freeze the full-query reference, independent of compact training changes.
        model.compact_noisy_training = False
        report.update(
            model_class=type(model).__name__,
            layers=model.net.num_hidden_layers,
            dtype=str(model.tensor_kwargs["dtype"]),
            parameter_count=sum(p.numel() for p in model.net.parameters()),
        )
        if args.load_only:
            report["status"] = "preflight_passed"
        else:
            from ar_v02_nano_validation import prepare_real_sampler, extend_latents, run_case, reuse_reference_masks

            reference_masks = reuse_reference_masks()
            report["reference_mask_reuse"] = True
            if args.smoke:
                args.chunk_sizes, args.full_chunks = [4], 1
            if (args.full_chunks < 17 and not args.smoke) or any(c not in (1, 2, 3, 4) for c in args.chunk_sizes):
                raise ValueError("acceptance requires C=1..4 and at least 17 full chunks")
            report["cases"] = []
            for c in args.chunk_sizes:
                base, item = prepare_real_sampler(model, ds_cfg, c, ROOT)
                if args.small_spatial:
                    patch = model.config.diffusion_expert_config.patch_spatial
                    base.gt_video = base.gt_video[:, :, :, :patch, :patch].contiguous()
                    base.layout = dataclasses.replace(base.layout, vision_tokens=1)
                    base.gen = dataclasses.replace(base.gen, x0_tokens_vision=[base.gt_video])
                tails = (
                    [args.tail] if args.tail is not None else ([1] if args.smoke else ([0] if c == 1 else range(1, c)))
                )
                if any(t != 0 if c == 1 else not 1 <= t < c for t in tails):
                    raise ValueError("invalid tail for C")
                for tail in tails:
                    sampler = extend_latents(base, args.full_chunks, tail)
                    spec = dict(
                        c=c,
                        tail=tail,
                        history=args.history,
                        schedule=args.schedule,
                        observed_history_chunks=15 if args.focus_boundaries else 0,
                    )
                    if args.focus_boundaries and (args.history != "gt" or args.smoke):
                        raise ValueError("boundary focus requires GT observed history and >=17 chunks")
                    result = run_case(sampler, spec, args.output / f"c{c}_tail{tail}")
                    result["fixture"] = item
                    report["cases"].append(result)
            report["status"] = "smoke_passed" if args.smoke else "passed"
            report["small_spatial"] = args.small_spatial
            report["tested_history"], report["tested_schedule"] = args.history, args.schedule
            # One CLI invocation tests just one history/schedule pair. Full design
            # acceptance also requires the other histories/schedule at full resolution.
            report["full_acceptance"] = False
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report.update(source_after=fingerprint(), finished=time.time())
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
