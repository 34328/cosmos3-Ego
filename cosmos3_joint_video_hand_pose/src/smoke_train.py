#!/usr/bin/env python3
"""Run real packed EgoVerse training steps on the configured distributed path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.utils import distributed, misc
from cosmos_framework.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_framework.utils.lazy_config import instantiate

from . import config as _config  # noqa: F401
from .audit_dataloader import audit_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOML = PROJECT_ROOT / "configs/overfit_v0_0.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--job-name", default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("multi-episode smoke test requires --steps >= 2")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise ValueError("the distributed smoke test requires torchrun with at least two ranks")
    with distributed_init():
        distributed.init()

    overrides = [
        "trainer.max_iter=1",
        "trainer.logging_iter=1",
        "checkpoint.save_iter=1000000",
        f"job.wandb_mode={args.wandb_mode}",
        f"job.name={args.job_name}",
    ]
    if args.max_tokens is not None:
        overrides.extend(
            [
                f"model.config.max_num_tokens_after_packing={args.max_tokens}",
                f"dataloader_train.max_sequence_length={args.max_tokens}",
            ]
        )
    config = load_experiment_from_toml(args.toml, overrides)
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    with model_init():
        model = instantiate(config.model)
    with data_loader_init():
        dataloader = instantiate(config.dataloader_train)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    model.train()
    trainer.callbacks.on_optimizer_init_start()
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    trainer.callbacks.on_optimizer_init_end()
    iteration = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    trainer.callbacks.on_train_start(model, iteration=iteration)
    dist.barrier()

    iterator = iter(dataloader)
    grad_accum_iter = 0
    records = []
    configured_cap = config.dataloader_train.max_sequence_length
    cap = int(configured_cap or config.model.config.max_num_tokens_after_packing)
    for step in range(args.steps):
        current_iteration = iteration + step + 1
        trainer.callbacks.on_before_dataloading(current_iteration)
        cpu_batch, stop = trainer._fetch_data_batch(model, iterator)
        trainer.callbacks.on_after_dataloading(current_iteration)
        if stop:
            raise RuntimeError(f"dataloader stopped before smoke step {step}")
        batch_audit = audit_batch(cpu_batch, cap)
        batch = misc.to(cpu_batch, device="cuda")
        trainer._cp_data_window.store_device_batch(batch)
        trainer.callbacks.on_training_step_start(model, batch, iteration=current_iteration)
        trainer.callbacks.on_training_step_batch_start(model, batch, iteration=current_iteration)
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        started = time.perf_counter()
        output, loss, grad_accum_iter = trainer.training_step(
            model,
            optimizer,
            scheduler,
            grad_scaler,
            batch,
            iteration=current_iteration,
            grad_accum_iter=grad_accum_iter,
        )
        dist.barrier()
        trainer.callbacks.on_training_step_batch_end(model, batch, output, loss, iteration=current_iteration)
        trainer.callbacks.on_training_step_end(model, batch, output, loss, iteration=current_iteration + 1)
        record = {
            "step": step,
            **batch_audit,
            "loss": float(loss.detach().cpu()),
            "video_loss": float(output["flow_matching_loss_vision"].detach().cpu()),
            "action_loss": float(output["flow_matching_loss_action"].detach().cpu()),
            "video_loss_weighted": float(output["egoverse_loss_video_weighted"].detach().cpu()),
            "action_loss_weighted": float(output["egoverse_loss_action_weighted"].detach().cpu()),
            "total_loss_metric": float(output["egoverse_loss_total"].detach().cpu()),
            "finite": bool(torch.isfinite(loss).item()),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "elapsed_seconds": time.perf_counter() - started,
        }
        subblock_sources = (
            "egoverse_loss_action_camera_translation_raw",
            "egoverse_loss_action_camera_rotation_raw",
            "egoverse_loss_action_right_wrist_translation_raw",
            "egoverse_loss_action_right_wrist_rotation_raw",
            "egoverse_loss_action_right_hand_latent_raw",
            "egoverse_loss_action_left_wrist_translation_raw",
            "egoverse_loss_action_left_wrist_rotation_raw",
            "egoverse_loss_action_left_hand_latent_raw",
        )
        for source in subblock_sources:
            if source not in output:
                raise KeyError(f"overfit_v0.0 smoke output missing action sub-block metric: {source}")
            value = output[source].detach().float()
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite action sub-block metric at smoke step {step}: {source}")
            record[source.removeprefix("egoverse_")] = float(value.cpu())
        if not record["finite"]:
            raise FloatingPointError(f"non-finite loss at smoke step {step}")
        records.append(record)

    local = {"rank": dist.get_rank(), "steps": records}
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if dist.get_rank() == 0:
        result = {"status": "success", "world_size": dist.get_world_size(), "steps": args.steps, "ranks": gathered}
        if args.wandb_mode != "disabled":
            import wandb

            if wandb.run is not None:
                result["wandb_run_id"] = wandb.run.id
                result["wandb_run_url"] = wandb.run.url
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("EGOVERSE_SMOKE_RESULT=" + json.dumps(result), flush=True)
    trainer.callbacks.on_train_end(model, iteration=iteration + args.steps)
    trainer.checkpointer.finalize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
