#!/usr/bin/env python3
"""Run real packed EgoVerse training steps on the configured distributed path."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np

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
DEFAULT_TOML = PROJECT_ROOT / "configs/overfit_v0_5_frame_delta_b3.toml"


def input_digest(value) -> str:
    """Typed SHA256 of values, without RNG use, graph retention or tensor writes."""
    digest = hashlib.sha256()

    def feed(tag, data=b""):
        digest.update(tag.encode() + b":" + str(len(data)).encode() + b":" + data)

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            feed("tensor", str((str(tensor.dtype), tuple(tensor.shape))).encode())
            # uint8 view supports bf16, empty tensors and scalar tensors.
            feed("bytes", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif is_dataclass(item) and not isinstance(item, type):
            feed("dataclass", type(item).__qualname__.encode())
            visit({field.name: getattr(item, field.name) for field in fields(item)})
        elif isinstance(item, dict):
            feed("dict", str(len(item)).encode())
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            feed(type(item).__name__, str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            feed(type(item).__name__, json.dumps(item, ensure_ascii=False).encode())
        else:
            raise TypeError(f"unsupported input audit type: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def batch_input_audit(batch) -> dict:
    """Per-sample window and post-CFG text evidence; never log media/text payloads."""

    def flatten(value):
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        return torch.as_tensor(value).detach().cpu().reshape(-1)

    samples = []
    for index in range(len(batch["video"])):
        sample = {}
        if "window_start" in batch:
            sample["window_start"] = int(flatten(batch["window_start"][index]).item())
        for name in (
            "source_frame_indices",
            "action_source_frame_indices",
            "future_action_source_frame_indices",
            "ar_boundary_source_indices",
        ):
            if name in batch:
                indexes = flatten(batch[name][index])
                sample[name] = {
                    "count": indexes.numel(),
                    "first": int(indexes[0]) if indexes.numel() else None,
                    "last": int(indexes[-1]) if indexes.numel() else None,
                    "sha256": input_digest(indexes),
                }
        sample["cfg_text_sha256"] = input_digest(batch["text_token_ids"][index])
        sample["cfg_text_num_tokens"] = flatten(batch["text_token_ids"][index]).numel()
        # Includes state/source geometry used to derive chunk-camera actions.
        payload = {
            name: value[index]
            for name, value in batch.items()
            if name
            in (
                "video",
                "action",
                "action_raw",
                "hand_visibility",
                "text_token_ids",
                "raw_action_dim",
                "action_valid_mask",
                "window_start",
                "source_frame_indices",
                "action_source_frame_indices",
                "future_action_source_frame_indices",
            )
            or name.startswith("ar_")
        }
        sample["raw_input_sha256"] = input_digest(payload)
        samples.append(sample)
    return {"schema": "smoke_input_v1", "samples": samples, "raw_batch_sha256": input_digest(samples)}


def install_denoise_input_audit(model, emit):
    """Capture actual clean/noisy model inputs after noise/precision conversion.

    CPU hashing synchronizes device copies, so audited timings are not benchmarks.
    Hashes establish input equality only; they do not fingerprint model parameters.
    """
    original = model.denoise

    def audited(net=None, data_batch_packed=None, memory=None, video_temporal_causal=None):
        step = getattr(model, "_ar_step", None)
        context = {
            "pass_number": getattr(memory, "pass_number", None),
            "chunk_size": getattr(step, "chunk_size", None),
            "window": getattr(step, "window", None),
            "video_temporal_causal": video_temporal_causal,
        }
        components = {
            field.name: input_digest(getattr(data_batch_packed, field.name)) for field in fields(data_batch_packed)
        }
        record = {
            **context,
            "field_sha256": components,
            "training_input_sha256": input_digest({"context": context, "fields": components}),
        }
        # Useful separate evidence for sigma versus sampled noise differences.
        for name in ("vision", "action"):
            modality = getattr(data_batch_packed, name, None)
            if modality is not None:
                record[name + "_tokens_sha256"] = input_digest(modality.tokens)
                record[name + "_timesteps_sha256"] = input_digest(modality.timesteps)
        emit(record)
        return original(
            net=net, data_batch_packed=data_batch_packed, memory=memory, video_temporal_causal=video_temporal_causal
        )

    model.denoise = audited


def capture_rng_state():
    """Rank-local Python/NumPy/CPU/current CUDA RNG, without random draws."""
    state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=(state[0], state[1].tolist(), *state[2:]),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    )


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def install_fixed_vae_inputs(model, directory, *, replay, context):
    """Freeze actual post-VAE inputs for denoiser gradient reproducibility.

    Still execute encoding on replay to keep chunk metadata and report numerical
    differences. Equal raw RGB/RNG is not proof of bitwise-equal cuDNN VAE output.
    This is test-only and never changes the ordinary trainer path.
    """
    original = model._encode_vision_x0_tokens

    def encode(*args, **kwargs):
        actual = original(*args, **kwargs)
        path = directory / f"rank{dist.get_rank():05d}.step{context['microstep']:05d}.vae.pt"
        if replay:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            reference = payload["latents"]
            if len(reference) != len(actual) or any(a.shape != b.shape for a, b in zip(actual, reference)):
                raise ValueError("fixed VAE geometry differs from replay")
            result = [b.to(device=a.device, dtype=a.dtype) for a, b in zip(actual, reference)]
            diagnostics = [
                dict(max_abs=float((a.float() - b.float()).abs().max()), equal=bool(torch.equal(a, b)))
                for a, b in zip(actual, result)
            ]
            restore_rng_state(payload["rng_after"])
            path.with_suffix(".replay.json").write_text(json.dumps(diagnostics) + "\n")
            return result
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            torch.save(dict(latents=[a.detach().cpu() for a in actual], rng_after=capture_rng_state()), handle)
        return actual

    model._encode_vision_x0_tokens = encode


def save_fixed_input(path, batch, rng, *, rank, world_size, iteration):
    """Trusted local torch artifact. Exclusive creation preserves earlier evidence."""
    payload = dict(
        schema="smoke_fixed_input_v1",
        rank=rank,
        world_size=world_size,
        iteration=iteration,
        batch=batch,
        rng=rng,
        batch_audit=batch_input_audit(batch),
        rng_sha256=input_digest(rng),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return payload


def load_fixed_input(path, *, rank, world_size, iteration):
    # Only use trusted artifacts captured in this user's own validation directory.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in dict(
        schema="smoke_fixed_input_v1", rank=rank, world_size=world_size, iteration=iteration
    ).items():
        if payload.get(key) != expected:
            raise ValueError(f"fixed-input {key} mismatch")
    if batch_input_audit(payload["batch"]) != payload["batch_audit"]:
        raise ValueError("fixed-input batch digest mismatch")
    if input_digest(payload["rng"]) != payload["rng_sha256"]:
        raise ValueError("fixed-input RNG digest mismatch")
    return payload


def parameter_fingerprint(net):
    return input_digest(
        {name: input_digest(p.to_local() if hasattr(p, "to_local") else p) for name, p in net.named_parameters()}
    )


@torch.no_grad()
def gradient_audit(net, *, scale=1.0):
    """Pre-clipping local-shard digests and independent global FP64 L2 norms.

    Sum FSDP shards but count replicas once. Reject partial placements.
    CPU reductions intentionally trade speed for an independent numerical check.
    """
    world = dist.get_world_size() if dist.is_initialized() else 1
    entries, contributions = {}, []
    parameters = list(net.named_parameters())
    for name, parameter in parameters:
        grad = parameter.grad
        if grad is None:
            entries[name] = {"present": False}
            contributions.append(0.0)
            continue
        shard_degree = 1
        placements = getattr(grad, "placements", ())
        for dimension, placement in enumerate(placements):
            if placement.is_shard():
                shard_degree *= grad.device_mesh.size(dimension)
            elif not placement.is_replicate():
                raise ValueError(f"unsupported gradient placement: {placement}")
        local = grad.to_local() if hasattr(grad, "to_local") else grad
        cpu = local.detach().contiguous().cpu()
        squared = float(cpu.double().square().sum()) / scale**2
        entries[name] = dict(
            present=True,
            dtype=str(cpu.dtype),
            shape=list(cpu.shape),
            placements=[str(p) for p in placements],
            sha256=input_digest(cpu),
            local_l2=squared**0.5,
        )
        contributions.append(squared * shard_degree / world)
    sums = torch.tensor(contributions, dtype=torch.float64, device=parameters[0][1].device)
    if dist.is_initialized():
        dist.all_reduce(sums)
    sums = sums.cpu().tolist()
    layers = {}
    for (name, entry), squared in zip(entries.items(), sums, strict=True):
        entry["global_l2"] = squared**0.5
        match = re.match(r"(.*?(?:layers|blocks)\.\d+)(?:\.|$)", name)
        layer = match.group(1) if match else name.rsplit(".", 1)[0]
        layers[layer] = layers.get(layer, 0.0) + squared
    return dict(
        schema="smoke_gradient_v1",
        amp_scale=scale,
        global_l2=sum(sums) ** 0.5,
        layers={name: value**0.5 for name, value in layers.items()},
        parameters=entries,
        local_gradient_sha256=input_digest(entries),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML)
    parser.add_argument("--steps", type=int, default=3, help="Completed optimizer updates, not microbatches")
    parser.add_argument("--save-final", action="store_true", help="Save and finalize a resumable checkpoint")
    parser.add_argument(
        "--expect-resume-iteration",
        type=int,
        default=None,
        help="Fail unless a separate restart restores this optimizer iteration",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--job-name", default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    fixed = parser.add_mutually_exclusive_group()
    fixed.add_argument("--capture-fixed-inputs", type=Path)
    fixed.add_argument("--replay-fixed-inputs", type=Path)
    parser.add_argument("--audit-gradients", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or (args.steps < 2 and args.expect_resume_iteration is None):
        raise ValueError("use --steps >= 2, or --steps 1 with --expect-resume-iteration for restart validation")
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
    if getattr(model, "whole_action_loss", False):
        model.configure_ar_loss_accumulation(config.trainer.grad_accum_iter)
        if config.trainer.grad_accum_iter != 1:
            raise ValueError(
                "v0.2 smoke accumulation needs preplanned per-modality window counts; "
                "use the contract helper in the window-aware trainer"
            )
    # Check scaled raw gradients BEFORE norm/clipping callbacks can sanitize them.
    from .ar_v02_contract import assert_finite_gradients

    original_after_backward = model.on_after_backward
    gradient_checks = []
    gradient_records = []

    def checked_after_backward():
        original_after_backward()
        gradient_checks.append(assert_finite_gradients(model.net.parameters()))
        if args.audit_gradients:
            gradient_records.append(gradient_audit(model.net, scale=grad_scaler.get_scale()))

    model.on_after_backward = checked_after_backward
    trainer.callbacks.on_optimizer_init_start()
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    trainer.callbacks.on_optimizer_init_end()
    # Match the official trainer: bind recoverable loader before DCP restores it.
    for callback in trainer.callbacks._callbacks:
        bind = getattr(callback, "bind_dataloader", None)
        if bind is not None:
            bind(dataloader)
    iteration = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    if args.expect_resume_iteration is not None and iteration != args.expect_resume_iteration:
        raise RuntimeError(f"expected resume iteration {args.expect_resume_iteration}, got {iteration}")
    fetch_count = trainer._resume_dataloader_fetch_count(model, iteration)
    if hasattr(dataloader, "set_start_iteration"):
        dataloader.set_start_iteration(fetch_count)
    trainer.callbacks.on_train_start(model, iteration=iteration)
    dist.barrier()
    initial_parameters_sha256 = parameter_fingerprint(model.net) if args.audit_gradients else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output.with_name(args.output.stem + f".rank{dist.get_rank():05d}.inputs.jsonl")
    # Separate rank files survive failures before the final collective/result JSON.
    audit_file = audit_path.open("x", encoding="utf-8")
    input_passes = []
    audit_context = {}
    if args.capture_fixed_inputs or args.replay_fixed_inputs:
        install_fixed_vae_inputs(
            model,
            args.capture_fixed_inputs or args.replay_fixed_inputs,
            replay=bool(args.replay_fixed_inputs),
            context=audit_context,
        )

    def emit_input(record):
        input_passes.append(record)
        audit_file.write(json.dumps({**audit_context, "event": "denoise", **record}) + "\n")
        audit_file.flush()

    install_denoise_input_audit(model, emit_input)
    iterator = iter(dataloader)
    grad_accum_iter = 0
    records = []
    configured_cap = config.dataloader_train.max_sequence_length
    # One-sample-per-step AR batches set max_sequence_length=None; audit against the model cap.
    cap = int(configured_cap or config.model.config.max_num_tokens_after_packing)
    completed_updates = 0
    step = 0
    while completed_updates < args.steps:
        # Match the official trainer: iteration is the completed-update count.
        current_iteration = iteration + completed_updates
        trainer.callbacks.on_before_dataloading(current_iteration)
        fixed_path = args.capture_fixed_inputs or args.replay_fixed_inputs
        if fixed_path is not None:
            fixed_path = fixed_path / f"rank{dist.get_rank():05d}.step{step:05d}.pt"
        fixed_payload = None
        if args.replay_fixed_inputs:
            fixed_payload = load_fixed_input(
                fixed_path, rank=dist.get_rank(), world_size=dist.get_world_size(), iteration=current_iteration
            )
            cpu_batch, stop = fixed_payload["batch"], False
        else:
            cpu_batch, stop = trainer._fetch_data_batch(model, iterator)
        trainer.callbacks.on_after_dataloading(current_iteration)
        if stop:
            raise RuntimeError(f"dataloader stopped before smoke step {step}")
        batch_audit = audit_batch(cpu_batch, cap, config.model.config.get("action_tokens_per_latent"))
        raw_input_audit = batch_input_audit(cpu_batch)
        input_passes.clear()
        audit_context.update(rank=dist.get_rank(), iteration=current_iteration, microstep=step)
        audit_file.write(json.dumps({**audit_context, "event": "batch", **raw_input_audit}) + "\n")
        audit_file.flush()
        batch = misc.to(cpu_batch, device="cuda")
        trainer._cp_data_window.store_device_batch(batch)
        trainer.callbacks.on_training_step_start(model, batch, iteration=current_iteration)
        trainer.callbacks.on_training_step_batch_start(model, batch, iteration=current_iteration)
        if fixed_payload is not None:
            restore_rng_state(fixed_payload["rng"])
        rng = capture_rng_state()
        if args.capture_fixed_inputs:
            save_fixed_input(
                fixed_path,
                cpu_batch,
                rng,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                iteration=current_iteration,
            )
        rng_sha256 = input_digest(rng)
        audit_file.write(json.dumps({**audit_context, "event": "rng", "rng_sha256": rng_sha256}) + "\n")
        audit_file.flush()
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
        gradient_path = None
        if args.audit_gradients:
            gradient_path = args.output.with_name(
                args.output.stem + f".rank{dist.get_rank():05d}.step{step:05d}.gradients.json"
            )
            with gradient_path.open("x") as handle:
                json.dump(gradient_records[-1], handle)
                handle.write("\n")
        trainer.callbacks.on_training_step_batch_end(model, batch, output, loss, iteration=current_iteration)
        if grad_accum_iter == 0:
            completed_updates += 1
            trainer.callbacks.on_training_step_end(model, batch, output, loss, iteration=current_iteration + 1)
        record = {
            "step": step,
            "optimizer_updates_completed": completed_updates,
            "grad_accum_position": grad_accum_iter,
            "raw_gradients_finite": True,
            "gradient_tensors_checked_global": gradient_checks[-1],
            "rng_sha256": rng_sha256,
            "gradient_audit_path": str(gradient_path) if gradient_path else None,
            "raw_gradient_global_l2": gradient_records[-1]["global_l2"] if args.audit_gradients else None,
            **batch_audit,
            "input_audit": raw_input_audit,
            "denoise_input_audit": list(input_passes),
            "loss": float(loss.detach().cpu()),
            "video_loss": float(output["flow_matching_loss_vision"].detach().cpu()),
            "action_loss": float(output["flow_matching_loss_action"].detach().cpu()),
            "video_loss_weighted": float(output["egoverse_loss_video_weighted"].detach().cpu()),
            "action_loss_weighted": float(output["egoverse_loss_action_weighted"].detach().cpu()),
            "total_loss_metric": float(output["egoverse_loss_total"].detach().cpu()),
            "finite": bool(torch.isfinite(loss).item()),
            **{
                name.removeprefix("egoverse_"): float(output[name].detach().float().cpu())
                for name in ("egoverse_ar_chunk_size", "egoverse_ar_window", "egoverse_sigma_action_mean")
                if name in output
            },
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
            if getattr(model, "whole_action_loss", False):
                break
            if source not in output:
                raise KeyError(f"overfit_v0.0 smoke output missing action sub-block metric: {source}")
            value = output[source].detach().float()
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite action sub-block metric at smoke step {step}: {source}")
            record[source.removeprefix("egoverse_")] = float(value.cpu())
        if not record["finite"]:
            raise FloatingPointError(f"non-finite loss at smoke step {step}")
        records.append(record)
        step += 1
        del output, loss, batch

    if grad_accum_iter != 0:
        raise RuntimeError("smoke ended inside a gradient accumulation window")
    if args.save_final:
        trainer.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration + completed_updates)
        trainer.checkpointer.finalize()
    audit_file.close()
    local = {
        "rank": dist.get_rank(),
        "steps": records,
        "input_audit_path": str(audit_path),
        "initial_parameters_sha256": initial_parameters_sha256,
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if dist.get_rank() == 0:
        result = {
            "status": "success",
            "world_size": dist.get_world_size(),
            "steps": args.steps,
            "optimizer_updates": completed_updates,
            "microsteps": step,
            "loaded_iteration": iteration,
            "final_iteration": iteration + completed_updates,
            "checkpoint_saved": args.save_final,
            "ranks": gathered,
        }
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
    trainer.callbacks.on_app_end()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
