"""Minimal project-specific W&B logging without changing the Cosmos core."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb

from cosmos_framework.callbacks.grad_clip import _group_params_by_mesh, _total_norm_by_mesh
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback, WandBCallback


BASE_LOSS_METRIC_SOURCES = {
    "loss/video_raw": "egoverse_loss_video_raw",
    "loss/action_raw": "egoverse_loss_action_raw",
    "loss/total": "egoverse_loss_total",
}

SUBBLOCK_LOSS_METRIC_SOURCES = {
    "loss/action_camera_translation_raw": "egoverse_loss_action_camera_translation_raw",
    "loss/action_camera_rotation_raw": "egoverse_loss_action_camera_rotation_raw",
    "loss/action_right_wrist_translation_raw": "egoverse_loss_action_right_wrist_translation_raw",
    "loss/action_right_wrist_rotation_raw": "egoverse_loss_action_right_wrist_rotation_raw",
    "loss/action_right_hand_latent_raw": "egoverse_loss_action_right_hand_latent_raw",
    "loss/action_left_wrist_translation_raw": "egoverse_loss_action_left_wrist_translation_raw",
    "loss/action_left_wrist_rotation_raw": "egoverse_loss_action_left_wrist_rotation_raw",
    "loss/action_left_hand_latent_raw": "egoverse_loss_action_left_hand_latent_raw",
}

LOSS_METRIC_SOURCES = BASE_LOSS_METRIC_SOURCES | SUBBLOCK_LOSS_METRIC_SOURCES


def _pre_clip_grad_norm(parameters: list[torch.Tensor]) -> torch.Tensor:
    """Global L2 grad norm over ``parameters`` without modifying gradients.

    Uses the same mesh grouping and DTensor reduction as the native GradClip
    callback; every parameter must already have a gradient.
    """
    total_norm, _ = _total_norm_by_mesh(_group_params_by_mesh(parameters))
    return total_norm

SIGMA_METRIC_SOURCES = {
    "sigma/action_mean": "egoverse_sigma_action_mean",
    "sigma/video_mean": "egoverse_sigma_video_mean",
    "sigma/video_min": "egoverse_sigma_video_min",
    "sigma/video_max": "egoverse_sigma_video_max",
    "sigma/video_low_0_1_fraction": "egoverse_sigma_video_low_0_1_fraction",
    "sigma/video_low_0_2_fraction": "egoverse_sigma_video_low_0_2_fraction",
}

GRAD_NORM_METRIC_NAMES = {
    "grad_norm/shared_pre_clip",
    "grad_norm/video_projection_pre_clip",
    "grad_norm/action_projection_pre_clip",
    "grad_norm/all_selected_pre_clip",
}

GRAD_NONFINITE_METRIC_NAMES = {
    "grad_nonfinite/shared_present",
    "grad_nonfinite/video_projection_present",
    "grad_nonfinite/action_projection_present",
    "grad_nonfinite/all_selected_present",
    "grad_nonfinite_elements/nan_count",
    "grad_nonfinite_elements/posinf_count",
    "grad_nonfinite_elements/neginf_count",
    "grad_nonfinite_elements/nonfinite_count",
    "grad_nonfinite_elements/selected_numel",
    "grad_nonfinite_elements/nonfinite_fraction",
}

LR_METRIC_NAMES = {
    "optim/lr_base",
    "optim/lr_action",
}


def filter_wandb_metrics(metrics: dict) -> dict:
    """Keep project losses, optimizer diagnostics, and real grad-clip events."""
    allowed = (
        set(LOSS_METRIC_SOURCES)
        | set(SIGMA_METRIC_SOURCES)
        | LR_METRIC_NAMES
        | GRAD_NORM_METRIC_NAMES
        | GRAD_NONFINITE_METRIC_NAMES
    )
    return {
        key: value
        for key, value in metrics.items()
        if key in allowed or key.startswith("grad_clip/")
    }


class LossOnlyWandBCallback(WandBCallback):
    """Compatibility name for the unmodified official W&B lifecycle.

    New configurations should register WandBCallback directly and add
    EgoVerseLossWandbCallback before GradClip for project diagnostics.
    """


def extract_loss_metrics(output_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the exact loss series required for the joint training dashboard."""
    required = tuple(BASE_LOSS_METRIC_SOURCES.values())
    missing = [source for source in required if source not in output_batch]
    if missing:
        raise KeyError(f"EgoVerse W&B metrics missing model outputs: {missing}")
    sources = LOSS_METRIC_SOURCES | SIGMA_METRIC_SOURCES
    return {name: output_batch[source] for name, source in sources.items() if source in output_batch}


class EgoVerseLossWandbCallback(Callback):
    """Persist each optimizer step locally, independently of W&B availability.

    Register alongside the official WandBCallback, before GradClip. JSONL rows
    use the trainer's completed-step counter; resume always appends. Nonfinite
    values become null with explicit diagnostics, never successful zero losses.
    """

    def __init__(self, nonfinite_detail_iterations: int = 5) -> None:
        super().__init__()
        self._loss_sums: dict[str, torch.Tensor] = {}
        self._loss_counts: dict[str, int] = {}
        self._pending_diagnostics: dict[int, dict[str, float]] = {}
        self._nonfinite_detail_iterations = int(nonfinite_detail_iterations)

    @torch.no_grad()
    def on_before_optimizer_step(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        """Log disjoint pre-clip norms without modifying gradients or RNG."""
        del optimizer, grad_scaler
        if (iteration + 1) % self.config.trainer.logging_iter:
            return

        parameter_groups: dict[str, list[torch.Tensor]] = {
            "grad_norm/shared_pre_clip": [],
            "grad_norm/video_projection_pre_clip": [],
            "grad_norm/action_projection_pre_clip": [],
        }
        named_parameter_groups: dict[str, list[tuple[str, torch.Tensor]]] = {
            name: [] for name in parameter_groups
        }
        for name, parameter in model.net.named_parameters():
            if parameter.grad is None:
                continue
            if "moe_gen" in name or "time_embedder" in name:
                parameter_groups["grad_norm/shared_pre_clip"].append(parameter)
                named_parameter_groups["grad_norm/shared_pre_clip"].append((name, parameter))
            elif any(key in name for key in ("vae2llm", "llm2vae", "vision_condition_embed")):
                parameter_groups["grad_norm/video_projection_pre_clip"].append(parameter)
                named_parameter_groups["grad_norm/video_projection_pre_clip"].append((name, parameter))
            elif any(key in name for key in ("action2llm", "llm2action", "action_modality_embed", "action_state_embed")):
                parameter_groups["grad_norm/action_projection_pre_clip"].append(parameter)
                named_parameter_groups["grad_norm/action_projection_pre_clip"].append((name, parameter))

        # Every selected optimizer parameter belongs to exactly one group.
        # An ablation may intentionally freeze one whole group; report that
        # group as zero rather than turning diagnostics into a training error.
        # All ranks execute the same group order because
        # _pre_clip_grad_norm performs distributed DTensor reductions.  First detect
        # whether each raw group contains NaN/Inf.  Cosmos' configured
        # GradClip(force_finite=True) sanitizes them immediately after this
        # callback, so hiding this distinction would make a sanitized norm look
        # like a genuinely finite backward pass.
        nonfinite: dict[str, float] = {}
        nonfinite_names = {
            "grad_norm/shared_pre_clip": "grad_nonfinite/shared_present",
            "grad_norm/video_projection_pre_clip": "grad_nonfinite/video_projection_present",
            "grad_norm/action_projection_pre_clip": "grad_nonfinite/action_projection_present",
        }
        for metric_name, parameters in parameter_groups.items():
            if not parameters:
                nonfinite[nonfinite_names[metric_name]] = 0.0
                continue
            raw_norm = _pre_clip_grad_norm(parameters)
            nonfinite[nonfinite_names[metric_name]] = float(not torch.isfinite(raw_norm).item())
        nonfinite["grad_nonfinite/all_selected_present"] = max(nonfinite.values())

        # For the first few diagnostic iterations, distinguish a genuinely
        # non-finite gradient element from a non-finite value produced only by
        # distributed norm bookkeeping. This never sanitizes gradients.
        # Exact FQNs and local-shard counts are persisted per rank; W&B receives
        # aggregate counts and a ratio across the distributed selected shards.
        if iteration <= self._nonfinite_detail_iterations:
            rank = dist.get_rank() if dist.is_initialized() else 0
            detail_records: list[dict[str, object]] = []
            totals = torch.zeros(4, dtype=torch.float64, device=next(model.net.parameters()).device)
            for group_name, named_parameters in named_parameter_groups.items():
                for fqn, parameter in named_parameters:
                    grad = parameter.grad
                    if grad is None:
                        continue
                    local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
                    local_grad = local_grad.detach()
                    nan_count = int(torch.isnan(local_grad).sum().item())
                    posinf_count = int(torch.isposinf(local_grad).sum().item())
                    neginf_count = int(torch.isneginf(local_grad).sum().item())
                    numel = int(local_grad.numel())
                    totals += totals.new_tensor([nan_count, posinf_count, neginf_count, numel])
                    bad_count = nan_count + posinf_count + neginf_count
                    if bad_count:
                        detail_records.append(
                            {
                                "iteration": int(iteration),
                                "rank": rank,
                                "group": group_name,
                                "fqn": fqn,
                                "shape": list(local_grad.shape),
                                "dtype": str(local_grad.dtype),
                                "numel": numel,
                                "nan_count": nan_count,
                                "posinf_count": posinf_count,
                                "neginf_count": neginf_count,
                                "nonfinite_count": bad_count,
                                "nonfinite_fraction": bad_count / max(numel, 1),
                            }
                        )
            if detail_records:
                trace_dir = Path(self.config.job.path_local) / "grad_nonfinite_trace"
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / f"rank_{rank:05d}.jsonl"
                with trace_path.open("a", encoding="utf-8") as handle:
                    for record in detail_records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if dist.is_initialized():
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            nan_count, posinf_count, neginf_count, selected_numel = totals.tolist()
            nonfinite_count = nan_count + posinf_count + neginf_count
            nonfinite.update(
                {
                    "grad_nonfinite_elements/nan_count": nan_count,
                    "grad_nonfinite_elements/posinf_count": posinf_count,
                    "grad_nonfinite_elements/neginf_count": neginf_count,
                    "grad_nonfinite_elements/nonfinite_count": nonfinite_count,
                    "grad_nonfinite_elements/selected_numel": selected_numel,
                    "grad_nonfinite_elements/nonfinite_fraction": nonfinite_count / max(selected_numel, 1.0),
                }
            )

        # Measure raw norms; sanitization belongs exclusively to native GradClip.
        norms: dict[str, torch.Tensor] = {}
        reference_device = next(model.net.parameters()).device
        for metric_name, parameters in parameter_groups.items():
            if not parameters:
                norms[metric_name] = torch.zeros((), device=reference_device)
                continue
            norm = _pre_clip_grad_norm(parameters)
            norms[metric_name] = norm.detach().float()
        norms["grad_norm/all_selected_pre_clip"] = torch.linalg.vector_norm(torch.stack(list(norms.values())))

        metrics = {name: value.item() for name, value in norms.items()} | nonfinite
        learning_rates = [float(lr) for lr in scheduler.get_last_lr()]
        if learning_rates:
            metrics.update({"optim/lr_base": min(learning_rates), "optim/lr_action": max(learning_rates)})
        # Do not advance W&B's step before the trainer increments its counter.
        self._pending_diagnostics[iteration + 1] = metrics

    @torch.no_grad()
    def on_training_step_batch_end(
        self, model, data_batch, output_batch, loss, iteration: int = 0,
    ) -> None:
        """Include every microbatch when Cosmos uses gradient accumulation."""
        del model, data_batch, loss, iteration
        for name, value in extract_loss_metrics(output_batch).items():
            value = value.detach().float().mean()
            if name in self._loss_sums:
                self._loss_sums[name] = self._loss_sums[name] + value
            else:
                self._loss_sums[name] = value.clone()
            self._loss_counts[name] = self._loss_counts.get(name, 0) + 1

    @torch.no_grad()
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if not self._loss_sums:
            self.on_training_step_batch_end(model, data_batch, output_batch, loss, iteration)
        names = tuple(LOSS_METRIC_SOURCES | SIGMA_METRIC_SOURCES)
        reference = next(iter(self._loss_sums.values()))
        # Fixed order and shape even when optional diagnostics differ by rank.
        packed = torch.stack([
            torch.stack((self._loss_sums.get(name, reference.new_zeros(())),
                         reference.new_tensor(self._loss_counts.get(name, 0))))
            for name in names
        ])
        if dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        metrics = {
            name: total / count
            for name, (total, count) in zip(names, packed.cpu().tolist(), strict=True)
            if count
        }
        self._loss_sums.clear()
        self._loss_counts.clear()
        metrics.update(self._pending_diagnostics.pop(iteration, {}))
        if not distributed.is_rank0():
            return

        invalid = {name: str(value) for name, value in metrics.items() if not math.isfinite(value)}
        invalid_losses = [name for name in invalid if name in LOSS_METRIC_SOURCES]
        row = {"step": int(iteration), "status": "nonfinite_loss" if invalid_losses else "ok"}
        row.update({name: None if name in invalid else value for name, value in metrics.items()})
        if invalid:
            row["nonfinite_metrics"] = invalid
        path = Path(self.config.job.path_local) / "loss_metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Close and flush every row; no train-end hook or live W&B run is needed.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if wandb.run is not None:
            online_metrics = {name: value for name, value in metrics.items() if name not in invalid}
            online_metrics["loss/nonfinite"] = int(bool(invalid_losses))
            online_metrics.update({f"nonfinite/{name}": 1 for name in invalid})
            wandb.log(online_metrics, step=iteration, commit=False)
