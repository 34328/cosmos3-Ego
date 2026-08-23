"""Minimal project-specific W&B logging without changing the Cosmos core."""

from __future__ import annotations

import torch
import wandb

from cosmos_framework.callbacks.wandb_log import _LossRecord
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

LR_METRIC_NAMES = {
    "optim/lr_base",
    "optim/lr_action",
}


def filter_wandb_metrics(metrics: dict) -> dict:
    """Keep overfit_v0.0 losses and the two effective learning rates."""
    allowed = set(LOSS_METRIC_SOURCES) | LR_METRIC_NAMES
    return {key: value for key, value in metrics.items() if key in allowed}


class LossOnlyWandBCallback(WandBCallback):
    """Initialize W&B normally while suppressing unrelated native metrics."""

    def __init__(self) -> None:
        super().__init__()
        self._unfiltered_log = None

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        super().on_train_start(model, iteration)
        if not distributed.is_rank0() or wandb.run is None:
            return
        self._unfiltered_log = wandb.log

        def loss_only_log(metrics: dict, *args, **kwargs):
            filtered = filter_wandb_metrics(metrics)
            if filtered:
                return self._unfiltered_log(filtered, *args, **kwargs)
            return None

        wandb.log = loss_only_log

    def on_before_optimizer_step(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del model, optimizer, grad_scaler
        if iteration % self.config.trainer.logging_iter or not distributed.is_rank0() or wandb.run is None:
            return

        # Cosmos creates separate parameter groups for the 1x base modules and
        # the 5x action projections.  Report both effective values; logging only
        # scheduler.get_last_lr()[0] can hide the action projection LR.
        learning_rates = [float(lr) for lr in scheduler.get_last_lr()]
        if not learning_rates:
            return
        wandb.log(
            {
                "optim/lr_base": min(learning_rates),
                "optim/lr_action": max(learning_rates),
            },
            step=iteration,
        )

    def on_training_step_end(self, *args, **kwargs) -> None:
        del args, kwargs

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if distributed.is_rank0() and self._unfiltered_log is not None:
            wandb.log = self._unfiltered_log
        super().on_train_end(model, iteration)


def extract_loss_metrics(output_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the exact loss series required for the joint training dashboard."""
    required = tuple(BASE_LOSS_METRIC_SOURCES.values())
    missing = [source for source in required if source not in output_batch]
    if missing:
        raise KeyError(f"EgoVerse W&B metrics missing model outputs: {missing}")
    return {name: output_batch[source] for name, source in LOSS_METRIC_SOURCES.items() if source in output_batch}


class EgoVerseLossWandbCallback(Callback):
    """Average component losses over the log window and all distributed ranks."""

    def __init__(self) -> None:
        super().__init__()
        self._records = {name: _LossRecord(name=name) for name in LOSS_METRIC_SOURCES}

    @torch.no_grad()
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, loss
        for name, value in extract_loss_metrics(output_batch).items():
            self._records[name].loss += value.detach().float()
            self._records[name].iter_count += 1

        if iteration % self.config.trainer.logging_iter:
            return

        # _LossRecord performs the required all-reduce; every rank must call it.
        metrics = {name: record.get_stat() for name, record in self._records.items()}
        if distributed.is_rank0() and wandb.run is not None:
            wandb.log(metrics, step=iteration)
