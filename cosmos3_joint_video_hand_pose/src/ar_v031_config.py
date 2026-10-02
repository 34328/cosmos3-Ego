"""Numerator-only prefix ablation; V0.3.0 training recipe and lifecycle retained."""
from __future__ import annotations

import json
from pathlib import Path

import attrs
import torch
import torch.distributed as dist
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cosmos_framework.utils import distributed
from cosmos_framework.utils.lazy_config import LazyCall as L

from .ar_v03_config import _ar_v03_experiment
from .ar_v031_model import EgoVerseARV031Model, EgoVerseARV031ModelConfig
from .wandb_metrics import EgoVerseLossWandbCallback

AR_V031_CONFIG_NAME = "rbs_wam_ar_v0_3_1_prefix_numerator_only_v1"
GROUP_MSE_NAMES = (
    "diagnostic/video_prefix_raw_mse", "diagnostic/video_target_raw_mse",
    "diagnostic/action_prefix_raw_mse", "diagnostic/action_target_raw_mse",
)


class ARV031GroupMSECallback(EgoVerseLossWandbCallback):
    """Keep the existing callback slot/raw8 and add detached coordinate-pooled MSE."""

    def __init__(self, nonfinite_detail_iterations=5):
        super().__init__(nonfinite_detail_iterations=nonfinite_detail_iterations)
        self._group_moments = None

    @torch.no_grad()
    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        super().on_training_step_batch_end(model, data_batch, output_batch, loss, iteration)
        current = model._ar_v031_group_mse_moments.detach()
        if self._group_moments is None:
            self._group_moments = current.clone()
        else:
            self._group_moments.add_(current)

    @torch.no_grad()
    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if self._group_moments is None:
            self.on_training_step_batch_end(model, data_batch, output_batch, loss, iteration)
        moments = self._group_moments
        self._group_moments = None
        if dist.is_initialized():
            dist.all_reduce(moments)
        if distributed.is_rank0():
            values = moments.cpu().tolist()
            metrics = {name: total / count if count else 0.0
                       for name, (total, count) in zip(GROUP_MSE_NAMES, values, strict=True)}
            counts = {name + "_coordinates": count
                      for name, (_, count) in zip(GROUP_MSE_NAMES, values, strict=True)}
            path = Path(self.config.job.path_local) / "ar_v031_group_mse.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(step=int(iteration), **metrics, **counts),
                                        allow_nan=False) + "\n")
            if wandb.run is not None:
                wandb.log(metrics | counts, step=iteration, commit=False)
        # Same callback position and original logging behavior; do not let the
        # native W&B callback commit this step before adding the four diagnostics.
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)


def _ar_v031_experiment():
    experiment = _ar_v03_experiment()
    model = experiment["model"]
    base = model["config"]
    model_config = EgoVerseARV031ModelConfig(**{
        field.name: base[field.name] for field in attrs.fields(OmegaConf.get_type(base))
    })
    experiment["model"] = L(EgoVerseARV031Model)(
        **{key: value for key, value in model.items() if key not in ("_target_", "config")},
        config=model_config,
    )
    experiment["job"].update(
        group="ar_v0_3_1", name="prefix_numerator_only_v1", wandb_mode="online"
    )
    experiment["trainer"]["callbacks"]["egoverse_loss_wandb"] = L(ARV031GroupMSECallback)(
        nonfinite_detail_iterations=5
    )
    return experiment


ConfigStore.instance().store(
    group="experiment", package="_global_", name=AR_V031_CONFIG_NAME,
    node=_ar_v031_experiment(),
)


def make_config():
    from cosmos_framework.configs.base.config import make_config as make_base_config
    return make_base_config()

