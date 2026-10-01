"""Explicit V0.3 experiment registration over the preserved V0.2 assets."""
from __future__ import annotations

import json
import math
from pathlib import Path

import attrs
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.functional.lr_scheduler import LambdaWarmUpCosineScheduler
from cosmos_framework.utils.lazy_config import LazyCall as L

from .ar_v03_loss import AR_V03_ACTION_CHANNEL_WEIGHTS
from .ar_v03_model import EgoVerseARV03Model, EgoVerseARV03ModelConfig
from .config import (
    COSMOS_REPO_ROOT, FIXED_CAMERA_ACTION_REPRESENTATION,
    _ar_v02_fixed_camera_experiment,
)

AR_V03_CONFIG_NAME = "rbs_wam_ar_v0_3_diffusion_forcing_wrist_weight_v1"
AR_V03_LR_MULTIPLIERS = dict.fromkeys((
    "action2llm", "llm2action", "action_modality_embed",
    "action_state_embed", "vision_condition_embed",
), 1.0)
AR_V03_LR_STEPS = (0, 100, 1500, 3000)


def learning_rate_receipt(config):
    """Assert the final CLI-composed schedule and evaluate the official lambda."""
    if not math.isclose(float(config.optimizer.lr), 1e-4, rel_tol=1e-12):
        raise ValueError("V0.3 requires peak optimizer.lr=1e-4")
    multipliers = dict(config.optimizer.lr_multipliers)
    if multipliers != AR_V03_LR_MULTIPLIERS:
        raise ValueError("V0.3 requires exactly the five explicit LR multipliers, all 1")
    scheduler = config.scheduler
    expected = dict(
        warm_up_steps=[100], cycle_lengths=[3000],
        f_start=[0.0], f_max=[1.0], f_min=[0.3],
    )
    if scheduler.lr_scheduler_type != "LambdaCosine":
        raise ValueError("V0.3 requires the official LambdaCosine scheduler")
    for name, value in expected.items():
        if list(scheduler[name]) != value:
            raise ValueError(f"V0.3 scheduler mismatch for {name}: {scheduler[name]} != {value}")
    official = LambdaWarmUpCosineScheduler(**expected, verbosity_interval=0)
    steps = [
        dict(step=step, multiplier=float(official(step)),
             lr=float(config.optimizer.lr) * float(official(step)))
        for step in AR_V03_LR_STEPS
    ]
    if not math.isclose(steps[-1]["lr"], 3e-5, rel_tol=1e-12):
        raise ValueError("V0.3 terminal LR is not 3e-5")
    return dict(
        schema="ar_v03_learning_rate_receipt_v1",
        scheduler="LambdaCosine", base_lr=float(config.optimizer.lr),
        lr_multipliers=multipliers, scheduler_config=expected,
        theoretical_lr=steps, max_iter=int(config.trainer.max_iter),
        save_iter=int(config.checkpoint.save_iter),
    )


class ARV03LearningRateReceiptCallback(Callback):
    """Read final config before the official optimizer is initialized."""
    def on_optimizer_init_start(self):
        receipt = learning_rate_receipt(self.config)
        if not dist.is_initialized() or dist.get_rank() == 0:
            root = Path(self.config.job.path_local)
            root.mkdir(parents=True, exist_ok=True)
            (root / "learning_rate_receipt.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False) + "\n"
            )
            print("AR_V03_LEARNING_RATE_RECEIPT " + json.dumps(receipt), flush=True)


def _ar_v03_experiment():
    experiment = _ar_v02_fixed_camera_experiment()
    # Preserve nested LazyDict/OmegaConf nodes; to_object converts typed
    # LazyDict fields to plain dicts which cannot be wrapped as this schema.
    base = experiment["model"]["config"]
    model_config = EgoVerseARV03ModelConfig(**{
        field.name: base[field.name] for field in attrs.fields(OmegaConf.get_type(base))
    })
    model_config.causal_training_strategy = "diffusion_forcing"
    model_config.sigma_diffusion_forcing = 0.02
    model_config.sigma_small = 0.02
    model_config.prefix_low_noise_enabled = True
    model_config.sigma_hist_max = 0.1
    model_config.action_channel_weights = list(AR_V03_ACTION_CHANNEL_WEIGHTS)
    experiment["model"] = L(EgoVerseARV03Model)(
        config=model_config, chunk_state_conditioning=True, seed=42,
        action_representation=FIXED_CAMERA_ACTION_REPRESENTATION,
        history_video_noise_prob=0.0, history_video_noise_sigma_max=0.2,
        _recursive_=False,
    )
    experiment["job"].update(
        group="ar_v0_3", name="diffusion_forcing_wrist_weight_v1",
        wandb_mode="online",
    )
    experiment["optimizer"].update(lr=1e-4, lr_multipliers=AR_V03_LR_MULTIPLIERS.copy())
    experiment["scheduler"].update(
        lr_scheduler_type="LambdaCosine", warm_up_steps=[100], cycle_lengths=[3000],
        f_start=[0.0], f_max=[1.0], f_min=[0.3],
    )
    experiment["trainer"]["max_iter"] = 3000
    experiment["checkpoint"]["save_iter"] = 500
    experiment["trainer"]["callbacks"]["ar_v03_lr_receipt"] = L(ARV03LearningRateReceiptCallback)()
    return experiment


ConfigStore.instance().store(
    group="experiment", package="_global_", name=AR_V03_CONFIG_NAME,
    node=_ar_v03_experiment(),
)


def make_config():
    from cosmos_framework.configs.base.config import make_config as make_base_config
    return make_base_config()
