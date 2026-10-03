"""V0.4 registration: V0.3.1 recipe with continuous video codec semantics."""
from __future__ import annotations

import attrs
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from cosmos_framework.utils.lazy_config import LazyCall as L

from .ar_v031_config import _ar_v031_experiment
from .ar_v04_model import EgoVerseARV04Model, EgoVerseARV04ModelConfig
from .ar_v04_checkpoint import ARV04VideoFormatCallback

AR_V04_CONFIG_NAME = "rbs_wam_ar_v0_4_continuous_video_v1"


def _ar_v04_experiment():
    experiment = _ar_v031_experiment()
    model = experiment["model"]
    base = model["config"]
    model_config = EgoVerseARV04ModelConfig(**{
        field.name: base[field.name] for field in attrs.fields(OmegaConf.get_type(base))
    })
    experiment["model"] = L(EgoVerseARV04Model)(
        **{key: value for key, value in model.items() if key not in ("_target_", "config")},
        config=model_config,
    )
    experiment["job"].update(group="ar_v0_4", name="ar_v0_4_continuous_video_v1", wandb_mode="online")
    experiment["trainer"]["callbacks"]["ar_v04_video_format"] = L(ARV04VideoFormatCallback)()
    return experiment


ConfigStore.instance().store(
    group="experiment", package="_global_", name=AR_V04_CONFIG_NAME,
    node=_ar_v04_experiment(),
)


def make_config():
    from cosmos_framework.configs.base.config import make_config as make_base_config
    return make_base_config()
