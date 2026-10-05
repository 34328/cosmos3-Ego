"""Target-only video DF; retain the validated full-segment data lifecycle."""
from hydra.core.config_store import ConfigStore
from cosmos_framework.utils.lazy_config import LazyCall as L, LazyDict

from .config import ROOT, full_segment_experiment
from .model_v03 import ARIT2VModelV03, ARIT2VModelV03Config

CONFIG_NAME = 'rbs_wam_ar_it2v_v0_3_target_only_df'


def target_only_experiment():
    c = full_segment_experiment(token_budget=65536)
    c.dataloader_train.dataloader.datasets.video.dataset.segment_statistics_path = str(
        ROOT /
        'outputs/maintenance/full_segment_retention_65536_20261005/summary.json')
    model_config = dict(c.model.config)
    model_config.update(local_attention_frames=32, history_noise_mode='lingbot_public',
                        clean_history_probability=.5, target_history_seed=42)
    c.model = L(ARIT2VModelV03)(
        config=ARIT2VModelV03Config(**LazyDict(model_config, flags={'allow_objects': True})),
        _recursive_=False)
    c.job.update(group='ar_it2v_v0_3', name='ar_it2v_v0_3_target_only_df')
    c.trainer.max_iter = 5000
    c.scheduler.cycle_lengths = [5000]
    return c


ConfigStore.instance().store(group='experiment', package='_global_',
                             name=CONFIG_NAME, node=target_only_experiment())
