"""Parallel prediction supervision with a shared noisy-GT history stream."""
from hydra.core.config_store import ConfigStore
from cosmos_framework.utils.lazy_config import LazyCall as L, LazyDict

from .config import ROOT, full_segment_experiment
from .model_v04 import ARIT2VModelV04, ARIT2VModelV04Config

CONFIG_NAME = 'rbs_wam_ar_it2v_v0_4_parallel_tf'


def parallel_tf_experiment():
    # This is the source-data budget. H/P expansion is reported separately;
    # User-approved 50k ceiling, rounded down to the native 128-token alignment.
    c = full_segment_experiment(token_budget=49920)
    c.dataloader_train.dataloader.datasets.video.dataset.segment_statistics_path = str(
        ROOT / 'outputs/maintenance/full_segment_retention_49920_20261006/summary.json')
    model_config = dict(c.model.config)
    model_config.update(local_attention_frames=32, history_noise_mode='lingbot_public',
                        clean_history_probability=.5, target_history_seed=42)
    c.model = L(ARIT2VModelV04)(
        config=ARIT2VModelV04Config(**LazyDict(model_config, flags={'allow_objects': True})),
        _recursive_=False)
    c.job.update(group='ar_it2v_v0_4', name='ar_it2v_v0_4_parallel_tf')
    c.trainer.max_iter = 5000
    c.scheduler.cycle_lengths = [5000]
    return c


ConfigStore.instance().store(group='experiment', package='_global_',
                             name=CONFIG_NAME, node=parallel_tf_experiment())
