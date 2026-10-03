"""Ensure the registered lifecycle is genuinely video-only and launchable."""
from cosmos3_ar_it2v.config import experiment


def test_video_recipe_and_fixed_batch_contract():
    c=experiment()
    assert c.model.config.action_gen is False
    assert c.model.config.sound_gen is False
    assert c.model.config.rectified_flow_training_config.normalize_loss_by_active
    data=c.dataloader_train.dataloader.datasets.video.dataset
    assert data.frame_stride==1  # native 30fps, no stride-2 frame dropping
    assert tuple(data.clip_frame_tiers)==(97,81,65,49,33,17)
    assert c.dataloader_train.max_samples_per_batch==4
    assert c.dataloader_train.max_sequence_length is None
    assert c.dataloader_train.dataloader.stateful
    assert c.dataloader_train.lazy_initialize_child_iterators
    assert c.optimizer.keys_to_select==['moe_gen','time_embedder','vae2llm','llm2vae']
    assert not c.optimizer.lr_multipliers
    assert c.job.wandb_mode=='online'
    assert c.trainer.max_iter==3000 and c.checkpoint.save_iter==500


def test_final_toml_composes_through_official_schema():
    from pathlib import Path
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    root=Path(__file__).resolve().parents[1]
    c=load_experiment_from_toml(str(root/'cosmos3_ar_it2v/configs/ego100h.toml'),[])
    # Native validate() initializes CUDA/distributed state; the real GPU
    # smoke exercises it. CPU regression checks the composed schema only.
    assert c.model.config.frames_per_chunk==4
    assert c.model.config.local_attention_frames==16
    assert c.model.config.parallelism.data_parallel_replicate_degree==2
    assert c.optimizer.lr==2e-5
    assert c.job.wandb_mode=='online'


def test_full_segment_recipe_uses_exclusive_token_budget():
    from cosmos3_ar_it2v.config import full_segment_experiment
    old=experiment()
    c=full_segment_experiment()
    assert c.dataloader_train.max_samples_per_batch is None
    assert c.dataloader_train.max_sequence_length==65536
    d=c.dataloader_train.dataloader.datasets.video.dataset
    assert d.sample_mode=='full_segment' and not d.random_window
    assert d.max_sequence_length==65536 and d.frame_stride==1
    assert c.optimizer==old.optimizer and c.scheduler==old.scheduler
    assert c.model.config.frames_per_chunk==old.model.config.frames_per_chunk
    assert c.model.config.local_attention_frames==old.model.config.local_attention_frames
    assert c.model.config.sigma_min==old.model.config.sigma_min
    assert c.model.config.sigma_max==old.model.config.sigma_max
    assert c.model.config.sigma_shift==old.model.config.sigma_shift
