from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore
from cosmos_framework.configs.base.defaults.callbacks import BASIC_CALLBACKS
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from .data import get_egoverse_it2v_dataset
from .wandb_compat import ensure_wandb_generate_id


ensure_wandb_generate_id()


model = copy.deepcopy(NANO_MODEL_CONFIG)
model.update(
    sound_gen=False,
    action_gen=True,
    resolution="480",
    max_num_tokens_after_packing=85_000,
    ema=dict(enabled=False, iteration_shift=0, rate=0.1),
)
model["parallelism"].update(
    data_parallel_shard_degree=4,
    data_parallel_replicate_degree=1,
    context_parallel_shard_degree=2,
)
model["compile"]["enabled"] = False
model["activation_checkpointing"]["mode"] = "full"
model["tokenizer"]["vae_path"] = "/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
model["vlm_config"]["tokenizer"]["pretrained_model_name"] = "/mnt/checkpoints/Cosmos3-Nano/text_tokenizer"
model["vlm_config"]["tokenizer"]["config_variant"] = "hf"
model["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] = (
    "/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/model/generator/reasoner/"
    "qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
)
model["rectified_flow_training_config"].update(
    loss_scale=1.0, shift={"256": 3, "480": 5, "720": 10}, sample_level_loss_averaging=True
)

callbacks = {name: copy.deepcopy(BASIC_CALLBACKS[name]) for name in ("iter_speed", "manual_gc", "wandb")}
ConfigStore.instance().store(
    group="callbacks", package="trainer.callbacks", name="egoverse_it2v_basic", node=callbacks
)

egoverse_it2v_v1 = LazyDict(dict(
    defaults=[
        {"override /data_train": None},
        {"override /data_val": None},
        {"override /model": "mot_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdacosine"},
        {"override /tokenizer": "wan2pt2_tokenizer"},
        {"override /sound_tokenizer": None},
        {"override /vlm_config": None},
        {"override /checkpoint": "local"},
        {"override /callbacks": ["egoverse_it2v_basic", "optimization", "job_monitor"]},
        {"override /ema": "power"},
        {"override /ckpt_type": "dcp"},
        "_self_",
    ],
    job=dict(project="egoverse_it2v", group="train", name="v1", wandb_mode="online"),
    model=L(OmniMoTModel)(config=model, _recursive_=False),
    optimizer=dict(
        betas=[0.9, 0.99],
        eps=1.0e-8,
        fused=True,
        keys_to_select=["moe_gen", "time_embedder", "vae2llm", "llm2vae"],
        lr=2.0e-5,
        lr_multipliers={},
        optimizer_type="FusedAdam",
        weight_decay=0.05,
    ),
    scheduler=dict(
        lr_scheduler_type="LambdaCosine",
        warm_up_steps=[30],
        cycle_lengths=[600],
        f_start=[0.0],
        f_max=[1.0],
        f_min=[0.1],
        verbosity_interval=0,
    ),
    trainer=dict(
        distributed_parallelism="fsdp",
        grad_accum_iter=1,
        logging_iter=1,
        max_iter=600,
        max_val_iter=None,
        run_validation=False,
        run_validation_on_start=False,
        save_zero_checkpoint=False,
        seed=42,
        timeout_period=999999999,
        compile_config=dict(recompile_limit=8, use_duck_shape=False),
        cudnn=dict(benchmark=True, deterministic=False),
        ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
        grad_scaler_args=dict(enabled=False),
        callbacks=dict(
            grad_clip=dict(clip_norm=1.0, force_finite=True),
            low_precision=dict(update_iter=1),
            manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
            skip_nan_step=dict(max_consecutive_nan=20),
        ),
    ),
    checkpoint=dict(
        broadcast_via_filesystem=True,
        dcp_async_mode_enabled=False,
        keys_not_to_resume=[],
        keys_to_skip_loading=["net_ema."],
        load_ema_to_reg=False,
        load_path="${oc.env:BASE_CHECKPOINT_PATH,/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464}",
        load_training_state=False,
        only_load_scheduler_state=False,
        save_iter=300,
        strict_resume=True,
        verbose=True,
    ),
    dataloader_train=L(PackingDataLoader)(
        audio_sample_rate=48000,
        dataset_name="egoverse_it2v",
        max_samples_per_batch=None,
        max_sequence_length="${model.config.max_num_tokens_after_packing}",
        patch_spatial=2,
        sound_latent_fps=0,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        dataloader=L(RankPartitionedDataLoader)(
            batch_size=1,
            in_order=False,
            num_workers=3,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            sampler=None,
            datasets=dict(
                egoverse_it2v=dict(
                    ratio=1,
                    dataset=L(get_egoverse_it2v_dataset)(
                        episodes_manifest="${oc.env:EGOVERSE_EPISODES_MANIFEST,/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose/artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/episodes.csv}",
                        segments_manifest="${oc.env:EGOVERSE_SEGMENTS_MANIFEST,/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose/artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/segments.csv}",
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        cfg_dropout_rate=0.1,
                        iterable_shuffle=True,
                        seed=42,
                        max_sequence_length="${model.config.max_num_tokens_after_packing}",
                        prompt_mode="episode_context_and_segment",
                    ),
                )
            ),
        ),
    ),
    dataloader_val=None,
    upload_reproducible_setup=False,
))

ConfigStore.instance().store(
    group="experiment", package="_global_", name="egoverse_it2v_v1", node=egoverse_it2v_v1
)
