from __future__ import annotations

import copy
from pathlib import Path

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.configs.base.defaults.callbacks import BASIC_CALLBACKS
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from .dataset import get_egoverse_cosmos_dataset
from .model import EgoVerseOmniMoTModel
from .wandb_compat import ensure_wandb_generate_id
from .wandb_metrics import EgoVerseLossWandbCallback, LossOnlyWandBCallback


COSMOS_REPO_ROOT = Path(__file__).resolve().parents[2]


# W&B 0.28 removed a helper still used by the native Cosmos initializer.
ensure_wandb_generate_id()

# Retain console progress and training housekeeping, but omit the expensive
# diagnostic callbacks that create thousands of unrelated W&B charts.
_EGOVERSE_BASIC_CALLBACKS = {
    _name: copy.deepcopy(BASIC_CALLBACKS[_name])
    for _name in ("iter_speed", "manual_gc", "load_pretrained")
}
_EGOVERSE_BASIC_CALLBACKS["wandb"] = L(LossOnlyWandBCallback)()
_EGOVERSE_BASIC_CALLBACKS["egoverse_loss_wandb"] = L(EgoVerseLossWandbCallback)()
ConfigStore.instance().store(
    group="callbacks", package="trainer.callbacks", name="egoverse_basic", node=_EGOVERSE_BASIC_CALLBACKS
)


def _model_config(action_loss_weight: float = 7.0) -> dict:
    config = copy.deepcopy(NANO_MODEL_CONFIG)
    config["sound_gen"] = False
    config["ema"]["enabled"] = False
    config["max_num_tokens_after_packing"] = 85_000
    config["resolution"] = "480"
    config["tokenizer"]["vae_path"] = "/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    config["activation_checkpointing"]["mode"] = "full"
    config["tokenizer"]["encode_exact_durations"] = None
    config["vlm_config"]["tokenizer"]["pretrained_model_name"] = "/mnt/checkpoints/Cosmos3-Nano/text_tokenizer"
    config["vlm_config"]["tokenizer"]["config_variant"] = "hf"
    config["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] = str(
        COSMOS_REPO_ROOT
        / "packages/cosmos3/cosmos_framework/model/generator/reasoner/"
        "qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    )
    config["parallelism"].update(
        data_parallel_shard_degree=4,
        data_parallel_replicate_degree=1,
        context_parallel_shard_degree=2,
    )
    config["diffusion_expert_config"].update(
        base_fps=24,
        enable_fps_modulation=True,
        load_weights_from_pretrained=False,
        patch_spatial=2,
        unified_3d_mrope_temporal_modality_margin=15000,
        unified_3d_mrope_reset_spatial_ids=True,
    )
    config["rectified_flow_training_config"].update(
        loss_scale=10.0,
        action_loss_weight=action_loss_weight,
        independent_action_schedule=False,
        sample_level_loss_averaging=True,
        shift={"256": 3, "480": 5, "720": 10},
        train_time_video_distribution="waver",
        train_time_weight="uniform",
        use_discrete_rf=False,
    )
    config["compile"]["enabled"] = False
    return config


egoverse_joint_video_hand_pose_overfit_v0_0 = LazyDict(
    dict(
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
            {"override /callbacks": ["egoverse_basic", "optimization", "job_monitor"]},
            {"override /ema": "power"},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="joint_video_hand_pose",
            group="overfit",
            name="overfit_v0.0",
            wandb_mode="online",
        ),
        model=L(EgoVerseOmniMoTModel)(
            config=_model_config(),
            lambda_out_of_fov=0.0,
            subblock_equal_weight=True,
            _recursive_=False,
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-8,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=2.0e-5,
            lr_multipliers={"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0},
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaCosine",
            warm_up_steps=[20],
            cycle_lengths=[2000],
            f_start=[0.0],
            f_max=[1.0],
            f_min=[0.1],
            verbosity_interval=0,
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=2000,
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
                device_monitor=dict(every_n=200, log_memory_detail=True, save_s3=False, step_size=1),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=10, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=20, gc_level=1, warm_up=1),
                skip_nan_step=dict(max_consecutive_nan=20),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=True,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=False,
            keys_not_to_resume=[],
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="${oc.env:BASE_CHECKPOINT_PATH,/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464}",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=300,
            strict_resume=True,
            verbose=True,
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="egoverse",
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
                    egoverse=dict(
                        ratio=1,
                        dataset=L(get_egoverse_cosmos_dataset)(
                            episodes_manifest=(
                                "/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose/artifacts/"
                                "cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/episodes.csv"
                            ),
                            segments_manifest=(
                                "/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose/artifacts/"
                                "cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/segments.csv"
                            ),
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            cfg_dropout_rate=0.1,
                            iterable_shuffle=True,
                            seed=42,
                            max_sequence_length="${model.config.max_num_tokens_after_packing}",
                            prompt_mode="episode_context_and_segment",
                            state_normalizer=(
                                "/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose/artifacts/"
                                "cosmos3_action_contract/v2/normalizers/state_normalizer.json"
                            ),
                            future_normalizer=(
                                "/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose/artifacts/"
                                "cosmos3_action_contract/v2/normalizers/future_delta_normalizer.json"
                            ),
                        ),
                    )
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# Same data/model/loss contract as overfit_v0.0, with the requested balanced
# effective learning rates: shared/video 4x and action 5x from a 2e-5 base.
egoverse_joint_video_hand_pose_overfit_v0_2_lr_balanced = copy.deepcopy(
    egoverse_joint_video_hand_pose_overfit_v0_0
)
egoverse_joint_video_hand_pose_overfit_v0_2_lr_balanced["job"]["name"] = "overfit_v0.2_lr_balanced"
egoverse_joint_video_hand_pose_overfit_v0_2_lr_balanced["optimizer"]["lr"] = 2.0e-5
egoverse_joint_video_hand_pose_overfit_v0_2_lr_balanced["optimizer"]["lr_multipliers"] = {
    "moe_gen": 4.0,
    "time_embedder": 4.0,
    "vae2llm": 4.0,
    "llm2vae": 4.0,
    "action2llm": 5.0,
    "llm2action": 5.0,
    "action_modality_embed": 5.0,
}
egoverse_joint_video_hand_pose_overfit_v0_2_lr_balanced["scheduler"]["warm_up_steps"] = [100]


ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="egoverse_joint_video_hand_pose_overfit_v0_0",
    node=egoverse_joint_video_hand_pose_overfit_v0_0,
)
ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="egoverse_joint_video_hand_pose_overfit_v0_2_lr_balanced",
    node=egoverse_joint_video_hand_pose_overfit_v0_2_lr_balanced,
)


def make_config():
    """Expose the native Cosmos config factory required by inference loaders."""
    from cosmos_framework.configs.base.config import make_config as make_base_config

    return make_base_config()
