from __future__ import annotations

import copy
from pathlib import Path

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.configs.base.defaults.callbacks import BASIC_CALLBACKS
from cosmos_framework.data.generator.joint_dataloader import RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModelConfig

from .ar_dataset import get_egoverse_ar_dataset
from .ar_v02_model import EgoVerseARV02Model
from .dataloader_state import EgoVerseDataLoaderStateCallback
from .wandb_compat import ensure_wandb_generate_id
from .wandb_metrics import EgoVerseLossWandbCallback
from .pretrain_probe import PretrainProbeCallback
from .formal_monitor import FormalTrainingMonitor


COSMOS_REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_CAMERA_ACTION_REPRESENTATION = "fixed_camera_wrist_local_delta_latent_v1"
FIXED_CAMERA_CONFIG_NAME = "rbs_wam_ar_v0_2_fixed_camera_wrist_local_delta_latent_v1"
DEFAULT_BASE_CHECKPOINT_PATH = "/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp"
DEFAULT_WAN_VAE_PATH = "/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
DEFAULT_TEXT_TOKENIZER_PATH = "/mnt/checkpoints/Cosmos3-Nano/text_tokenizer"
BASE_CHECKPOINT_PATH = f"${{oc.env:BASE_CHECKPOINT_PATH,{DEFAULT_BASE_CHECKPOINT_PATH}}}"
WAN_VAE_PATH = f"${{oc.env:WAN_VAE_PATH,{DEFAULT_WAN_VAE_PATH}}}"
TEXT_TOKENIZER_PATH = f"${{oc.env:TEXT_TOKENIZER_PATH,{DEFAULT_TEXT_TOKENIZER_PATH}}}"

# W&B 0.28 removed a helper still used by the native Cosmos initializer.
ensure_wandb_generate_id()

# Preserve the native logging and housekeeping lifecycle. Optional visualization,
# MoE diagnostics, norm/sigma monitoring, and MFU/OFU callbacks are omitted.
_EGOVERSE_BASIC_CALLBACKS = {
    _name: copy.deepcopy(BASIC_CALLBACKS[_name])
    for _name in (
        "wandb", "wandb_2x", "iter_speed", "manual_gc",
        "load_pretrained", "param_count", "sequence_packing_padding",
    )
}
_EGOVERSE_BASIC_CALLBACKS["egoverse_loss_wandb"] = L(EgoVerseLossWandbCallback)()
ConfigStore.instance().store(
    group="callbacks", package="trainer.callbacks", name="egoverse_basic", node=_EGOVERSE_BASIC_CALLBACKS
)


def _model_config(action_loss_weight: float = 1.0) -> dict:
    config = copy.deepcopy(NANO_MODEL_CONFIG)
    config["sound_gen"] = False
    config["ema"]["enabled"] = False
    # Joint action-token layouts are not safe under CP=2 in the current MoT
    # backward path.  Formal joint recipes use CP=1/FSDP-8 and a conservative
    # 60K packing cap; keep the dataset and packer limits sourced from here.
    config["max_num_tokens_after_packing"] = 60_000
    config["resolution"] = "480"
    config["tokenizer"]["vae_path"] = WAN_VAE_PATH
    config["activation_checkpointing"]["mode"] = "full"
    config["tokenizer"]["encode_exact_durations"] = None
    config["vlm_config"]["tokenizer"]["pretrained_model_name"] = TEXT_TOKENIZER_PATH
    config["vlm_config"]["tokenizer"]["config_variant"] = "hf"
    config["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] = str(
        COSMOS_REPO_ROOT / "packages/cosmos3/cosmos_framework/model/generator/reasoner/"
        "qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    )
    config["parallelism"].update(
        data_parallel_shard_degree=8,
        data_parallel_replicate_degree=1,
        context_parallel_shard_degree=1,
    )
    config["diffusion_expert_config"].update(
        base_fps=24,
        enable_action_state_embedding=True,
        enable_vision_condition_embedding=True,
        enable_fps_modulation=True,
        load_weights_from_pretrained=False,
        patch_spatial=2,
        unified_3d_mrope_temporal_modality_margin=15000,
        unified_3d_mrope_reset_spatial_ids=True,
    )
    config["rectified_flow_training_config"].update(
        loss_scale=1.0,
        action_loss_weight=action_loss_weight,
        independent_action_schedule=True,
        normalize_loss_by_active=True,
        shift_action=5,
        sample_level_loss_averaging=True,
        shift={"256": 3, "480": 5, "720": 10},
        train_time_video_distribution="waver",
        train_time_weight="uniform",
        use_discrete_rf=False,
    )
    config.update(
        video_temporal_causal=True,
        joint_attn_implementation="three_way",
        causal_training_strategy="teacher_forcing",
        action_tokens_per_latent=8,
        supervise_temporal_causal_actions=True,
    )
    config["compile"]["enabled"] = False
    return config


def _ar_v02_experiment(chunk_state_conditioning: bool):
    """Current multitask recipe, composed directly over official Cosmos defaults."""
    from .ar_v02_dataloader import JointChunkPackingDataLoader
    from .ar_v02_contract import ARTrainingContractCallback

    root = COSMOS_REPO_ROOT / "outputs/data_expansion_20260928"
    prepared = root / "prepared_v02"
    return LazyDict(
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
                group="ar_v0_2",
                name="multitask20h_joint_chunk_cond_v1",
                wandb_mode="online",
            ),
            model=L(EgoVerseARV02Model)(
                config=OmniMoTCausalModelConfig(
                    **LazyDict(_model_config(), flags={"allow_objects": True}),
                    teacher_forcing_kv_implementation="singleview_threeway_kv",
                    teacher_forcing_frames_per_chunk=4,
                    teacher_forcing_detach_clean_kv=False,
                ),
                chunk_state_conditioning=chunk_state_conditioning,
                seed=42,
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
                    "action_state_embed",
                    "vision_condition_embed",
                ],
                lr=2.0e-5,
                # Neutral until the next explicitly approved optimizer recipe.
                lr_multipliers={},
                optimizer_type="FusedAdam",
                weight_decay=0.05,
            ),
            scheduler=dict(
                lr_scheduler_type="LambdaCosine",
                warm_up_steps=[100],
                cycle_lengths=[1200],
                f_start=[0.0],
                f_max=[1.0],
                f_min=[0.1],
                verbosity_interval=0,
            ),
            trainer=dict(
                distributed_parallelism="fsdp",
                grad_accum_iter=1,
                logging_iter=1,
                max_iter=1200,
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
                    manual_gc=dict(every_n=1, gc_level=2, warm_up=0),
                    dataloader_state=L(EgoVerseDataLoaderStateCallback)(),
                    pretrain_probe=L(PretrainProbeCallback)(enabled=False),
                    ar_v02_contract=L(ARTrainingContractCallback)(
                        state_normalizer=str(prepared / "chunk_state_normalizer.json"),
                        action_normalizer=str(prepared / "future_frame_delta_normalizer.json"),
                        valid_windows_manifest=str(prepared / "valid_windows.json"),
                        official_checkpoint=BASE_CHECKPOINT_PATH,
                        check_raw_gradients=True,
                    ),
                    skip_nan_step=dict(max_consecutive_nan=20),
                ),
            ),
            checkpoint=dict(
                broadcast_via_filesystem=True,
                dcp_async_mode_enabled=False,
                enable_gcs_patch_in_boto3=False,
                keys_not_to_resume=[],
                keys_to_skip_loading=["net_ema.", "action_state_embed", "vision_condition_embed"],
                load_ema_to_reg=False,
                load_path=BASE_CHECKPOINT_PATH,
                load_training_state=False,
                only_load_scheduler_state=False,
                save_iter=600,
                strict_resume=True,
                verbose=True,
                load_from_object_store=dict(bucket="", credentials="", enabled=False),
                save_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            dataloader_train=L(JointChunkPackingDataLoader)(
                joint_max_samples=4,
                lazy_initialize_child_iterators=True,
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
                    in_order=True,
                    stateful=True,
                    num_workers=3,
                    persistent_workers=True,
                    pin_memory=True,
                    prefetch_factor=2,
                    sampler=None,
                    datasets=dict(
                        egoverse=dict(
                            ratio=1,
                            dataset=L(get_egoverse_ar_dataset)(
                                episodes_manifest=str(root / "episodes.csv"),
                                segments_manifest=str(root / "segments.csv"),
                                tokenizer_config="${model.config.vlm_config.tokenizer}",
                                cfg_dropout_rate=0.1,
                                iterable_shuffle=True,
                                seed=42,
                                max_sequence_length="${model.config.max_num_tokens_after_packing}",
                                prompt_mode="segment_only",
                                # Still required by the dataset's base state contract.
                                state_normalizer=str(
                                    COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/artifacts/"
                                    "cosmos3_action_contract/v2/normalizers/state_normalizer.json"
                                ),
                                chunk_state_normalizer=str(prepared / "chunk_state_normalizer.json"),
                                future_normalizer=str(prepared / "future_frame_delta_normalizer.json"),
                                valid_windows_manifest=str(prepared / "valid_windows.json"),
                                frame_stride=2,
                                clip_frame_tiers=(129, 65, 33),
                                speed_factors={"egoverse": 0.5},
                                random_window=True,
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


def _ar_v02_fixed_camera_experiment():
    """Explicit fixed-camera recipe; missing new artifacts fail at callback construction.

    Keep this recipe separate from the registered legacy v0.2 aliases.  In
    particular, do not reuse the old wrist-local codec or its normalizers.
    """
    experiment = _ar_v02_experiment(True)
    # Eight-rank save/resume replay is exact with fixed cuDNN kernels.
    # Autotuning produced different losses after restoring identical data/RNG.
    experiment["trainer"]["cudnn"].update(benchmark=False, deterministic=True)
    root = COSMOS_REPO_ROOT / "outputs/data_expansion_20260928"
    prepared = root / "prepared_fixed_camera_wrist_local_delta_latent_v1_t273"
    codec_root = COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v3_wrist_local_pca15_train744"
    fixed_hand_codecs = (
        str(codec_root / "right_pca15.pt"),
        str(codec_root / "left_pca15.pt"),
    )
    state_normalizer = str(prepared / "chunk_state_normalizer.json")
    future_normalizer = str(prepared / "future_normalizer.json")
    valid_windows = str(prepared / "valid_windows.json")

    experiment["job"].update(
        group="ar_v0_2_fixed_camera",
        name="multitask20h_fixed_camera_wrist_local_delta_latent_v1",
    )
    experiment["model"]["action_representation"] = FIXED_CAMERA_ACTION_REPRESENTATION
    experiment["model"]["config"].parallelism.data_parallel_replicate_degree = 2
    experiment["optimizer"]["lr_multipliers"] = dict.fromkeys(
        ("action2llm", "llm2action", "action_modality_embed", "action_state_embed", "vision_condition_embed"), 5
    )
    experiment["scheduler"]["cycle_lengths"] = [1000]
    experiment["trainer"]["max_iter"] = 1000
    experiment["checkpoint"]["save_iter"] = 500
    experiment["trainer"]["callbacks"]["formal_monitor"] = L(FormalTrainingMonitor)()
    dataset = experiment["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
    dataset.update(
        action_representation=FIXED_CAMERA_ACTION_REPRESENTATION,
        right_codec=fixed_hand_codecs[0],
        left_codec=fixed_hand_codecs[1],
        # Explicitly remove the legacy base state profile from this recipe.
        state_normalizer=None,
        chunk_state_normalizer=state_normalizer,
        future_normalizer=future_normalizer,
        valid_windows_manifest=valid_windows,
        clip_frame_tiers=(273, 257, 129, 65, 33),
    )
    contract = experiment["trainer"]["callbacks"]["ar_v02_contract"]
    contract.update(
        representation=FIXED_CAMERA_ACTION_REPRESENTATION,
        state_normalizer=state_normalizer,
        action_normalizer=future_normalizer,
        valid_windows_manifest=valid_windows,
        right_hand_codec=fixed_hand_codecs[0],
        left_hand_codec=fixed_hand_codecs[1],
    )
    return experiment


# Historical names retain their original action semantics for saved configs.
# Only ar_v0_2.toml is kept as the legacy regression entrypoint.
for _name in ("ar_v0_2", "ar_v0_2_c", "ar_v0_2_multitask"):
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name="egoverse_joint_video_hand_pose_" + _name,
        node=_ar_v02_experiment(True),
    )

# A separate name prevents legacy aliases/checkpoints from silently changing
# semantics. It is selectable explicitly once all fixed-camera assets exist.
ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=FIXED_CAMERA_CONFIG_NAME,
    node=_ar_v02_fixed_camera_experiment(),
)


def make_config():
    """Expose the native Cosmos config factory required by inference loaders."""
    from cosmos_framework.configs.base.config import make_config as make_base_config

    return make_base_config()
