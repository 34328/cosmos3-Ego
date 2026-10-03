"""Pure-video AR adaptation; official Cosmos3 lifecycle and GEN optimizer."""
from pathlib import Path
import copy
from hydra.core.config_store import ConfigStore
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.configs.base.defaults.callbacks import BASIC_CALLBACKS
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L, LazyDict
from .wandb_compat import ensure_wandb_generate_id
from .model import ARIT2VModel, ARIT2VModelConfig
from .dataset import get_egoverse_it2v_dataset
from .monitor import VideoTrainingMonitor
from .dataloader import RecoverablePackingDataLoader, IT2VDataLoaderStateCallback

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = 'rbs_wam_ar_it2v_v0_1_ego100h_cmd_stage1'
FULL_SEGMENT_CONFIG_NAME = 'rbs_wam_ar_it2v_v0_2_ego100h_full_segments'
ensure_wandb_generate_id()
callbacks = {k:copy.deepcopy(BASIC_CALLBACKS[k]) for k in (
    'wandb','wandb_2x','iter_speed','manual_gc','load_pretrained','param_count','sequence_packing_padding')}
# Ordered before native optimization callbacks so nonfinite gradients cannot be sanitized.
callbacks['ar_it2v_monitor'] = L(VideoTrainingMonitor)()
callbacks['dataloader_state'] = L(IT2VDataLoaderStateCallback)()
ConfigStore.instance().store(group='callbacks',package='trainer.callbacks',name='ar_it2v_basic',node=callbacks)


def experiment():
    c = copy.deepcopy(NANO_MODEL_CONFIG)
    c.update(action_gen=False,sound_gen=False,vision_gen=True,resolution='480',
        video_temporal_causal=True,joint_attn_implementation='three_way',causal_training_strategy='diffusion_forcing',
        max_num_tokens_after_packing=45056)
    c['ema']['enabled'] = False
    c['compile']['enabled'] = False
    c['activation_checkpointing']['mode'] = 'full'
    c['parallelism'].update(data_parallel_shard_degree=8,data_parallel_replicate_degree=2,context_parallel_shard_degree=1)
    c['tokenizer'].update(vae_path='${oc.env:WAN_VAE_PATH,/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}',encode_exact_durations=None)
    c['vlm_config']['tokenizer'].update(config_variant='hf',pretrained_model_name='${oc.env:TEXT_TOKENIZER_PATH,/mnt/checkpoints/Cosmos3-Nano/text_tokenizer}')
    c['vlm_config']['model_instance']['config']['base_config']['json_file'] = str(ROOT/'packages/cosmos3/cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json')
    c['diffusion_expert_config'].update(load_weights_from_pretrained=False,enable_fps_modulation=True,
        enable_action_state_embedding=False,enable_vision_condition_embedding=False)
    c['rectified_flow_training_config'].update(loss_scale=1.,normalize_loss_by_active=True,
        sample_level_loss_averaging=True,train_time_weight='uniform',use_discrete_rf=False)
    return LazyDict(dict(
        defaults=[{'override /data_train':None},{'override /data_val':None},
            {'override /model':'mot_fsdp'},{'override /optimizer':'adamw'},
            {'override /scheduler':'lambdacosine'},{'override /tokenizer':'wan2pt2_tokenizer'},
            {'override /sound_tokenizer':None},{'override /vlm_config':None},
            {'override /checkpoint':'local'},{'override /callbacks':['ar_it2v_basic','optimization']},
            {'override /ema':'power'},{'override /ckpt_type':'dcp'},'_self_'],
        job=dict(project='rbs_wam_ar_it2v',group='ar_it2v_v0_1',name='ar_it2v_v0_1_ego100h_cmd_stage1',wandb_mode='online'),
        model=L(ARIT2VModel)(config=ARIT2VModelConfig(**LazyDict(c,flags={'allow_objects':True})),_recursive_=False),
        optimizer=dict(betas=[.9,.95],eps=1e-6,fused=True,keys_to_select=['moe_gen','time_embedder','vae2llm','llm2vae'],
            lr=2e-5,lr_multipliers={},optimizer_type='AdamW',weight_decay=0.),
        scheduler=dict(lr_scheduler_type='LambdaCosine',warm_up_steps=[100],cycle_lengths=[3000],f_start=[0.],f_max=[1.],f_min=[.3],verbosity_interval=0),
        trainer=dict(distributed_parallelism='fsdp',grad_accum_iter=1,logging_iter=1,max_iter=3000,
            run_validation=False,run_validation_on_start=False,save_zero_checkpoint=False,seed=42,timeout_period=999999999,
            grad_scaler_args=dict(enabled=False),callbacks=dict(grad_clip=dict(clip_norm=1.,force_finite=False),
                manual_gc=dict(every_n=1,gc_level=2,warm_up=0),iter_speed=dict(every_n=10,save_s3=False),skip_nan_step=dict(max_consecutive_nan=1))),
        checkpoint=dict(load_path='${oc.env:BASE_CHECKPOINT_PATH,/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp}',
            load_training_state=False,strict_resume=True,keys_to_skip_loading=['net_ema.'],save_iter=500,
            broadcast_via_filesystem=True,dcp_async_mode_enabled=False,enable_gcs_patch_in_boto3=False,
            load_from_object_store=dict(enabled=False),save_to_object_store=dict(enabled=False)),
        dataloader_train=L(RecoverablePackingDataLoader)(audio_sample_rate=48000,dataset_name='egoverse_it2v',
            max_samples_per_batch=4,max_sequence_length=None,lazy_initialize_child_iterators=True,patch_spatial=2,
            sound_latent_fps=0,tokenizer_spatial_compression_factor=16,tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(batch_size=1,in_order=True,stateful=True,num_workers=3,
                persistent_workers=True,pin_memory=True,prefetch_factor=2,sampler=None,
                datasets=dict(video=dict(ratio=1,dataset=L(get_egoverse_it2v_dataset)(
                    episodes_manifest='/mnt/lzh/cosmos-EgoWAM/training_manifests/mecka_100h_v1_episodes.csv',
                    segments_manifest='/mnt/lzh/cosmos-EgoWAM/training_manifests/mecka_100h_v1_segments.csv',
                    tokenizer_config='${model.config.vlm_config.tokenizer}',split='train',seed=42,
                    frame_stride=1,clip_frame_tiers=(97,81,65,49,33,17),cfg_dropout_rate=.1))))),
        dataloader_val=None,upload_reproducible_setup=False),flags={'allow_objects':True})

ConfigStore.instance().store(group='experiment',package='_global_',name=CONFIG_NAME,node=experiment())


def full_segment_experiment(token_budget=65536):
    """Complete segments, native packing; optimizer remains unchanged.

    65536 is a validation budget, not permission to omit oversized segments.
    Dataset preflight must reject the full manifest until all samples fit an
    approved resource policy. V0.1 remains reproducible above.
    """
    if token_budget < 128 or token_budget % 128:
        raise ValueError('Token budget must be a positive multiple of 128')
    c=experiment()
    c.job.update(group='ar_it2v_v0_2',name='ar_it2v_v0_2_full_segments')
    c.model.config.max_num_tokens_after_packing=int(token_budget)
    c.dataloader_train.max_samples_per_batch=None
    c.dataloader_train.max_sequence_length=int(token_budget)
    data=c.dataloader_train.dataloader.datasets.video.dataset
    data.update(sample_mode='full_segment',random_window=False,
        max_sequence_length=int(token_budget),
        segment_statistics_path=str(ROOT/'outputs/maintenance/full_segment_audit_20261003/summary.json'))
    return c


ConfigStore.instance().store(group='experiment',package='_global_',
    name=FULL_SEGMENT_CONFIG_NAME,node=full_segment_experiment())

def make_config():
    from cosmos_framework.configs.base.config import make_config as base
    return base()
