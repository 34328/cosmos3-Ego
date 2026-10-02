"""Cosmos3 pure-video IT2V causal adaptation (CMD Stage 1 core recipe)."""
from __future__ import annotations
from itertools import accumulate
import attrs
import torch
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel, OmniMoTCausalModelConfig
from cosmos_framework.model.generator.utils.kv_cache import DualKVCache, TeacherForcingMemoryState
from .attention import ChunkCausalAttention, chunk_ids


@attrs.define(slots=False)
class ARIT2VModelConfig(OmniMoTCausalModelConfig):
    frames_per_chunk: int = 4
    local_attention_frames: int = 16
    sigma_min: float = .02
    sigma_max: float = .98
    sigma_shift: float = 5.0


def sample_chunk_sigmas(frame_counts, *, chunk_size=4, shift=5.0, sigma_min=.02, sigma_max=.98, device='cpu', generator=None):
    """Uniform base time -> rational shift -> post-shift clamp; clean first frame.

    All frames in a future chunk share one draw; all sample/chunk draws are
    independent. Padding in ragged batches is zero and never reaches the pack.
    """
    if not frame_counts or min(frame_counts) < 2 or chunk_size < 1:
        raise ValueError("IT2V needs at least one future frame and a positive chunk size")
    if not (0 <= sigma_min < sigma_max <= 1 and shift > 0):
        raise ValueError("invalid sigma support/shift")
    result = torch.zeros(len(frame_counts), max(frame_counts), device=device, dtype=torch.float32)
    for i,t in enumerate(frame_counts):
        ids = chunk_ids(torch.arange(t, device=device),chunk_size)
        n = (t-2)//chunk_size+1
        u = torch.rand(n,device=device,generator=generator,dtype=torch.float32)
        sg = (shift*u/(1+(shift-1)*u)).clamp(sigma_min,sigma_max)
        result[i,1:t] = sg[ids[1:]-1]
    return result


class ChunkDFMemory(TeacherForcingMemoryState):
    """Official three-way metadata adapter; one differentiable stream, no replay."""
    def __init__(self, *, text_lengths, device, local_attention_frames, **kwargs):
        super().__init__(**kwargs)
        self.text_lengths = tuple(text_lengths)
        self.attention = ChunkCausalAttention(self.vision_token_shapes, text_lengths, device=device,
            frames_per_chunk=self.frames_per_chunk, local_attention_frames=local_attention_frames)

    def init(self, hidden_states, device):
        super().init(hidden_states,device)
        if self.new_und_len != sum(self.text_lengths):
            raise ValueError("Packed caption lengths do not match actual text tokens")
        self.und_kv_offsets = torch.tensor([0,*accumulate(self.text_lengths)],device=device,dtype=torch.int32)

    def read_for_layer(self, layer_idx):
        value = self._read_teacher_forcing_base_value(layer_idx)
        value.uses_rolling_gen_cache = False
        value.gen_attention_override = self.attention
        return value

    def write_for_layer(self, layer_idx, kv_to_store):
        # History K/V are in the same graph; never detach or replay a clean pass.
        return None


class ARIT2VModel(OmniMoTCausalModel):
    def __init__(self, config):
        if config.action_gen or config.sound_gen or getattr(config,'lidar_gen',False) or not config.vision_gen:
            raise ValueError("AR IT2V is strictly vision-only; action/audio/lidar must be disabled")
        if not config.video_temporal_causal or config.causal_training_strategy != 'diffusion_forcing' or config.joint_attn_implementation != 'three_way':
            raise ValueError("AR IT2V requires temporal-causal three-way diffusion forcing")
        if config.parallelism.context_parallel_shard_degree != 1 or config.enable_moba:
            raise ValueError("AR IT2V currently requires CP1 without MoBA")
        if config.frames_per_chunk < 1 or config.local_attention_frames < config.frames_per_chunk:
            raise ValueError("invalid chunk/local-window configuration")
        if getattr(config, 'lidar_state_ch', None) is not None:
            raise ValueError('Pure-video IT2V must not instantiate lidar projections')
        if not config.rectified_flow_training_config.normalize_loss_by_active:
            raise ValueError('IT2V flow loss must exclude condition frames from its denominator')
        if config.rectified_flow_training_config.train_time_weight != 'uniform':
            raise ValueError("CMD Stage1 requires uniform flow loss weighting")
        super().__init__(config)

    def memory_init_training(self, gen_data_clean, data_batch, input_text_indexes):
        for name in ('x0_tokens_action','x0_tokens_sound','x0_tokens_lidar'):
            value = getattr(gen_data_clean,name,None)
            if value is not None and len(value):
                raise ValueError(f"Pure-video data unexpectedly contains {name}")
        if gen_data_clean.is_image_batch:
            raise ValueError("AR IT2V training requires video clips")
        return super().memory_init_training(gen_data_clean,data_batch,input_text_indexes)

    def pre_noise_memory_hook(self, packed_sequence, gen_data_clean, memory_info):
        if any(getattr(packed_sequence,name,None) is not None for name in ('action','sound','lidar')):
            raise ValueError("Pure-video pack cannot contain non-video generation modalities")
        for mask in packed_sequence.vision.condition_mask:
            expected = torch.zeros_like(mask)
            expected[0] = 1
            if not torch.equal(mask,expected):
                raise ValueError("IT2V requires exactly latent frame zero as the clean condition")
        if '_tf_memory_state' in memory_info:
            raise ValueError("DF has no teacher-forcing replay pass")
        return memory_info

    def build_memory_state(self, packed_seq, memory_info):
        if memory_info.get('dual_kv_cache') is not None:
            return super().build_memory_state(packed_seq,memory_info)
        text_lengths = [int(n) for n,mode in zip(packed_seq.split_lens,packed_seq.attn_modes) if mode == 'causal']
        net = self.net
        return ChunkDFMemory(text_lengths=text_lengths,device=packed_seq.vision.tokens[0].device,
            local_attention_frames=self.config.local_attention_frames,
            vision_token_shapes=packed_seq.vision.token_shapes,num_action_tokens_per_supertoken=0,
            null_action_supertokens=False,segment_idx=0,
            dual_kv_cache=[DualKVCache(gen_cache_size=2) for _ in range(net.num_hidden_layers)],
            num_kv_heads=net.num_kv_heads,head_dim=net.head_dim,detach_clean_kv=False,
            clamp_empty_varlen_kv=self.config.clamp_empty_varlen_kv,frames_per_chunk=self.config.frames_per_chunk)

    def _get_train_noise_level_vision(self,batch_size,is_image_batch,num_vision_latent_frames,
                                    resolutions=None,num_tokens=None,iteration=None):
        if is_image_batch or len(num_vision_latent_frames) != batch_size:
            raise ValueError("AR IT2V expects one latent video length per sample")
        generator = None
        if iteration is not None and torch.are_deterministic_algorithms_enabled():
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            generator = torch.Generator(device=self.tensor_kwargs_fp32['device']).manual_seed(iteration*65536+rank)
        cfg = self.config
        sigmas = sample_chunk_sigmas(num_vision_latent_frames,chunk_size=cfg.frames_per_chunk,
            shift=cfg.sigma_shift,sigma_min=cfg.sigma_min,sigma_max=cfg.sigma_max,
            device=self.tensor_kwargs_fp32['device'],generator=generator)
        return sigmas*float(self.rectified_flow_video.noise_scheduler.config.num_train_timesteps),sigmas

