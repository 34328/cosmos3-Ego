"""Single-pass parallel TF; native VAE, P noising, transformer and flow loss."""
from __future__ import annotations

import copy

import attrs
import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

from .attention_v04 import ParallelTeacherForcingAttention
from .model import ARIT2VModel, ARIT2VModelConfig
from .model_v03 import sample_lingbot_public_history_sigmas


@attrs.define(slots=False)
class ARIT2VModelV04Config(ARIT2VModelConfig):
    prediction_noise_mode: str = "cosmos3_native"
    history_noise_mode: str = "lingbot_public"
    clean_history_probability: float = .5
    target_history_seed: int = 42
    loss_scope: str = "all_prediction_chunks_history_copy_unsupervised"


def history_generator(*, iteration, rank, seed, device="cpu", epsilon=False):
    """Recoverable H stream independent of native P sigma/epsilon RNG state."""
    if iteration < 0 or rank < 0:
        raise ValueError("iteration and rank must be nonnegative")
    mixed = (int(seed) + 0x4152495432563034) ^ (int(iteration)*1_000_003) ^ (int(rank)*9_176_293)
    if epsilon:
        mixed ^= 0x455053494C4F4E
    return torch.Generator(device=device).manual_seed(mixed & ((1 << 63)-1))


def sample_history_sigmas(frame_counts, *, generator, clean_history_probability=.5):
    counts = tuple(map(int, frame_counts))
    if not counts or min(counts) < 2 or not 0 <= clean_history_probability <= 1:
        raise ValueError("invalid parallel TF lengths or clean-history probability")
    if generator.device.type != "cpu":
        raise ValueError("history schedule requires its independent CPU RNG")
    sigmas = torch.zeros(len(counts), max(counts), dtype=torch.float32)
    clean_rows = []
    for i, t in enumerate(counts):
        clean = bool(torch.rand((), generator=generator) < clean_history_probability)
        values, _ = sample_lingbot_public_history_sigmas(t-1, generator=generator)
        if not clean:
            sigmas[i, 1:t] = values
        clean_rows.append(clean)
    return sigmas, tuple(clean_rows)


def build_parallel_tf_pack(source, history_tokens, history_timesteps):
    """Extend each native [text,P] sample to [text,H,P], without repacking text.

    Both streams retain native first-frame conditioning and the same absolute
    spatial/temporal positions. H's other frames remain ordinary video tokens,
    including when sigma=0. The original pack is kept intact for native loss.
    """
    vision = source.vision
    n = len(vision.token_shapes)
    if len(history_tokens) != n or history_timesteps.shape[0] != n:
        raise ValueError("history payload must match every packed segment")
    if source.attn_modes != [mode for _ in range(n) for mode in ("causal", "full")]:
        raise ValueError("parallel TF requires one caption and video split per sample")
    out = copy.copy(source)
    out.vision = copy.copy(vision)
    out._sequence_pack_metadata = None
    out.sample_lens, out.split_lens = [], []
    text_indexes, video_indexes, positions, mse_indexes, timesteps = [], [], [], [], []
    tokens, masks, noisy_frames, shapes, item_lens = [], [], [], [], []
    device = source.position_ids.device
    old_cursor, new_cursor, time_cursor = 0, 0, 0
    for i, (t, h, w) in enumerate(vision.token_shapes):
        count = t*h*w
        text_count, gen_count = source.split_lens[2*i:2*i+2]
        if gen_count != count or source.sample_lens[i] != text_count+count:
            raise ValueError("unexpected native pure-video packed geometry")
        if history_tokens[i].shape != vision.tokens[i].shape:
            raise ValueError("H and P must have identical latent geometry")
        expected_frames = torch.arange(1, t, device=vision.noisy_frame_indexes[i].device)
        if not torch.equal(vision.noisy_frame_indexes[i], expected_frames):
            raise ValueError("parallel TF requires exactly frame zero as condition")
        text_indexes.append(torch.arange(new_cursor, new_cursor+text_count, device=device))
        begin = new_cursor+text_count
        video_indexes.append(torch.arange(begin, begin+2*count, device=device))
        original = source.position_ids[:, old_cursor:old_cursor+text_count+count]
        positions.extend((original[:, :text_count], original[:, text_count:], original[:, text_count:]))
        # These indexes also drive native timestep embedding. H receives no
        # direct loss because its predictions never reach the native loss call.
        mse_indexes.extend((torch.arange(begin+h*w, begin+count, device=device),
                            torch.arange(begin+count+h*w, begin+2*count, device=device)))
        per_stream = (t-1)*h*w
        timesteps.extend((history_timesteps[i, 1:t].to(device).repeat_interleave(h*w),
                          vision.timesteps[time_cursor:time_cursor+per_stream]))
        tokens.append(torch.cat((history_tokens[i], vision.tokens[i]), dim=2))
        masks.append(torch.cat((vision.condition_mask[i], vision.condition_mask[i]), dim=0))
        noisy_frames.append(torch.cat((expected_frames, expected_frames+t)))
        shapes.append((2*t, h, w))
        item_lens.append([2*count])
        out.sample_lens.append(text_count+2*count)
        out.split_lens.extend((text_count, 2*count))
        old_cursor += text_count+count
        new_cursor += text_count+2*count
        time_cursor += per_stream
    if old_cursor != source.sequence_length or time_cursor != vision.timesteps.numel():
        raise ValueError("native sequence/timestep metadata contains unexpected rows")
    out.sequence_length = new_cursor
    out.text_indexes = torch.cat(text_indexes)
    # Native video packs also carry next-token text labels, even when the
    # model disables its text prediction head. Keep their original row order,
    # labels and weights, remapping only the positions after H is inserted.
    if source.ce_loss_indexes is not None and source.ce_loss_indexes.numel():
        text_rows = torch.searchsorted(source.text_indexes, source.ce_loss_indexes)
        if (torch.any(text_rows >= source.text_indexes.numel())
                or not torch.equal(source.text_indexes[text_rows], source.ce_loss_indexes)):
            raise ValueError("native text loss indexes must reference text tokens")
        out.ce_loss_indexes = out.text_indexes[text_rows]
    out.position_ids = torch.cat(positions, dim=1)
    out.uses_single_timestep = False
    out.vision.sequence_indexes = torch.cat(video_indexes)
    out.vision.mse_loss_indexes = torch.cat(mse_indexes)
    out.vision.timesteps = torch.cat(timesteps)
    out.vision.tokens, out.vision.condition_mask = tokens, masks
    out.vision.noisy_frame_indexes, out.vision.token_shapes = noisy_frames, shapes
    out.vision_item_split_lens = item_lens
    if source.vision_condition_type_mask is not None:
        parts = torch.split(source.vision_condition_type_mask,
                            [t*h*w for t, h, w in vision.token_shapes])
        out.vision_condition_type_mask = torch.cat([p.repeat(2) for p in parts])
    out.prepare_sequence_pack_metadata()
    return out


class ARIT2VModelV04(ARIT2VModel):
    def __init__(self, config):
        if config.history_noise_mode != "lingbot_public":
            raise ValueError("V0.4 requires LingBot's published history schedule")
        if not 0 <= config.clean_history_probability <= 1:
            raise ValueError("invalid V0.4 clean-history probability")
        super().__init__(config)

    def _get_train_noise_level_vision(self, batch_size, is_image_batch,
                                     num_vision_latent_frames, resolutions=None,
                                     num_tokens=None, iteration=None):
        """One official Cosmos draw per P chunk; repeat over that chunk's frames.

        The official sampler owns the distribution, resolution/token-based
        shift, timestep scale and RNG. Only its sampling geometry is changed
        from individual frames to [1,C,C,...] non-condition chunks. There is
        no project-specific shift or clamp in this prediction path.
        """
        counts = tuple(map(int, num_vision_latent_frames))
        chunk_size = self.config.frames_per_chunk
        if is_image_batch or len(counts) != batch_size or not counts or min(counts) < 2:
            raise ValueError("parallel TF expects one video with future frames per sample")
        if chunk_size < 1:
            raise ValueError("parallel TF requires a positive chunk size")
        chunk_counts = [(t-2)//chunk_size+1 for t in counts]
        chunk_timesteps, chunk_sigmas = OmniMoTModel._get_train_noise_level_vision(
            self, batch_size=batch_size, is_image_batch=False,
            num_vision_latent_frames=chunk_counts, resolutions=resolutions,
            num_tokens=num_tokens, iteration=iteration)
        timesteps = chunk_timesteps.new_zeros((batch_size, max(counts)))
        sigmas = chunk_sigmas.new_zeros((batch_size, max(counts)))
        for i, (t, n) in enumerate(zip(counts, chunk_counts, strict=True)):
            timesteps[i, 1:t] = chunk_timesteps[i, :n].repeat_interleave(chunk_size)[:t-1]
            sigmas[i, 1:t] = chunk_sigmas[i, :n].repeat_interleave(chunk_size)[:t-1]
        return timesteps, sigmas

    def _add_noise_to_input(self, gen_data_clean, packed_sequence, sigmas, *,
                            iteration=None, **kwargs):
        if iteration is None:
            raise ValueError("parallel TF history requires a recoverable training iteration")
        # Complete native P noising first, preserving the exact original RNG draws.
        noised = super()._add_noise_to_input(gen_data_clean, packed_sequence,
                                           sigmas, iteration=iteration, **kwargs)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        clean = gen_data_clean.x0_tokens_vision
        counts = [x.shape[2] for x in clean]
        seed = self.config.target_history_seed
        history_sigmas, clean_rows = sample_history_sigmas(counts,
            generator=history_generator(iteration=iteration, rank=rank, seed=seed),
            clean_history_probability=self.config.clean_history_probability)
        epsilon_rng = history_generator(iteration=iteration, rank=rank, seed=seed,
                                        device=self.tensor_kwargs_fp32["device"], epsilon=True)
        epsilon = [torch.randn(x.shape, generator=epsilon_rng, **self.tensor_kwargs_fp32) for x in clean]
        sigma_rows = [history_sigmas[i, :t].to(**self.tensor_kwargs_fp32).reshape(t, 1, 1)
                      for i, t in enumerate(counts)]
        history, _ = self.rectified_flow_video.get_interpolation(epsilon, clean, sigma_rows)
        packed_sequence.it2v_v04_history = [x.to(**self.tensor_kwargs) for x in history]
        packed_sequence.it2v_v04_history_timesteps = history_sigmas * float(
            self.rectified_flow_video.noise_scheduler.config.num_train_timesteps)
        packed_sequence.it2v_v04_clean_history = clean_rows
        return noised

    def build_memory_state(self, packed_seq, memory_info):
        if not hasattr(packed_seq, "it2v_v04_history"):
            return super().build_memory_state(packed_seq, memory_info)
        if memory_info.get("dual_kv_cache") is not None:
            raise ValueError("parallel TF training cannot consume an inference KV cache")
        parallel = build_parallel_tf_pack(packed_seq, packed_seq.it2v_v04_history,
                                          packed_seq.it2v_v04_history_timesteps)
        memory = super().build_memory_state(parallel, memory_info)
        text_lengths = packed_seq.split_lens[::2]
        memory.attention = ParallelTeacherForcingAttention(packed_seq.vision.token_shapes,
            text_lengths, device=packed_seq.vision.tokens[0].device,
            frames_per_chunk=self.config.frames_per_chunk,
            local_attention_frames=self.config.local_attention_frames)
        packed_seq.it2v_v04_parallel_pack = parallel
        return memory

    def denoise(self, net=None, data_batch_packed=None, memory=None, video_temporal_causal=None):
        parallel = getattr(data_batch_packed, "it2v_v04_parallel_pack", None)
        if parallel is None:
            return super().denoise(net=net, data_batch_packed=data_batch_packed,
                memory=memory, video_temporal_causal=video_temporal_causal)
        out = super().denoise(net=net, data_batch_packed=parallel,
                             memory=memory, video_temporal_causal=video_temporal_causal)
        out["preds_vision"] = [p[:, :, t:] for p, (t, _, _) in
                              zip(out["preds_vision"], data_batch_packed.vision.token_shapes, strict=True)]
        self.v04_source_tokens = data_batch_packed.sequence_length
        self.v04_transformer_tokens = parallel.sequence_length
        return out

    def training_step(self, data_batch, iteration):
        out, loss = super().training_step(data_batch, iteration)
        out["v04_source_tokens"] = self.v04_source_tokens
        out["v04_transformer_tokens"] = self.v04_transformer_tokens
        return out, loss
