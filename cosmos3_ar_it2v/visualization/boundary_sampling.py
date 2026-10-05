"""Training-free conditional boundary probe; the default AR sampler is unchanged.

The extra condition is the previous *predicted* last latent at its original
absolute time start-1, not a copy placed at start. Only the normal C4 new latents
are sampler variables. This is a new inference condition distribution: the
native AR attention sees one known boundary plus four new latents as its current
unit, so this probe is not claimed to match the training attention mask exactly.
"""
from __future__ import annotations

import math
from dataclasses import replace

import torch

from cosmos3_ar_it2v.inference import (
    _make_chunk_cache, cache_chunk_index, chunk_ranges, refresh_latents,
)
from cosmos_framework.model.generator.utils.memory import MemoryState


class BoundaryMemoryState(MemoryState):
    """Read-only native memory view excluding the boundary now present in pack."""

    def __init__(self, native_memory, boundary_tokens):
        if boundary_tokens < 1:
            raise ValueError("boundary_tokens must be positive")
        if native_memory.write_gen_cache:
            raise ValueError("boundary sampling must never write intermediate KV")
        self.native = native_memory
        self.boundary_tokens = boundary_tokens

    def init(self, hidden_states, device):
        return self.native.init(hidden_states, device)

    def read_for_layer(self, layer_idx):
        value = self.native.read_for_layer(layer_idx)
        if value.for_cuda_graphs or value.post_saturation_static_compile:
            raise ValueError("boundary probe requires eager dynamic memory")
        k, v = value.gen_k_hist, value.gen_v_hist
        if k is None or v is None or k.shape[1] != v.shape[1] or k.shape[1] < self.boundary_tokens:
            raise ValueError("history must contain the completed boundary exactly once")
        keep = k.shape[1] - self.boundary_tokens
        return replace(value, gen_k_hist=k[:, :keep] if keep else None,
                       gen_v_hist=v[:, :keep] if keep else None)

    def write_for_layer(self, layer_idx, kv_to_store):
        return self.native.write_for_layer(layer_idx, kv_to_store)

    def is_gen_only(self):
        return self.native.is_gen_only()

    def requires_natten_metadata(self):
        return self.native.requires_natten_metadata()


def rollout_boundary_chunks(first_latent, latent_frames, *, chunk_size, seed,
                            context_sigma, denoise, refresh, history_mode="gt",
                            gt_latents=None):
    """Keep C4 partition/RNG/output unchanged and share one completed boundary.

    The denoiser receives a cloned previous prediction only after the first
    target unit. GT is passed exclusively to completed-history refresh writes.
    No boundary is appended twice to the output or written at a second time.
    """
    ranges = chunk_ranges(latent_frames, chunk_size)
    if first_latent.ndim != 5 or first_latent.shape[0] != 1 or first_latent.shape[2] != 1:
        raise ValueError("first_latent must be [1,C,1,H,W]")
    if history_mode not in ("gt", "generated"):
        raise ValueError("history_mode must be gt or generated")
    if history_mode == "gt":
        shape = list(first_latent.shape)
        shape[2] = latent_frames
        if not isinstance(gt_latents, torch.Tensor) or list(gt_latents.shape) != shape:
            raise ValueError("GT requires the complete continuous matching latent segment")
        if gt_latents.dtype != first_latent.dtype or gt_latents.device != first_latent.device:
            raise ValueError("GT must match the first latent dtype/device")
        if not torch.equal(gt_latents[:, :, :1], first_latent):
            raise ValueError("GT and image condition differ")
    elif gt_latents is not None:
        raise ValueError("generated history must not receive GT latents")
    refresh_latents(first_latent, context_sigma, seed=seed)
    chunks = [first_latent.clone()]
    refresh(first_latent.clone(), start=0, sigma=0.)
    for start, end in ranges[1:]:
        shape = list(first_latent.shape)
        shape[2] = end - start
        generator = torch.Generator(device=first_latent.device).manual_seed(seed + start)
        noise = torch.randn(shape, generator=generator, device=first_latent.device,
                            dtype=first_latent.dtype)
        boundary = chunks[-1][:, :, -1:].clone() if start > 1 else None
        clean = denoise(noise, start=start, boundary=boundary)
        if clean.shape != noise.shape or not torch.isfinite(clean).all():
            raise ValueError("denoised chunk shape or finiteness mismatch")
        chunks.append(clean.clone())
        if end < latent_frames:
            history = gt_latents[:, :, start:end] if history_mode == "gt" else clean
            if not torch.isfinite(history).all():
                raise ValueError("nonfinite history chunk")
            refresh(refresh_latents(history, context_sigma, seed=seed + 100000 + start),
                    start=start, sigma=context_sigma)
    return torch.cat(chunks, dim=2)


def sample_known_boundary(model, noise, boundary, *, packed_sequences, memories,
                          guidance, num_steps, seed, start, latent_frames):
    """Use the official RF solver/CFG with only new latents as sampler state.

    The known boundary is reconstructed each callback, so it cannot be changed
    by numerical integration. Dropping its returned velocity is equivalent to
    the official conditioned velocity mask, without adding frozen variables to
    the solver state. New target noise and its seed exactly match baseline.
    """
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
    if start <= 1 or boundary.shape != noise[:, :, :1].shape:
        raise ValueError("boundary must be the prior prediction at start-1")
    if len(packed_sequences) != (1 if guidance == 1 else 2) or len(memories) != len(packed_sequences):
        raise ValueError("CFG packs and memory branches differ")

    def velocity_fn(state, timestep):
        def run_branch(pack, current, branch_timestep, branch):
            index = 0 if branch == "conditional" else 1
            value = torch.cat((boundary, current), dim=2)
            # Native setter routes the timestep only to noncondition patches.
            OmniMoTCausalModel._set_ar_vision_noise(model, pack, value, branch_timestep)
            pack.to_cuda()
            output = model.denoise(data_batch_packed=pack, memory=memories[index])
            velocity = torch.stack(output["preds_vision"])
            if velocity.shape != value.shape:
                raise ValueError("boundary forward returned an unexpected latent shape")
            return velocity[:, :, 1:]

        return OmniMoTCausalModel._predict_ar_velocity_with_cfg(
            model, noise_x=state, timestep=timestep, vision_shape=noise.shape,
            packed_seq=packed_sequences[0],
            packed_seq_uncond=packed_sequences[1] if guidance != 1 else None,
            guidance=guidance, normalize_cfg=False, run_branch=run_branch)

    result = OmniMoTCausalModel._run_ar_sampler(
        model, velocity_fn, noise.flatten(start_dim=1), sampler_mode="rf",
        num_steps=num_steps, shift=model.config.sigma_shift, seed=seed,
        sample_idx=start, num_frames=latent_frames, distilled_num_steps=None)
    return result.reshape(noise.shape)


@torch.no_grad()
def generate_boundary_latents(model, batch, *, num_steps=35, guidance=1., seed=42,
                              context_sigma=.02, history_mode="gt", return_reference=False):
    """Generate a full latent sequence; optionally return the same continuous GT.

    GT mode is an oracle-history diagnostic. Generated mode is autonomous I+T
    rollout and never uses the encoded GT beyond the conditioned first latent.
    No VAE decode/encode occurs at an AR boundary. Train weights remain intact.
    """
    from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive
    from cosmos_framework.data.generator.sequence_packing.modality import compute_text_split_length

    if num_steps < 1 or not math.isfinite(guidance):
        raise ValueError("invalid sampler arguments")
    if history_mode not in ("gt", "generated"):
        raise ValueError("history_mode must be gt or generated")
    if model.config.action_gen or model.config.compile.enabled:
        raise ValueError("boundary probe requires pure-video eager inference")
    if model.parallel_dims is not None and model.parallel_dims.cfgp_enabled:
        raise ValueError("boundary probe uses sequential CFG")
    clean = model.get_data_and_condition(batch, vision_condition_indexes=None)
    if clean.batch_size != 1 or clean.x0_tokens_action is not None or len(clean.x0_tokens_vision) != 1:
        raise ValueError("expected one pure-video continuous segment")
    reference = clean.x0_tokens_vision[0].to(**model.tensor_kwargs)
    frames = reference.shape[2]
    C, W = model.config.frames_per_chunk, model.config.local_attention_frames
    if C != 4 or W != 16:
        raise ValueError("this diagnostic preserves the trained C4/local16 recipe")
    cond, uncond = model._get_inference_text_tokens(batch, False)
    texts = [cond[0], uncond[0]] if guidance != 1 else [cond[0]]
    offsets = [compute_text_split_length(len(t), model.llm_special_tokens, has_generation=True) for t in texts]
    caches = [[_make_chunk_cache(C, W) for _ in range(model.net.num_hidden_layers)] for _ in texts]
    fps = clean.fps_vision.tolist()
    expert = model.config.diffusion_expert_config
    patch = expert.patch_spatial
    max_t = model.rectified_flow_video.noise_scheduler.config.num_train_timesteps
    spatial_tokens = math.ceil(reference.shape[3] / patch) * math.ceil(reference.shape[4] / patch)
    history_tokens = (W - C) * spatial_tokens

    def pack(value, start, sigma, branch, condition=False):
        packed = pack_input_sequence_autoregressive(
            vision_latent=value.to(**model.tensor_kwargs), action_latent=None,
            text_tokens=texts[branch] if start == 0 else None, timestep=float(sigma * max_t),
            fps_vision=fps, fps_action=None, special_tokens=model.llm_special_tokens,
            latent_patch_size=patch, condition_frame_indexes_vision=[0] if condition else [],
            condition_frame_indexes_action=[], frame_idx=start,
            temporal_compression_factor=model.tokenizer_vision_gen.temporal_compression_factor,
            video_temporal_causal=True, action_dim=model.config.max_action_dim,
            enable_fps_modulation=expert.enable_fps_modulation, base_fps=expert.base_fps,
            cached_text_offset=None if start == 0 else offsets[branch],
            unified_3d_mrope_temporal_modality_margin=expert.unified_3d_mrope_temporal_modality_margin,
            force_action_tokens=False)
        if packed.action is not None:
            raise ValueError("unexpected action tokens")
        packed.to_cuda()
        return packed

    def memory(packed, start, branch, *, write=False):
        return model.build_memory_state(packed, dict(
            dual_kv_cache=caches[branch], frame_idx=cache_chunk_index(start, C),
            write_gen_cache=write, use_ar_rolling=False, transfer_history_sink_tokens=0,
            transfer_history_max_tokens=history_tokens))

    def refresh(value, *, start, sigma):
        for branch in range(len(texts)):
            packed = pack(value, start, sigma, branch, condition=start == 0)
            model.denoise(data_batch_packed=packed, memory=memory(packed, start, branch, write=True))

    def denoise(noise, *, start, boundary):
        if boundary is None:
            # First target chunk has exactly the baseline route and input.
            return model.generate_next_frame(
                packed_seq=pack(noise, start, 1., 0),
                packed_seq_uncond=pack(noise, start, 1., 1) if guidance != 1 else None,
                curr_vision_latent=noise, curr_action_latent=None, cond_text_tokens=texts[0],
                uncond_text_tokens=texts[1] if guidance != 1 else [], gen_data_clean=clean,
                dual_kv_cache=caches[0], dual_kv_cache_uncond=caches[1] if guidance != 1 else None,
                frame_idx=start, cache_frame_idx=cache_chunk_index(start, C), num_frames=frames,
                guidance=guidance, num_steps=num_steps, shift=model.config.sigma_shift,
                seed=seed, fps_vision_list=fps, fps_action_list=[], use_ar_rolling_path=False,
                transfer_history_sink_tokens=0, transfer_history_max_tokens=history_tokens)
        packs = [pack(torch.cat((boundary, noise), dim=2), start - 1, 1., branch, condition=True)
                 for branch in range(len(texts))]
        memories = [BoundaryMemoryState(memory(packed, start, branch), spatial_tokens)
                    for branch, packed in enumerate(packs)]
        return sample_known_boundary(model, noise, boundary, packed_sequences=packs,
                                     memories=memories, guidance=guidance, num_steps=num_steps,
                                     seed=seed, start=start, latent_frames=frames)

    predicted = rollout_boundary_chunks(
        reference[:, :, :1], frames, chunk_size=C, seed=seed, context_sigma=context_sigma,
        denoise=denoise, refresh=refresh, history_mode=history_mode,
        gt_latents=reference if history_mode == "gt" else None)
    return (predicted, reference) if return_reference else predicted
