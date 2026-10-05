"""Pure-video chunk AR inference using the official Cosmos sampler and KV storage.

The first latent is clean. A generated C-frame chunk is refreshed in ONE forward
so deeper-layer keys preserve within-chunk bidirectional dependencies. Position
indexes remain absolute even when the attention history is cropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def cache_chunk_index(start: int, chunk_size: int = 4):
    if start < 0 or (start > 0 and (start - 1) % chunk_size):
        raise ValueError("cache writes must start at a complete chunk boundary")
    return 0 if start == 0 else 1 + (start - 1) // chunk_size


def chunk_ranges(latent_frames: int, chunk_size: int = 4):
    if chunk_size < 1 or latent_frames < 1:
        raise ValueError("latent video and chunk size must be positive")
    return [(0, 1)] + [(i, min(i + chunk_size, latent_frames)) for i in range(1, latent_frames, chunk_size)]


def _make_chunk_cache(chunk_size, local_frames):
    from cosmos_framework.model.generator.utils.kv_cache import DualKVCache
    # Native capacity includes the current write slot: only capacity-1 entries
    # are fetched as history. Entries are whole chunks, except the first image.
    capacity = max(2, math.ceil((local_frames - chunk_size) / chunk_size) + 1)
    return DualKVCache(gen_cache_size=capacity, preallocate_ring=False)


def refresh_latents(clean, sigma: float, *, seed: int):
    """Refresh a history copy; never modify generated output or consume global RNG."""
    if not math.isfinite(sigma) or not 0 <= sigma <= 1:
        raise ValueError("context sigma must be finite and in [0, 1]")
    if sigma == 0:
        return clean.clone()
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(clean.shape, device=clean.device, dtype=torch.float32, generator=generator)
    return ((1 - sigma) * clean.float() + sigma * noise).to(clean.dtype)


def rollout_chunks(first_latent, latent_frames, *, chunk_size, seed, context_sigma, denoise, refresh,
                   history_mode="generated", gt_latents=None):
    """Generate outputs, then refresh only completed chunks as causal history.

    GT history is a diagnostic: ``gt_latents`` must be one continuously encoded
    segment, not independently encoded chunks. Its completed ``start:end`` slice
    replaces the history write only; outputs always remain model predictions.
    Callbacks receive absolute latent positions and never receive future GT.
    """
    ranges = chunk_ranges(latent_frames, chunk_size)
    if first_latent.ndim != 5 or first_latent.shape[0] != 1 or first_latent.shape[2] != 1:
        raise ValueError("first_latent must be [1,C,1,H,W]")
    if history_mode not in ("generated", "gt"):
        raise ValueError("history_mode must be 'generated' or 'gt'")
    if history_mode == "gt":
        expected = list(first_latent.shape)
        expected[2] = latent_frames
        if not isinstance(gt_latents, torch.Tensor) or list(gt_latents.shape) != expected:
            raise ValueError("GT history requires the complete matching [1,C,T,H,W] latent segment")
        if gt_latents.device != first_latent.device or gt_latents.dtype != first_latent.dtype:
            raise ValueError("GT history must match the first latent device and dtype")
        if not torch.equal(gt_latents[:, :, :1], first_latent):
            raise ValueError("GT history must have the same conditioned first latent")
    elif gt_latents is not None:
        raise ValueError("gt_latents is only accepted with history_mode='gt'")
    # Validate even for a one-frame-only call.
    refresh_latents(first_latent, context_sigma, seed=seed)
    chunks = [first_latent.clone()]
    refresh(first_latent.clone(), start=0, sigma=0.0)
    for start, end in ranges[1:]:
        generator = torch.Generator(device=first_latent.device).manual_seed(seed + start)
        shape = list(first_latent.shape)
        shape[2] = end - start
        noise = torch.randn(shape, generator=generator, device=first_latent.device, dtype=first_latent.dtype)
        clean = denoise(noise, start=start)
        if clean.shape != noise.shape or not torch.isfinite(clean).all():
            raise ValueError("denoised chunk shape or finiteness mismatch")
        chunks.append(clean.clone())
        if end < latent_frames:
            history_source = gt_latents[:, :, start:end] if history_mode == "gt" else clean
            if not torch.isfinite(history_source).all():
                raise ValueError("nonfinite history chunk")
            history = refresh_latents(history_source, context_sigma, seed=seed + 100000 + start)
            refresh(history, start=start, sigma=context_sigma)
    return torch.cat(chunks, dim=2)


@torch.no_grad()
def generate_latents(model, batch, *, num_steps=35, guidance=1.0, seed=42, context_sigma=0.02,
                     history_mode="generated"):
    """Pure I+T rollout, optionally replacing completed history with continuous GT.

    ``gt`` is an oracle-history diagnostic, not autonomous I+T generation.
    The conditioned first latent and all prediction outputs retain their meaning.
    """
    from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive
    from cosmos_framework.data.generator.sequence_packing.modality import compute_text_split_length

    if num_steps < 1 or not math.isfinite(guidance):
        raise ValueError("invalid sampler arguments")
    if history_mode not in ("generated", "gt"):
        raise ValueError("history_mode must be 'generated' or 'gt'")
    if model.config.action_gen:
        raise ValueError("IT2V requires action_gen=False")
    if model.config.compile.enabled:
        raise ValueError("chunk AR preview currently requires eager inference")
    if model.parallel_dims is not None and model.parallel_dims.cfgp_enabled:
        raise ValueError("preview uses sequential CFG, not CFG parallelism")
    clean = model.get_data_and_condition(batch, vision_condition_indexes=[[0]])
    if clean.batch_size != 1 or clean.x0_tokens_action is not None or len(clean.x0_tokens_vision) != 1:
        raise ValueError("expected one pure-video sample, without action tokens")
    reference = clean.x0_tokens_vision[0].to(**model.tensor_kwargs)
    frames = reference.shape[2]
    chunk_size = model.config.frames_per_chunk
    local_frames = model.config.local_attention_frames
    chunk_ranges(frames, chunk_size)
    if local_frames < chunk_size:
        raise ValueError("local attention window must include the current chunk")
    cond, uncond = model._get_inference_text_tokens(batch, False)
    texts = [cond[0], uncond[0]] if guidance != 1 else [cond[0]]
    offsets = [compute_text_split_length(len(t), model.llm_special_tokens, has_generation=True) for t in texts]
    caches = [[_make_chunk_cache(chunk_size, local_frames)
               for _ in range(model.net.num_hidden_layers)] for _ in texts]
    fps = clean.fps_vision.tolist()
    tcf = model.tokenizer_vision_gen.temporal_compression_factor
    expert = model.config.diffusion_expert_config
    patch = expert.patch_spatial
    max_t = model.rectified_flow_video.noise_scheduler.config.num_train_timesteps
    patches_per_frame = math.ceil(reference.shape[3] / patch) * math.ceil(reference.shape[4] / patch)
    history_tokens = (local_frames - chunk_size) * patches_per_frame

    def pack(value, start, sigma, branch, condition=False):
        out = pack_input_sequence_autoregressive(
            vision_latent=value.to(**model.tensor_kwargs), action_latent=None,
            text_tokens=texts[branch] if start == 0 else None,
            timestep=float(sigma * max_t), fps_vision=fps, fps_action=None,
            special_tokens=model.llm_special_tokens, latent_patch_size=patch,
            condition_frame_indexes_vision=[0] if condition else [],
            condition_frame_indexes_action=[], frame_idx=start,
            temporal_compression_factor=tcf, video_temporal_causal=True,
            action_dim=model.config.max_action_dim,
            enable_fps_modulation=expert.enable_fps_modulation, base_fps=expert.base_fps,
            cached_text_offset=None if start == 0 else offsets[branch],
            unified_3d_mrope_temporal_modality_margin=expert.unified_3d_mrope_temporal_modality_margin,
            force_action_tokens=False,
        )
        if out.action is not None:
            raise AssertionError("pure-video pack unexpectedly contains actions")
        out.to_cuda()
        return out

    def refresh(value, *, start, sigma):
        for branch in range(len(texts)):
            packed = pack(value, start, sigma, branch, condition=start == 0)
            memory = model.build_memory_state(packed, dict(
                dual_kv_cache=caches[branch], frame_idx=cache_chunk_index(start, chunk_size), write_gen_cache=True,
                use_ar_rolling=False, transfer_history_sink_tokens=0,
                transfer_history_max_tokens=history_tokens))
            model.denoise(data_batch_packed=packed, memory=memory)

    def denoise(noise, *, start):
        return model.generate_next_frame(
            packed_seq=pack(noise, start, 1., 0),
            packed_seq_uncond=pack(noise, start, 1., 1) if guidance != 1 else None,
            curr_vision_latent=noise, curr_action_latent=None,
            cond_text_tokens=texts[0], uncond_text_tokens=texts[1] if guidance != 1 else [],
            gen_data_clean=clean, dual_kv_cache=caches[0],
            dual_kv_cache_uncond=caches[1] if guidance != 1 else None,
            frame_idx=start, cache_frame_idx=cache_chunk_index(start, chunk_size), num_frames=frames,
            guidance=guidance, num_steps=num_steps, shift=model.config.sigma_shift,
            seed=seed, fps_vision_list=fps, fps_action_list=[],
            use_ar_rolling_path=False, transfer_history_sink_tokens=0,
            transfer_history_max_tokens=history_tokens)

    return rollout_chunks(reference[:, :, :1], frames, chunk_size=chunk_size,
                          seed=seed, context_sigma=context_sigma, denoise=denoise, refresh=refresh,
                          history_mode=history_mode, gt_latents=reference if history_mode == "gt" else None)


def training_layout_batch(sample):
    """Use the official packing collator, including ragged captions/vision."""
    from collections import deque
    from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn
    packer = object.__new__(PackingDataLoader)
    packer.buffers = [deque()]
    packer.dataloaders = [iter([custom_collate_fn([sample])])]
    output = {}
    PackingDataLoader._update_output_batch(packer, output, PackingDataLoader._get_next_sample(packer, 0))
    return output


def load_model(toml, checkpoint):
    from cosmos3_ar_it2v import config as _registration  # noqa: F401
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.utils.generator.model_loader import _load_model
    from cosmos_framework.model.generator.utils.safetensors_loader import load_vfm_model
    config = load_experiment_from_toml(toml, [
        "job.wandb_mode=disabled",
        "model.config.parallelism.data_parallel_shard_degree=1",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "model.config.parallelism.context_parallel_shard_degree=1",
        "model.config.parallelism.enable_inference_mode=true",
        "model.config.activation_checkpointing.mode=none",
        "model.config.compile.enabled=false",
    ])
    config.validate()
    config.freeze()
    model = instantiate(config.model).cuda()
    model.on_train_start()
    checkpoint = Path(checkpoint).resolve()
    if list(checkpoint.glob("*.safetensors")):
        load_vfm_model(model.net, str(checkpoint), credential_path=None,
                       parallel_dims=model.parallel_dims)
    else:
        if not (checkpoint / ".metadata").is_file():
            raise ValueError("checkpoint must be a safetensors directory or completed DCP model directory")
        _load_model(model, str(checkpoint), credential_path=None, keys_to_skip_loading=[])
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes-manifest", required=True)
    parser.add_argument("--segments-manifest", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--context-sigma", type=float, default=.02)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("inference needs an allocated idle GPU")
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.lazy_config import instantiate
    import imageio.v2 as imageio
    import numpy as np
    distributed.init()
    try:
        model, config = load_model(args.toml, args.checkpoint)
        dataset_configs = config.dataloader_train.dataloader.datasets
        if len(dataset_configs) != 1:
            raise ValueError("preview expects exactly one IT2V dataset")
        ds_cfg = next(iter(dataset_configs.values())).dataset
        dataset = instantiate(ds_cfg, episodes_manifest=args.episodes_manifest,
                              segments_manifest=args.segments_manifest, split=args.split,
                              iterable_shuffle=False, random_window=False, cfg_dropout_rate=0.0)
        sample = dataset.get_item_at_window(args.row_index, window_start=args.start_frame)
        latents = generate_latents(model, training_layout_batch(sample), num_steps=args.steps,
                                   guidance=args.guidance, seed=args.seed, context_sigma=args.context_sigma)
        with torch.no_grad():
            decoded = model.decode(latents.to(**model.tensor_kwargs))
        if not torch.isfinite(decoded).all():
            raise ValueError("nonfinite decoded RGB")
        pred = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        gt = sample["video"].permute(1, 2, 3, 0).cpu().numpy()
        if pred.shape != gt.shape:
            raise ValueError(f"decoded/GT shape mismatch: {pred.shape} vs {gt.shape}")
        true_frames = int(sample.get('video_true_num_frames', len(gt)))
        pred, gt = pred[:true_frames], gt[:true_frames]
        fps = float(sample["conditioning_fps"])
        # Remove only deterministic bottom padding, preserving original motion.
        pred, gt = pred[:, :360], gt[:, :360]
        for name, video in (("generated", pred), ("gt", gt), ("preview", np.concatenate([gt, pred], axis=2))):
            imageio.mimwrite(str(output / (name + ".mp4")), video, fps=fps, codec="libx264", macro_block_size=1)
        np.savez_compressed(output / "rollout.npz", generated=pred, gt=gt,
                            source_frame_indices=sample["source_frame_indices"].numpy())
        metadata = dict(vars(args), checkpoint=str(Path(args.checkpoint).resolve()),
                        sample_id=sample["sample_id"], caption=sample["ai_caption"],
                        source_frame_indices=sample["source_frame_indices"].tolist(),
                        frames=true_frames,
                        video_true_num_frames=true_frames,
                        video_temporal_padding=int(sample.get('video_temporal_padding', 0)),
                        latent_frames=latents.shape[2], frames_per_chunk=model.config.frames_per_chunk,
                        local_attention_frames=model.config.local_attention_frames,
                        fps=fps, modalities=["text", "video"], history="generated",
                        toml_sha256=hashlib.sha256(Path(args.toml).read_bytes()).hexdigest(),
                        videos={name: name + ".mp4" for name in ("generated", "gt", "preview")})
        (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        print(json.dumps(dict(output=str(output), frames=len(pred), fps=fps)))
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
