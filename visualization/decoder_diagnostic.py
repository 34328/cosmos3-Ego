"""Compare decoder prefixes using one continuous, normalized latent segment.

The GT-prefix montage is a collection of GT-conditioned single-block
diagnostics. It is not a continuous rollout: each block is decoded from the
original segment start with its own complete GT prefix. No RGB is re-encoded.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from cosmos3_ar_it2v.inference import chunk_ranges


LATENT_FORMAT = "cosmos3_ar_it2v_continuous_normalized_latents_v1"
COMPARISON_LABELS = {
    "predicted_prefix": "整段预测 latent 连续解码",
    "gt_prefix_montage": "GT 前缀条件单块诊断拼图；非连续 rollout",
    "gt_reconstruction": "完整 GT latent 连续重建",
}


def rgb_frame_range(start: int, end: int) -> tuple[int, int]:
    """Absolute half-open RGB interval for causal 4x latent ``start:end``."""
    if not 0 <= start < end:
        raise ValueError("latent range must be nonempty and nonnegative")
    return (0 if start == 0 else 1 + 4 * (start - 1), 1 + 4 * (end - 1))


def _validate_latents(predicted_latents, gt_latents, true_frames, frames_per_chunk):
    for name, value in (("predicted", predicted_latents), ("GT", gt_latents)):
        if (not isinstance(value, torch.Tensor) or value.ndim != 5
                or value.shape[0] != 1 or min(value.shape) < 1):
            raise ValueError(f"{name} latents must be [1,C,T,H,W]")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} latents must be finite floating-point model latents")
    if (predicted_latents.shape != gt_latents.shape
            or predicted_latents.device != gt_latents.device):
        raise ValueError("predicted and GT latents must match shape and device")
    if not torch.equal(predicted_latents[:, :, :1], gt_latents[:, :, :1]):
        raise ValueError("predicted and GT must share the conditioned first latent")
    if not isinstance(true_frames, int) or isinstance(true_frames, bool) or true_frames < 1:
        raise ValueError("true_frames must be a positive integer")
    if not isinstance(frames_per_chunk, int) or isinstance(frames_per_chunk, bool):
        raise ValueError("frames_per_chunk must be a positive integer")
    latent_frames = predicted_latents.shape[2]
    if latent_frames != 1 + (true_frames - 1 + 3) // 4:
        raise ValueError("true length must match the complete causal 4x latent segment")
    return chunk_ranges(latent_frames, frames_per_chunk)


def _decode_from_zero(model, latents):
    """Delegate the full prefix, including latent zero, to official model.decode."""
    value = latents.to(**getattr(model, "tensor_kwargs", {}))
    decoded = model.decode(value)
    expected_frames = 1 + 4 * (latents.shape[2] - 1)
    if (not isinstance(decoded, torch.Tensor) or decoded.ndim != 5
            or decoded.shape[:3] != (1, 3, expected_frames)):
        raise ValueError("model.decode must return [1,3,1+4*(T-1),H,W]")
    if not torch.isfinite(decoded).all():
        raise ValueError("nonfinite decoded RGB")
    # Keep native decoder dtype/range. Only detach and move for bounded GPU use.
    return decoded.detach().cpu()


def _validate_decoder(model):
    tokenizer = getattr(model, "tokenizer_vision_gen", None)
    if tokenizer is not None:
        if not tokenizer.is_causal or tokenizer.temporal_compression_factor != 4:
            raise ValueError("diagnostic requires the official causal 4x video tokenizer")
        if (getattr(tokenizer, "_keep_decoder_cache", False)
                or getattr(tokenizer, "keep_decoder_cache", False)):
            raise ValueError("full-prefix diagnostic cannot run inside a cached decoder scope")


@torch.no_grad()
def decode_gt_prefix(model, predicted_latents, gt_latents, *, true_frames: int,
                     frames_per_chunk: int = 4):
    """Decode only the GT-prefix single-block montage, with no A/C recomputation.

    Each call starts at latent zero and contains only completed GT history plus
    the current predicted block. The returned native RGB tensor is on CPU and
    trimmed to true_frames. This montage is not a continuous rollout.
    """
    ranges = _validate_latents(predicted_latents, gt_latents, true_frames, frames_per_chunk)
    _validate_decoder(model)
    parts = []
    for start, end in ranges:
        prefix = torch.cat((gt_latents[:, :, :start], predicted_latents[:, :, start:end]), dim=2)
        rgb_start, rgb_end = rgb_frame_range(start, end)
        decoded = _decode_from_zero(model, prefix)
        part = decoded[:, :, rgb_start:min(rgb_end, true_frames)].clone()
        if parts and part.shape[3:] != parts[0].shape[3:]:
            raise ValueError("prefix decoder spatial shapes differ")
        parts.append(part)
        del decoded, prefix
    gt_prefix_montage = torch.cat(parts, dim=2)
    if gt_prefix_montage.shape[2] != true_frames:
        raise ValueError("GT-prefix montage does not cover every true frame")
    return gt_prefix_montage


@torch.no_grad()
def decode_comparison(model, predicted_latents, gt_latents, *, true_frames: int,
                      frames_per_chunk: int = 4):
    """Return three raw RGB tensors ``[1,3,true_frames,H,W]`` on CPU.

    A and B use exactly the same predicted latents. B delegates to
    decode_gt_prefix; C is the complete GT reconstruction. The official
    decoder owns normalization and all causal cache operations.
    """
    _validate_latents(predicted_latents, gt_latents, true_frames, frames_per_chunk)
    _validate_decoder(model)
    predicted_prefix = _decode_from_zero(model, predicted_latents)[:, :, :true_frames].clone()
    gt_prefix_montage = decode_gt_prefix(model, predicted_latents, gt_latents,
        true_frames=true_frames, frames_per_chunk=frames_per_chunk)
    gt_reconstruction = _decode_from_zero(model, gt_latents)[:, :, :true_frames].clone()
    if gt_prefix_montage.shape != predicted_prefix.shape or gt_reconstruction.shape != predicted_prefix.shape:
        raise ValueError("decoded comparison shapes differ")
    return {
        "predicted_prefix": predicted_prefix,
        "gt_prefix_montage": gt_prefix_montage,
        "gt_reconstruction": gt_reconstruction,
    }


def save_latent_archive(path, predicted_latents, gt_latents, *, true_frames: int,
                        frames_per_chunk: int = 4, metadata=None):
    """Save lossless model-normalized tensors and an explicit continuous format.

    ``metadata`` should supply checkpoint, seed, sampler settings, history mode,
    caption and source-frame identity from the caller's actual run. Reserved
    geometry/format fields below are derived, not accepted from the caller.
    Existing files are never overwritten.
    """
    ranges = _validate_latents(predicted_latents, gt_latents, true_frames, frames_per_chunk)
    provenance = json.loads(json.dumps(dict(metadata or {}), ensure_ascii=False, allow_nan=False))
    latent_frames = predicted_latents.shape[2]
    provenance.update(
        latent_format=LATENT_FORMAT,
        latent_normalization="official model/tokenizer normalized latent; no additional transform",
        latent_shape=list(predicted_latents.shape), latent_dtype=str(predicted_latents.dtype),
        gt_latent_dtype=str(gt_latents.dtype),
        latent_frames=latent_frames, true_frames=true_frames,
        aligned_frames=1 + 4 * (latent_frames - 1),
        video_temporal_padding=1 + 4 * (latent_frames - 1) - true_frames,
        temporal_compression_factor=4, frames_per_chunk=frames_per_chunk,
        latent_chunk_ranges=[list(r) for r in ranges],
        rgb_chunk_ranges=[[start, min(end, true_frames)]
                          for start, end in (rgb_frame_range(*r) for r in ranges)],
        comparison_labels=COMPARISON_LABELS.copy(),
        gt_prefix_montage_is_continuous_rollout=False,
    )
    payload = {
        "format": LATENT_FORMAT,
        "predicted_latents": predicted_latents.detach().cpu().clone(),
        "gt_latents": gt_latents.detach().cpu().clone(),
        "metadata": provenance,
    }
    path = Path(path)
    with path.open("xb") as output:
        torch.save(payload, output)
    return provenance


def load_latent_archive(path):
    """Load only the explicit continuous format; reject old RGB/chunk archives."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != LATENT_FORMAT:
        raise ValueError("archive is not an explicit continuous normalized latent segment")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("latent_format") != LATENT_FORMAT:
        raise ValueError("continuous latent metadata is missing")
    ranges = _validate_latents(payload.get("predicted_latents"), payload.get("gt_latents"),
                               metadata.get("true_frames"), metadata.get("frames_per_chunk"))
    predicted = payload["predicted_latents"]
    expected = {
        "latent_shape": list(predicted.shape), "latent_dtype": str(predicted.dtype),
        "gt_latent_dtype": str(payload["gt_latents"].dtype),
        "latent_frames": predicted.shape[2], "aligned_frames": 1 + 4 * (predicted.shape[2] - 1),
        "video_temporal_padding": 1 + 4 * (predicted.shape[2] - 1) - metadata["true_frames"],
        "temporal_compression_factor": 4, "latent_chunk_ranges": [list(r) for r in ranges],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("continuous latent metadata disagrees with saved tensors")
    return payload
