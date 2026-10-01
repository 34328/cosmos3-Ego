"""Video-only augmentation of the TF condition pass; never mutate GT/targets.

The official network uses mse_loss_indexes also to route timestep embeddings.
Here they only route embeddings: this pack is never passed to loss computation.
"""
import dataclasses
import torch
from .ar_v02_layout import VIDEO


def sample_history_sigmas(layouts, *, probability, sigma_max, seed, device):
    if not 0 <= probability <= 1 or not 0 <= sigma_max <= 1:
        raise ValueError("history probability and sigma_max must be in [0,1]")
    generator = torch.Generator(device=device).manual_seed(seed)
    result = []
    for layout in layouts:
        roles, chunks, _ = layout.video_metadata(device=device)
        values = torch.zeros(roles.numel(), device=device)
        if probability and sigma_max and torch.rand((), generator=generator, device=device) < probability:
            per_chunk = torch.rand(len(layout.boundaries) + 1, generator=generator, device=device) * sigma_max
            values[roles == VIDEO] = per_chunk[chunks[roles == VIDEO]]
        result.append(values)
    return result, generator


def history_noise_pack(packed, frame_sigmas, *, generator, max_timestep):
    """Return the original object at sigma=0, otherwise a copy of vision only.

    Input must be a condition/refresh pack. U frames must have sigma=0.
    A separate RNG means target noise and sigma draws keep their original order.
    """
    if packed.vision is None or len(frame_sigmas) != len(packed.vision.tokens):
        raise ValueError("history sigmas must match video payloads")
    for value, token in zip(frame_sigmas, packed.vision.tokens):
        if value.shape != (token.shape[2],) or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("invalid per-frame history sigma")
    if not any(torch.count_nonzero(s).item() for s in frame_sigmas):
        return packed
    vision = packed.vision
    tokens, masks, indexes, times, frame_indexes = [], [], [], [], []
    cursor = 0
    for token, shape, sigma, condition in zip(vision.tokens, vision.token_shapes, frame_sigmas, vision.condition_mask):
        sigma = sigma.to(device=token.device, dtype=torch.float32)
        patches = shape[1] * shape[2]
        active = sigma > 0
        types = packed.vision_condition_type_mask[cursor:cursor + shape[0] * patches].reshape(shape[0], patches)
        if types[active.to(types.device)].any():
            raise ValueError("U condition images must remain clean")
        out = token.clone()
        rows = torch.where(active)[0]
        if rows.numel():
            clean = token.index_select(2, rows).float()
            noise = torch.randn(clean.shape, device=token.device, generator=generator, dtype=torch.float32)
            s = sigma[rows].reshape(1, 1, -1, 1, 1)
            out[:, :, rows] = ((1 - s) * clean + s * noise).to(token.dtype)
        mask = torch.ones_like(condition)
        mask[rows.to(mask.device)] = 0
        keep = active.repeat_interleave(patches)
        indexes.append(vision.sequence_indexes[cursor:cursor + keep.numel()][keep.to(vision.sequence_indexes.device)])
        times.append((sigma[active] * max_timestep).repeat_interleave(patches))
        tokens.append(out)
        masks.append(mask)
        frame_indexes.append(rows)
        cursor += keep.numel()
    modified = dataclasses.replace(vision, tokens=tokens, condition_mask=masks,
        mse_loss_indexes=torch.cat(indexes), timesteps=torch.cat(times).float(), noisy_frame_indexes=frame_indexes)
    return dataclasses.replace(packed, vision=modified, uses_single_timestep=False)
