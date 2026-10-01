"""V0.3 joint diffusion-forcing history refresh on the V0.2 sampler/cache."""

import dataclasses
import math

import torch

from cosmos_framework.model.generator.teacher_forcing import mark_modality_as_clean_condition

from .ar_v02_inference import JointARSampler
from .ar_v02_packing import select_joint_pack


def _refresh_modality(modality, condition_types, *, sigma, generator, max_timestep, action=False):
    """Copy payload and timestep routing while preserving clean U/S conditions."""
    tokens, masks, indexes, timesteps, frame_indexes = [], [], [], [], []
    cursor = 0
    for token, shape, mask in zip(modality.tokens, modality.token_shapes, modality.condition_mask):
        frames = shape[0]
        patches = 1 if action else shape[1] * shape[2]
        size = frames * patches
        types = condition_types[cursor:cursor + size].reshape(frames, patches)
        if not torch.equal(types.any(dim=1), types.all(dim=1)):
            raise ValueError("condition roles must be constant within each video frame")
        active = ~types[:, 0].bool()
        rows = torch.where(active)[0].to(token.device)
        out = token.clone()
        if action:
            if token.shape != (frames, 64) or torch.count_nonzero(token[:, 57:]):
                raise ValueError("V0.3 refresh requires 57D action and zero 7D padding")
            clean = token[rows, :57].float()
        else:
            clean = token.index_select(2, rows).float()
        if rows.numel():
            noise = torch.randn(clean.shape, device=token.device, dtype=torch.float32, generator=generator)
            noised = ((1 - sigma) * clean + sigma * noise).to(token.dtype)
            if action:
                out[rows, :57] = noised
            else:
                out[:, :, rows] = noised
        keep = active.repeat_interleave(patches)
        sequence = modality.sequence_indexes[cursor:cursor + size]
        selected = sequence[keep.to(sequence.device)]
        indexes.append(selected)
        timesteps.append(torch.full((selected.numel(),), sigma * max_timestep,
                                    device=modality.timesteps.device, dtype=torch.float32))
        new_mask = torch.ones_like(mask)
        new_mask[rows.to(mask.device)] = 0
        tokens.append(out)
        masks.append(new_mask)
        frame_indexes.append(rows)
        cursor += size
    if cursor != condition_types.numel() or cursor != modality.sequence_indexes.numel():
        raise ValueError("refresh condition metadata does not match modality payload")
    return dataclasses.replace(modality, tokens=tokens, condition_mask=masks,
                               mse_loss_indexes=torch.cat(indexes), timesteps=torch.cat(timesteps),
                               noisy_frame_indexes=frame_indexes)


def refresh_noise_pack(packed, *, sigma_small, generator, video_max_timestep, action_max_timestep):
    """Noised history write only; input/GT and clean U/S stay untouched.

    mse_loss_indexes here route timestep embeddings, and never enter loss computation.
    Sigma zero returns the original clean pack without consuming random numbers.
    """
    sigma = float(sigma_small)
    if not math.isfinite(sigma) or not 0 <= sigma <= 1:
        raise ValueError("sigma_small must be finite and in [0,1]")
    if sigma == 0:
        return packed
    if packed.vision is None or packed.action is None:
        raise ValueError("joint refresh requires both video and action")
    vision = _refresh_modality(packed.vision, packed.vision_condition_type_mask,
                               sigma=sigma, generator=generator, max_timestep=video_max_timestep)
    action = _refresh_modality(packed.action, packed.action_state_mask,
                               sigma=sigma, generator=generator, max_timestep=action_max_timestep, action=True)
    return dataclasses.replace(packed, vision=vision, action=action, uses_single_timestep=False)


class DiffusionForcingJointARSampler(JointARSampler):
    """Keep V0.2 denoising/output order; only completed-block KV refresh changes."""

    @torch.no_grad()
    def _cache_forward(self, video, action, indexes, *, chunk, phase, video_sigma=0.0, action_sigma=0.0):
        if phase != "refresh" or self.sigma_small == 0:
            return super()._cache_forward(video, action, indexes, chunk=chunk, phase=phase,
                                          video_sigma=video_sigma, action_sigma=action_sigma)
        self._cache_template.vision.tokens = [video]
        self._cache_template.action.tokens = [action]
        packed = select_joint_pack(self._cache_template, self.layout, indexes, include_text=False)
        for modality in (packed.vision, packed.action):
            mark_modality_as_clean_condition(modality)
        generator = torch.Generator(device=video.device).manual_seed(self._history_noise_seed + chunk)
        packed = refresh_noise_pack(
            packed, sigma_small=self.sigma_small, generator=generator,
            video_max_timestep=self.model.rectified_flow_video.noise_scheduler.config.num_train_timesteps,
            action_max_timestep=self.model.rectified_flow_action.noise_scheduler.config.num_train_timesteps,
        )
        if self._cache_phase != (chunk, phase):
            self.cache.begin(indexes, chunk=chunk, capture=True, include_text=False)
            self._cache_phase = (chunk, phase)
        packed.to_cuda()
        self.model._cast_generated_tokens_to_precision(packed)
        out = self.model.denoise(data_batch_packed=packed, memory=self.cache)
        self.cache.ensure_complete()
        return out

    @torch.no_grad()
    def sample(self, *, sigma_small=0.02, **kwargs):
        sigma = float(sigma_small)
        if not math.isfinite(sigma) or not 0 <= sigma <= 1:
            raise ValueError("sigma_small must be finite and in [0,1]")
        if kwargs.pop("history_video_sigma", 0.0) != 0:
            raise ValueError("V0.3 uses joint sigma_small instead of video-only history_video_sigma")
        if sigma and (not kwargs.get("use_cache", True) or kwargs.get("verify_cache", False)):
            raise ValueError("nonzero sigma_small requires persistent cache without clean-reference verification")
        self.sigma_small = sigma
        output = super().sample(history_video_sigma=0.0, **kwargs)
        for report in self.chunk_reports:
            report.update(sigma_small=sigma, history_video_sigma=sigma, history_action_sigma=sigma,
                          clean_refresh_calls=int(sigma == 0 and kwargs.get("use_cache", True)),
                          noisy_refresh_calls=int(sigma > 0))
        return output
