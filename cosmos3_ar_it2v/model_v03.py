"""Single-target DF with low-noise GT history; native Cosmos training/loss."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache

import attrs
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from .attention import chunk_ids
from .model import ARIT2VModel, ARIT2VModelConfig


@attrs.define(slots=False)
class ARIT2VModelV03Config(ARIT2VModelConfig):
    history_noise_mode: str = "lingbot_public"
    history_sigma_max: float = .1
    clean_history_probability: float = .5
    target_history_seed: int = 42


@dataclass(frozen=True)
class TargetChunkPlan:
    """One target per segment, in the unchanged packed vision-item order."""

    frame_counts: tuple[int, ...]
    target_chunks: tuple[int, ...]
    clean_history: tuple[bool, ...]
    chunk_size: int


def target_history_generator(*, iteration: int, rank: int, seed: int):
    """Separate, stateless stream; never advances native sigma/epsilon RNGs."""
    if iteration < 0 or rank < 0:
        raise ValueError("iteration and rank must be nonnegative")
    # Domain-separated from native iteration*65536+rank sigma/epsilon seeds.
    mixed = (int(seed) + 0x4152495432563033) ^ (int(iteration) * 1_000_003) ^ (int(rank) * 9_176_293)
    return torch.Generator(device="cpu").manual_seed(mixed & ((1 << 63) - 1))


@lru_cache(maxsize=1)
def _lingbot_public_sigma_schedule():
    """Official flow scheduler used by LingBot's published training recipe."""
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5).sigmas.cpu()


def sample_lingbot_public_history_sigmas(num_latents, *, generator):
    """Per-latent indices in [500,999], without target min/max clamping.

    The native descending schedule is shift5(1-index/1000), so these indices
    give sigma approximately .004975.. .833333. Index 999 is not sigma zero.
    """
    indices = torch.randint(500, 1000, (num_latents,), generator=generator)
    return _lingbot_public_sigma_schedule()[indices], indices


def sample_target_chunk_sigmas(base_sigmas, frame_counts, *, chunk_size=4,
                               history_sigma_max=.1, clean_history_probability=.5,
                               history_noise_mode="lingbot_public", generator):
    """Keep the original target draw; replace only earlier blocks with low sigma.

    ``base_sigmas`` must already have been drawn for the full unchanged batch
    by V0.2. Each segment selects k uniformly from its non-condition chunks.
    One Bernoulli selects entirely clean history. ``lingbot_public`` otherwise
    samples scheduler indices 500..999 independently for each historical latent.
    The optional ``uniform`` mode draws one U(0, history_sigma_max) per chunk.
    Later chunks retain their original draws but cannot be seen by target k.
    """
    counts = tuple(map(int, frame_counts))
    if not counts or min(counts) < 2 or chunk_size < 1:
        raise ValueError("one future latent and a positive chunk size are required")
    if not (0 <= history_sigma_max <= 1 and 0 <= clean_history_probability <= 1):
        raise ValueError("invalid history noise support/probability")
    if history_noise_mode not in ("lingbot_public", "uniform"):
        raise ValueError("unknown history noise mode")
    if base_sigmas.ndim != 2 or base_sigmas.shape != (len(counts), max(counts)):
        raise ValueError("base sigma rows must match the full ragged batch")
    if generator.device.type != "cpu":
        raise ValueError("target/history RNG must be its independent CPU stream")
    sigmas = base_sigmas.clone()
    targets, clean = [], []
    # Index draws get their own stream. Target selection/clean Bernoulli retain
    # the initial uniform-mode draw protocol and never touch native RNG state.
    history_generator = torch.Generator(device="cpu").manual_seed(
        (generator.initial_seed() ^ 0x4C494E47424F54) & ((1 << 63) - 1))
    for i, t in enumerate(counts):
        n = (t - 2) // chunk_size + 1
        target = int(torch.randint(1, n + 1, (1,), generator=generator).item())
        clean_row = bool(torch.rand((), generator=generator).item() < clean_history_probability)
        # Draw a full row regardless of clean probability/support: changing
        # history settings does not change subsequent segments' target draws.
        uniform_history = torch.rand(n, generator=generator, dtype=torch.float32) * history_sigma_max
        if history_noise_mode == "lingbot_public":
            history, _ = sample_lingbot_public_history_sigmas(t - 1, generator=history_generator)
        else:
            history = uniform_history.repeat_interleave(chunk_size)[:t-1]
        if clean_row:
            history.zero_()
        history = history.to(device=sigmas.device, dtype=sigmas.dtype)
        begin = 1 + (target - 1) * chunk_size
        sigmas[i, 1:begin] = history[:begin-1]
        sigmas[i, 0] = 0
        sigmas[i, t:] = 0
        targets.append(target)
        clean.append(clean_row)
    return sigmas, TargetChunkPlan(counts, tuple(targets), tuple(clean), chunk_size)


class ARIT2VModelV03(ARIT2VModel):
    """Native single forward with only target k contributing direct supervision.

    Historical inputs remain in the differentiable causal stream. Their output
    velocities get zero direct loss, while their hidden K/V still receive the
    target's gradient. No history is relabeled as a conditioning frame.
    """

    def __init__(self, config):
        if config.history_noise_mode not in ("lingbot_public", "uniform"):
            raise ValueError("unknown V0.3 history noise mode")
        if not (0 <= config.history_sigma_max <= 1 and 0 <= config.clean_history_probability <= 1):
            raise ValueError("invalid V0.3 history noise support/probability")
        super().__init__(config)

    def training_step(self, data_batch, iteration):
        if getattr(self, "_v03_step_iteration", None) is not None:
            raise RuntimeError("nested V0.3 training steps cannot share target metadata")
        self._v03_step_iteration = int(iteration)
        self._v03_pending_plan = None
        try:
            return super().training_step(data_batch, iteration)
        finally:
            # Plans belong to their packed sequence, not the model's next
            # microbatch, inference invocation, or resumed training step.
            self._v03_pending_plan = None
            self._v03_step_iteration = None

    def _get_train_noise_level_vision(self, batch_size, is_image_batch,
                                    num_vision_latent_frames, resolutions=None,
                                    num_tokens=None, iteration=None):
        if iteration is None or getattr(self, "_v03_step_iteration", None) != int(iteration):
            raise RuntimeError("V0.3 sigma sampling requires the current training-step scope")
        if getattr(self, "_v03_pending_plan", None) is not None:
            raise RuntimeError("unconsumed V0.3 target metadata")
        # Preserve V0.2's complete sigma sampling call and RNG draw order.
        _, original_sigmas = super()._get_train_noise_level_vision(
            batch_size, is_image_batch, num_vision_latent_frames,
            resolutions=resolutions, num_tokens=num_tokens, iteration=iteration)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        cfg = self.config
        generator = target_history_generator(iteration=int(iteration), rank=rank,
                                             seed=cfg.target_history_seed)
        sigmas, plan = sample_target_chunk_sigmas(
            original_sigmas, num_vision_latent_frames, chunk_size=cfg.frames_per_chunk,
            history_noise_mode=cfg.history_noise_mode,
            history_sigma_max=cfg.history_sigma_max,
            clean_history_probability=cfg.clean_history_probability, generator=generator)
        self._v03_pending_plan = plan
        return sigmas * float(self.rectified_flow_video.noise_scheduler.config.num_train_timesteps), sigmas

    def pre_noise_memory_hook(self, packed_sequence, gen_data_clean, memory_info):
        memory_info = super().pre_noise_memory_hook(packed_sequence, gen_data_clean, memory_info)
        plan = getattr(self, "_v03_pending_plan", None)
        if plan is None or getattr(self, "_v03_step_iteration", None) is None:
            raise RuntimeError("V0.3 packed sequence has no current target plan")
        counts = tuple(int(x.shape[2]) for x in gen_data_clean.x0_tokens_vision)
        mask_counts = tuple(int(mask.shape[0]) for mask in packed_sequence.vision.condition_mask)
        if counts != plan.frame_counts or mask_counts != counts:
            raise ValueError("V0.3 target plan does not match this packed sequence")
        packed_sequence.it2v_v03_target_plan = plan
        self._v03_pending_plan = None
        return memory_info

    def _compute_losses(self, out_net, data_batch_packed, gen_data_noised,
                        timesteps, is_image_batch, **kwargs):
        plan = getattr(data_batch_packed, "it2v_v03_target_plan", None)
        if not isinstance(plan, TargetChunkPlan):
            raise ValueError("V0.3 training loss requires its packed target plan")
        masks = data_batch_packed.vision.condition_mask
        if (tuple(int(mask.shape[0]) for mask in masks) != plan.frame_counts
                or plan.chunk_size != self.config.frames_per_chunk
                or len(plan.target_chunks) != len(masks)):
            raise ValueError("V0.3 loss plan does not match packed video geometry")
        loss_pack = copy.copy(data_batch_packed)
        loss_pack.vision = copy.copy(data_batch_packed.vision)
        loss_pack.vision.condition_mask = []
        for t, target, mask in zip(plan.frame_counts, plan.target_chunks, masks, strict=True):
            if not 1 <= target <= (t - 2) // plan.chunk_size + 1:
                raise ValueError("V0.3 target chunk is outside this segment")
            ids = chunk_ids(torch.arange(t, device=mask.device), plan.chunk_size)
            selected = (ids == target).reshape(-1, *([1] * (mask.ndim - 1)))
            loss_pack.vision.condition_mask.append(1 - (1 - mask.float()) * selected)
        # ARIT2VModel applies real-frame tail weights to this loss-only mask.
        # The native flow loss divides by target active coordinates, then keeps
        # native per-segment/global sample averaging. No n_chunks correction.
        return super()._compute_losses(
            out_net=out_net, data_batch_packed=loss_pack,
            gen_data_noised=gen_data_noised, timesteps=timesteps,
            is_image_batch=is_image_batch, **kwargs)
