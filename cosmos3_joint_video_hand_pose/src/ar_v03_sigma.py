"""Stateless low-noise prefix draws, independent from official RF/noise RNG."""

from dataclasses import dataclass
import hashlib
import math

import torch


PREFIX_SIGMA_RNG_NAMESPACE = "ar_v03_uniform_prefix_sigma_v1"


@dataclass(frozen=True)
class PrefixLowNoisePlan:
    mask: torch.Tensor  # CPU [B,max_chunks+1], block IDs start at 1
    sigmas: torch.Tensor  # CPU [B,max_chunks+1], shared video/action values
    metadata: list[dict]


def sample_prefix_low_noise(chunk_counts, *, seed, iteration, rank, sigma_hist_max=0.1):
    """Per sample L~Unif{1..N}; blocks 1..L-1 get shared U[0,max) sigma.

    A private CPU generator is recreated from a stable namespace, experiment
    seed, optimizer iteration, DP rank and sample ordinal. There is no mutable
    generator cursor to checkpoint, and no CPU/CUDA global RNG is consumed.
    AR V0.3's fixed grad_accum_iter=1 makes the optimizer iteration identify
    the batch; replaying a checkpoint with the same layout/iteration/rank
    reproduces exactly the same prefix regardless of prior process draws.
    """
    if iteration is None or int(iteration) < 0 or int(rank) < 0:
        raise ValueError("prefix sampling requires a nonnegative iteration and rank")
    if not math.isfinite(float(sigma_hist_max)) or not 0 < float(sigma_hist_max) <= 1:
        raise ValueError("sigma_hist_max must be finite and in (0,1]")
    counts = [int(n) for n in chunk_counts]
    if not counts or any(n < 1 for n in counts):
        raise ValueError("one positive actual chunk count per sample is required")
    shape = (len(counts), max(counts) + 1)
    mask = torch.zeros(shape, dtype=torch.bool)
    sigmas = torch.zeros(shape, dtype=torch.float32)
    metadata = []
    for sample, count in enumerate(counts):
        payload = f"{PREFIX_SIGMA_RNG_NAMESPACE}:{int(seed)}:{int(iteration)}:{int(rank)}:{sample}"
        private_seed = int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "little") % (1 << 63)
        generator = torch.Generator(device="cpu").manual_seed(private_seed)
        length = int(torch.randint(1, count + 1, (1,), generator=generator))
        values = torch.rand(length - 1, generator=generator, dtype=torch.float32) * float(sigma_hist_max)
        mask[sample, 1:length] = True
        sigmas[sample, 1:length] = values
        metadata.append(dict(
            sample_index=sample, n_chunks=count, prefix_length=length,
            prefix_chunks=list(range(1, length)), shared_sigmas=values.tolist(),
            seed=private_seed, iteration=int(iteration), rank=int(rank),
            distribution="uniform", sigma_hist_max=float(sigma_hist_max),
            rng_namespace=PREFIX_SIGMA_RNG_NAMESPACE,
        ))
    return PrefixLowNoisePlan(mask=mask, sigmas=sigmas, metadata=metadata)


def apply_prefix_low_noise(original_sigmas, plan):
    """Replace only prefix slots; keep all suffix/unused RF draw bits intact."""
    if tuple(original_sigmas.shape) != tuple(plan.mask.shape):
        raise ValueError("RF chunk sigma matrix differs from prefix layout")
    mask = plan.mask.to(original_sigmas.device)
    replacement = plan.sigmas.to(device=original_sigmas.device, dtype=original_sigmas.dtype)
    return torch.where(mask, replacement, original_sigmas)


def continuous_rf_timesteps(sigmas, rectified_flow):
    """Match official OmniMoT continuous training sigma→time, without re-shift.

    omni_mot_model._get_train_noise_level_vision/action multiplies the already
    shifted physical sigma by noise_scheduler.config.num_train_timesteps.
    The official flow sampler's _sigma_to_t uses the identical conversion;
    get_discrete_timestamp is unrelated and must not quantize this path.
    """
    return sigmas * rectified_flow.noise_scheduler.config.num_train_timesteps
