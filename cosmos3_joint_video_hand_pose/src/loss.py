from __future__ import annotations

from collections.abc import Callable

import torch


ACTION_SUBBLOCKS = {
    "camera_translation": (slice(0, 3), None),
    "camera_rotation": (slice(3, 9), None),
    "right_wrist_translation": (slice(9, 12), 0),
    "right_wrist_rotation": (slice(12, 18), 0),
    "right_hand_latent": (slice(18, 33), 0),
    "left_wrist_translation": (slice(33, 36), 1),
    "left_wrist_rotation": (slice(36, 42), 1),
    "left_hand_latent": (slice(42, 57), 1),
}


def visibility_weighted_action_flow_loss(
    *,
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    visibility: list[torch.Tensor],
    time_weight: Callable[[int, int, torch.Tensor], torch.Tensor] | None = None,
    lambda_out_of_fov: float = 0.0,
    subblock_equal_weight: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute an equally weighted action loss over active physical sub-blocks.

    When ``subblock_equal_weight`` is true, every active sub-block has the same per-sample weight regardless of its
    channel count.  This keeps the 15D hand latent from overwhelming a 3D
    wrist translation just because it has more coordinates.  A hand-side
    visibility mask applies identically to its translation, rotation, and
    latent blocks, including their numerators and denominators.
    """
    if not 0 <= lambda_out_of_fov <= 1:
        raise ValueError("lambda_out_of_fov must be in [0,1]")
    if not (len(pred) == len(target) == len(condition_mask) == len(visibility)):
        raise ValueError("action loss lists must have the same number of samples")
    if not pred:
        raise ValueError("action loss requires at least one sample")

    block_loss_sums = {name: pred[0].new_zeros(()) for name in ACTION_SUBBLOCKS}
    block_active_samples = {name: pred[0].new_zeros(()) for name in ACTION_SUBBLOCKS}
    block_weight_sums = {name: pred[0].new_zeros(()) for name in ACTION_SUBBLOCKS}
    sample_active_blocks = []
    per_sample_losses = []

    for sample_index, (prediction, label, clean_mask, visible) in enumerate(
        zip(pred, target, condition_mask, visibility, strict=True)
    ):
        if prediction.shape != label.shape or prediction.ndim != 2 or prediction.shape[1] < 57:
            raise ValueError("pred/target must have matching [T,D>=57] shapes")
        frames = prediction.shape[0]
        noisy = (1.0 - clean_mask.reshape(frames).to(prediction)).detach()
        visible = visible.reshape(frames, 2).to(device=prediction.device, dtype=prediction.dtype).detach()
        if time_weight is None:
            temporal = torch.ones_like(noisy)
        else:
            raw_temporal = time_weight(sample_index, frames, prediction)
            # Cosmos' base/teacher-forcing schedule supplies one timestep per
            # sample, while diffusion-forcing supplies one per frame.  Treat a
            # scalar (or singleton) weight as constant across this sample.
            raw_temporal = torch.as_tensor(raw_temporal, device=prediction.device, dtype=prediction.dtype)
            if raw_temporal.numel() == 1:
                temporal = raw_temporal.expand(frames)
            elif raw_temporal.numel() == frames:
                temporal = raw_temporal.reshape(frames)
            else:
                raise ValueError(
                    f"time_weight must return one value or {frames} values, got {raw_temporal.numel()}"
                )
            temporal = temporal.detach()
        # Visibility weights participate in both numerator and denominator so
        # changing lambda does not dilute the hand loss.  Cosmos' rectified-flow
        # timestep weight intentionally remains numerator-only, matching the
        # native flow-matching objective.
        hand_weights = (
            noisy * (visible[:, 0] + lambda_out_of_fov * (1.0 - visible[:, 0])),
            noisy * (visible[:, 1] + lambda_out_of_fov * (1.0 - visible[:, 1])),
        )
        squared_error = (prediction[:, :57] - label[:, :57]).square()
        sample_block_sum = prediction.new_zeros(())
        sample_block_weight = prediction.new_zeros(())
        for name, (channel_slice, hand_index) in ACTION_SUBBLOCKS.items():
            per_frame = squared_error[:, channel_slice].mean(dim=-1)
            weight = noisy if hand_index is None else hand_weights[hand_index]
            denominator = weight.sum()
            active = (denominator > 0).to(dtype=prediction.dtype)
            block_loss = (per_frame * temporal * weight).sum() / denominator.clamp_min(1e-12)
            # The legacy reduction aggregates camera/right/left groups by
            # native action width. overfit_v0.0 uses equal physical sub-blocks.
            aggregation_weight = active if subblock_equal_weight else active * float(channel_slice.stop - channel_slice.start)
            sample_block_sum = sample_block_sum + block_loss * aggregation_weight
            sample_block_weight = sample_block_weight + aggregation_weight
            block_loss_sums[name] = block_loss_sums[name] + block_loss * active
            block_active_samples[name] = block_active_samples[name] + active
            block_weight_sums[name] = block_weight_sums[name] + denominator.detach()
        per_sample_losses.append(sample_block_sum / sample_block_weight.clamp_min(1.0))
        sample_active_blocks.append(sample_block_weight.detach())

    per_sample_losses_tensor = torch.stack(per_sample_losses)
    total = per_sample_losses_tensor.mean()
    block_losses = {
        name: block_loss_sums[name] / block_active_samples[name].clamp_min(1.0) for name in ACTION_SUBBLOCKS
    }
    metrics = {
        **{f"{name}_loss": value for name, value in block_losses.items()},
        **{f"{name}_weight": block_weight_sums[name] for name in ACTION_SUBBLOCKS},
        "active_action_blocks": torch.stack(sample_active_blocks).mean(),
        "per_sample_losses": per_sample_losses_tensor,
    }
    return total, metrics
