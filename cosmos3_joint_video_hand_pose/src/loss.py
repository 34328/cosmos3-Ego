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


def whole_action_flow_loss(
    *,
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    visibility: list[torch.Tensor],
    valid_mask: list[torch.Tensor | None] | None = None,
    mask_out_of_fov: bool = True,
    collect_field_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """v0.2: one masked mean over all valid future (time, coordinate) pairs.

    Conditions, structural padding and the last seven padded coordinates
    contribute neither numerator nor denominator. The legacy default also
    excludes unseen hands. Wrist-local delta actions disable that FOV mask:
    tracking validity is enforced by rejecting corrupt windows upstream.
    Visibility remains metadata, never a replacement for tracking validity.
    Samples have
    equal weight; empty samples are reported separately for global reduction.
    Action timestep weighting is uniform. v0.1's block reduction is unchanged.
    """
    if not pred or not (len(pred) == len(target) == len(condition_mask) == len(visibility)):
        raise ValueError("action loss requires matching non-empty sample lists")
    masks = [None] * len(pred) if valid_mask is None else valid_mask
    if len(masks) != len(pred):
        raise ValueError("valid_mask must have one entry per sample")
    losses, counts = [], []
    field_losses = {name: [] for name in ACTION_SUBBLOCKS} if collect_field_metrics else {}
    for prediction, label, condition, visible, valid in zip(
        pred, target, condition_mask, visibility, masks, strict=True
    ):
        if prediction.ndim != 2 or prediction.shape != label.shape or prediction.shape[1] not in (57, 64):
            raise ValueError("prediction and target must match [T,57] or [T,64]")
        rows = len(prediction)
        if condition.numel() != rows or visible.shape != (rows, 2):
            raise ValueError("condition must have T entries and visibility must be [T,2]")
        if not torch.isfinite(prediction).all() or not torch.isfinite(label).all():
            raise ValueError("non-finite action loss input; reject corrupt data before forward")
        condition = condition.reshape(rows).to(prediction.device).detach()
        if not ((condition == 0) | (condition == 1)).all():
            raise ValueError("condition mask must be binary")
        visible = visible.to(prediction.device).detach()
        if not ((visible == 0) | (visible == 1)).all():
            raise ValueError("visibility mask must be binary")
        keep = (~condition.bool())[:, None].expand(rows, 57).clone()
        if mask_out_of_fov:
            keep[:, 9:33] &= visible[:, 0:1].bool()
            keep[:, 33:57] &= visible[:, 1:2].bool()
        if valid is not None:
            # Native Cosmos masks may be channel-only; v0.2 also needs
            # coordinate-level masks for adjacent-label validity and padding.
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if valid.ndim != 2 or valid.shape[0] not in (1, rows) or valid.shape[1] not in (57, prediction.shape[1]):
                raise ValueError("valid_mask must be [57/D], [1,57/D] or [T,57/D]")
            valid = valid.to(prediction.device).detach()
            if not ((valid == 0) | (valid == 1)).all():
                raise ValueError("valid_mask must be binary")
            keep &= valid[:, :57].bool()
        # Mask BEFORE subtraction/square: finite extreme masked coordinates
        # can otherwise overflow and produce NaN gradients (0 * inf).
        prediction_valid = torch.where(keep, prediction[:, :57].float(), 0)
        label_valid = torch.where(keep, label[:, :57].float(), 0)
        error = (prediction_valid - label_valid).square()
        count = keep.sum()
        numerator = error.sum()
        losses.append(numerator / count.clamp_min(1))
        counts.append(count)
        # Diagnostics only: reuse the exact objective mask, detach before any
        # extra reduction. These means never participate in backward.
        for name in field_losses:
            columns, _ = ACTION_SUBBLOCKS[name]
            field_losses[name].append(
                error.detach()[:, columns].sum() / keep[:, columns].sum().clamp_min(1)
            )
    per_sample = torch.stack(losses)
    valid_coordinates = torch.stack(counts)
    active = valid_coordinates > 0
    total = per_sample.sum() / active.sum().clamp_min(1)
    return total, {
        "per_sample_losses": per_sample,
        "active_samples": active,
        "valid_coordinates": valid_coordinates,
        **({"field_per_sample_losses": {name: torch.stack(values) for name, values in field_losses.items()}}
           if collect_field_metrics else {}),
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
                raise ValueError(f"time_weight must return one value or {frames} values, got {raw_temporal.numel()}")
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
            aggregation_weight = (
                active if subblock_equal_weight else active * float(channel_slice.stop - channel_slice.start)
            )
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


def whole_video_flow_loss(*, pred, target, condition_mask, time_weight):
    """FP32 per-sample weighted video MSE over future coordinates only.

    Shapes follow Cosmos predictions [C,T,H,W] (optional leading singleton).
    The rectified-flow weight remains numerator-only; U_k contributes nothing.
    Returns weighted per-sample losses, unlike Cosmos' unweighted log vector.
    """
    if not pred or not (len(pred) == len(target) == len(condition_mask)):
        raise ValueError("video loss requires matching non-empty sample lists")
    values, unweighted, counts = [], [], []
    for i, (prediction, label, condition) in enumerate(zip(pred, target, condition_mask, strict=True)):
        if prediction.shape != label.shape or prediction.ndim not in (4, 5):
            raise ValueError("video prediction/target must match [C,T,H,W] or [1,C,T,H,W]")
        if prediction.ndim == 5 and prediction.shape[0] != 1:
            raise ValueError("each video item must contain one sample")
        frames = prediction.shape[-3]
        if condition.numel() != frames:
            raise ValueError("video condition mask must contain one entry per frame")
        condition = condition.detach().to(prediction.device).reshape(frames)
        if not ((condition == 0) | (condition == 1)).all():
            raise ValueError("video condition mask must be binary")
        if not torch.isfinite(prediction).all() or not torch.isfinite(label).all():
            raise ValueError("non-finite video loss input")
        keep = (~condition.bool()).reshape(frames, 1, 1).expand_as(prediction)
        squared = (torch.where(keep, prediction.float(), 0) - torch.where(keep, label.float(), 0)).square()
        weight = torch.as_tensor(
            time_weight(i, frames, prediction), device=prediction.device, dtype=torch.float32
        ).detach()
        if weight.numel() not in (1, frames) or not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("video time weight must be finite non-negative scalar or one per frame")
        weight = weight.reshape(-1, 1, 1)
        count = keep.sum()
        unweighted.append(squared.sum() / count.clamp_min(1))
        values.append((squared * weight).sum() / count.clamp_min(1))
        counts.append(count)
    per_sample = torch.stack(values)
    valid_coordinates = torch.stack(counts)
    active = valid_coordinates > 0
    return per_sample.sum() / active.sum().clamp_min(1), {
        "per_sample_losses": per_sample,
        "active_samples": active,
        "valid_coordinates": valid_coordinates,
        "unweighted_per_sample_losses": torch.stack(unweighted),
    }
