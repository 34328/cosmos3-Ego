"""Versioned AR V0.3 action objective; V0.2 remains unchanged."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from .loss import ACTION_SUBBLOCKS, whole_action_flow_loss as _v02_whole_action_flow_loss


AR_V03_ACTION_CHANNEL_WEIGHTS = tuple(
    3.0 if 9 <= channel < 18 or 33 <= channel < 42 else 1.0
    for channel in range(57)
)


def whole_action_flow_loss(
    *,
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    visibility: list[torch.Tensor],
    valid_mask: list[torch.Tensor | None] | None = None,
    mask_out_of_fov: bool = True,
    collect_field_metrics: bool = False,
    channel_weights: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean weighted future-coordinate MSE, then mean over active samples.

    Both numerator and denominator include the same channel weights. Weights
    are finite, strictly positive, fixed 57D metadata, detached from autograd.
    The eight field diagnostics remain the original unweighted masked means.
    None or all-one weights delegate to the V0.2 numerical path verbatim.
    """
    kwargs = dict(
        pred=pred, target=target, condition_mask=condition_mask,
        visibility=visibility, valid_mask=valid_mask,
        mask_out_of_fov=mask_out_of_fov, collect_field_metrics=collect_field_metrics,
    )
    if channel_weights is None:
        return _v02_whole_action_flow_loss(**kwargs)
    weights = torch.as_tensor(channel_weights, dtype=torch.float32).detach()
    if weights.shape != (57,) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("channel_weights must contain 57 finite positive values")
    if (weights == 1).all():
        return _v02_whole_action_flow_loss(**kwargs)
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
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if valid.ndim != 2 or valid.shape[0] not in (1, rows) or valid.shape[1] not in (57, prediction.shape[1]):
                raise ValueError("valid_mask must be [57/D], [1,57/D] or [T,57/D]")
            valid = valid.to(prediction.device).detach()
            if not ((valid == 0) | (valid == 1)).all():
                raise ValueError("valid_mask must be binary")
            keep &= valid[:, :57].bool()
        prediction_valid = torch.where(keep, prediction[:, :57].float(), 0)
        label_valid = torch.where(keep, label[:, :57].float(), 0)
        error = (prediction_valid - label_valid).square()
        sample_weights = weights.to(device=prediction.device)
        denominator = (keep * sample_weights).sum()
        numerator = (error * sample_weights).sum()
        # Preserve exact normalization even when all positive weights are <1.
        losses.append(numerator / torch.where(denominator > 0, denominator, 1))
        counts.append(keep.sum())
        for name in field_losses:
            columns, _ = ACTION_SUBBLOCKS[name]
            field_losses[name].append(
                error.detach()[:, columns].sum() / keep[:, columns].sum().clamp_min(1)
            )
    per_sample = torch.stack(losses)
    valid_coordinates = torch.stack(counts)
    active = valid_coordinates > 0
    return per_sample.sum() / active.sum().clamp_min(1), {
        "per_sample_losses": per_sample,
        "active_samples": active,
        "valid_coordinates": valid_coordinates,
        **({"field_per_sample_losses": {
            name: torch.stack(values) for name, values in field_losses.items()
        }} if collect_field_metrics else {}),
    }
