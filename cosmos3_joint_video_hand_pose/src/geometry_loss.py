"""Differentiable 3D hand geometry objectives for the fixed 57D action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .hand_skeleton import bone_lengths, prepend_wrist


RIGHT_HAND_LATENT = slice(18, 33)
LEFT_HAND_LATENT = slice(42, 57)


@dataclass(frozen=True)
class GeometryLossConfig:
    enabled: bool = False
    decode_weight: float = 0.10
    bone_weight: float = 0.02
    velocity_weight: float = 0.02
    sigma_min: float = 0.2
    sigma_max: float = 0.8
    warmup_steps: int = 100
    ramp_steps: int = 200
    bone_mode: str = "oracle_excess"
    bone_margin: float = 0.0
    huber_delta: float = 0.01
    out_of_fov_bone_weight: float = 1.0
    target_mode: str = "decoded_oracle"

    @classmethod
    def from_value(cls, value: GeometryLossConfig | dict[str, Any] | None) -> GeometryLossConfig:
        config = value if isinstance(value, cls) else cls(**(value or {}))
        if not 0 <= config.sigma_min <= config.sigma_max <= 1:
            raise ValueError("geometry sigma range must satisfy 0 <= min <= max <= 1")
        if min(config.decode_weight, config.bone_weight, config.velocity_weight) < 0:
            raise ValueError("geometry loss weights must be non-negative")
        if config.warmup_steps < 0 or config.ramp_steps < 0:
            raise ValueError("geometry warmup/ramp steps must be non-negative")
        if config.huber_delta <= 0:
            raise ValueError("geometry huber_delta must be positive")
        if config.bone_mode not in {"oracle_excess", "absolute"}:
            raise ValueError(f"unsupported bone_mode {config.bone_mode!r}")
        if config.target_mode not in {"decoded_oracle", "raw_gt"}:
            raise ValueError(f"unsupported target_mode {config.target_mode!r}")
        return config

    def ramp(self, iteration: int) -> float:
        if iteration <= self.warmup_steps:
            return 0.0
        if self.ramp_steps == 0:
            return 1.0
        return min(max((iteration - self.warmup_steps) / self.ramp_steps, 0.0), 1.0)


def predict_clean_action(
    noisy_action: torch.Tensor,
    predicted_velocity: torch.Tensor,
    sigma_action: torch.Tensor,
) -> torch.Tensor:
    """Recover ``x0`` for ``xt = sigma*epsilon + (1-sigma)*x0``."""
    if noisy_action.shape != predicted_velocity.shape:
        raise ValueError("noisy action and predicted velocity shapes differ")
    sigma = torch.as_tensor(sigma_action, device=noisy_action.device, dtype=noisy_action.dtype)
    if sigma.numel() == 1:
        sigma = sigma.reshape(1, 1)
    elif sigma.numel() == noisy_action.shape[0]:
        sigma = sigma.reshape(noisy_action.shape[0], 1)
    else:
        raise ValueError(f"sigma must be scalar or have one value per frame, got {tuple(sigma.shape)}")
    return noisy_action - sigma * predicted_velocity


def clean_action_from_target(epsilon: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    if epsilon.shape != target_velocity.shape:
        raise ValueError("epsilon and target velocity shapes differ")
    return epsilon - target_velocity


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denominator = mask.expand_as(values).sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def _hand_scale(raw_gt: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    lengths = bone_lengths(prepend_wrist(raw_gt.float()))
    trusted = lengths[visible]
    if trusted.numel() == 0:
        trusted = lengths[:1]
    return trusted.median(dim=0).values.mean().clamp_min(1e-6)


def hand_geometry_losses(
    *,
    predicted_points: torch.Tensor,
    oracle_points: torch.Tensor,
    raw_gt_points: torch.Tensor,
    visible: torch.Tensor,
    active: torch.Tensor,
    config: GeometryLossConfig,
) -> dict[str, torch.Tensor]:
    """Compute one hand's point, bone and temporal objectives and metrics."""
    expected = (predicted_points.shape[0], 20, 3)
    if tuple(predicted_points.shape) != expected or oracle_points.shape != predicted_points.shape:
        raise ValueError("decoded hand tensors must have matching [T,20,3] shapes")
    if raw_gt_points.shape != predicted_points.shape:
        raise ValueError("raw GT hand tensor must match decoded [T,20,3] shape")
    for name, value in (("predicted", predicted_points), ("oracle", oracle_points), ("raw_gt", raw_gt_points)):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite {name} hand points")

    visible = visible.reshape(-1).to(device=predicted_points.device, dtype=torch.bool)
    active = active.reshape(-1).to(device=predicted_points.device, dtype=torch.bool)
    point_mask = active & visible
    scale = _hand_scale(raw_gt_points, visible)
    point_target = oracle_points if config.target_mode == "decoded_oracle" else raw_gt_points
    point_error = F.smooth_l1_loss(
        predicted_points / scale, point_target / scale, reduction="none", beta=config.huber_delta
    )
    decode = _masked_mean(point_error, point_mask)
    mpjpe = _masked_mean(torch.linalg.vector_norm(predicted_points - raw_gt_points, dim=-1), point_mask)

    pred_lengths = bone_lengths(prepend_wrist(predicted_points))
    oracle_lengths = bone_lengths(prepend_wrist(oracle_points))
    gt_lengths = bone_lengths(prepend_wrist(raw_gt_points.float()))
    trusted = gt_lengths[visible]
    reference = (trusted if trusted.numel() else gt_lengths[:1]).median(dim=0).values.clamp_min(1e-6)
    pred_relative = (pred_lengths - reference).abs() / reference
    oracle_relative = (oracle_lengths - reference).abs() / reference
    pred_bone_error = F.smooth_l1_loss(
        pred_relative, torch.zeros_like(pred_relative), reduction="none", beta=config.huber_delta
    ).mean(dim=-1)
    oracle_bone_error = F.smooth_l1_loss(
        oracle_relative, torch.zeros_like(oracle_relative), reduction="none", beta=config.huber_delta
    ).mean(dim=-1)
    if config.bone_mode == "oracle_excess":
        bone_per_frame = torch.relu(pred_bone_error - oracle_bone_error - config.bone_margin)
    else:
        bone_per_frame = pred_bone_error
    bone_mask = active & (visible | (config.out_of_fov_bone_weight > 0))
    bone_weights = torch.where(visible, torch.ones_like(pred_bone_error), config.out_of_fov_bone_weight)
    bone = _masked_mean(bone_per_frame * bone_weights, bone_mask)

    # A clean condition frame may serve as the reference for the first future
    # velocity, but is never itself a target.
    pair_mask = active[1:] & visible[1:] & visible[:-1]
    predicted_velocity = predicted_points[1:] - predicted_points[:-1]
    oracle_velocity = point_target[1:] - point_target[:-1]
    velocity_error = F.smooth_l1_loss(
        predicted_velocity / scale, oracle_velocity / scale, reduction="none", beta=config.huber_delta
    )
    velocity = _masked_mean(velocity_error, pair_mask)
    return {
        "decode": decode,
        "bone": bone,
        "velocity": velocity,
        "mpjpe_raw": mpjpe,
        "bone_relative_mean": _masked_mean(pred_relative, active),
        "coverage": point_mask.float().mean(),
    }
