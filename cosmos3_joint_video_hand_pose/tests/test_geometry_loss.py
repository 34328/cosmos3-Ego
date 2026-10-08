from __future__ import annotations

import torch

from cosmos3_joint_video_hand_pose.src.action import CODEC_ROOT
from cosmos3_joint_video_hand_pose.src.codec import FrozenHandMLPAE15
from cosmos3_joint_video_hand_pose.src.geometry_loss import (
    GeometryLossConfig,
    clean_action_from_target,
    hand_geometry_losses,
    predict_clean_action,
)


def test_clean_action_recovery() -> None:
    clean = torch.randn(4, 64)
    epsilon = torch.randn_like(clean)
    sigma = torch.tensor([[0.1], [0.3], [0.7], [0.9]])
    target_velocity = epsilon - clean
    noisy = sigma * epsilon + (1.0 - sigma) * clean
    assert torch.allclose(predict_clean_action(noisy, target_velocity, sigma), clean, atol=1e-6, rtol=1e-5)
    assert torch.allclose(clean_action_from_target(epsilon, target_velocity), clean, atol=1e-6, rtol=1e-5)


def test_differentiable_decoder_keeps_codec_frozen() -> None:
    codec = FrozenHandMLPAE15(CODEC_ROOT / "right_mlp15_primary.pt")
    latent = torch.randn(3, 15, requires_grad=True)
    expected = codec.decode(latent.detach())
    actual = codec.decode_differentiable(latent)
    assert torch.allclose(actual, expected)
    actual.square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in codec.parameters())


def test_geometry_masks_oracle_and_temporal_loss() -> None:
    codec = FrozenHandMLPAE15(CODEC_ROOT / "right_mlp15_primary.pt")
    latent = torch.randn(4, 15)
    oracle = codec.decode(latent)
    config = GeometryLossConfig(enabled=True)
    result = hand_geometry_losses(
        predicted_points=oracle.clone(),
        oracle_points=oracle,
        raw_gt_points=oracle,
        visible=torch.tensor([True, True, False, True]),
        active=torch.tensor([False, True, True, True]),
        config=config,
    )
    assert result["decode"].item() == 0.0
    assert result["bone"].item() == 0.0
    assert result["velocity"].item() == 0.0

    changed = oracle.clone()
    changed[1, 4, 0] += 0.1
    changed_result = hand_geometry_losses(
        predicted_points=changed,
        oracle_points=oracle,
        raw_gt_points=oracle,
        visible=torch.ones(4, dtype=torch.bool),
        active=torch.tensor([False, True, True, True]),
        config=config,
    )
    assert changed_result["decode"] > 0
    assert changed_result["bone"] > 0


def test_condition_frame_is_only_a_temporal_reference() -> None:
    codec = FrozenHandMLPAE15(CODEC_ROOT / "right_mlp15_primary.pt")
    oracle = codec.decode(torch.randn(3, 15))
    changed = oracle.clone()
    changed[0] += 1.0  # condition frame: excluded from point loss
    result = hand_geometry_losses(
        predicted_points=changed,
        oracle_points=oracle,
        raw_gt_points=oracle,
        visible=torch.ones(3, dtype=torch.bool),
        active=torch.tensor([False, True, True]),
        config=GeometryLossConfig(enabled=True),
    )
    assert result["decode"].item() == 0.0
    assert result["velocity"] > 0
