import json
from pathlib import Path

import torch

from cosmos3_joint_video_hand_pose.src.codec import FrozenHandMLPAE15
from cosmos3_joint_video_hand_pose.src.normalization import PiecewiseAsinhNormalizer


ROOT = Path(__file__).resolve().parents[1]


def test_codec_is_frozen_and_has_reversible_shapes():
    codec = FrozenHandMLPAE15(
        ROOT
        / "cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v2_4/option_b_mlp15/right_mlp15_primary.pt"
    )
    points = torch.randn(4, 20, 3) * 0.05
    latent = codec.encode(points)
    decoded = codec.decode(latent)
    assert latent.shape == (4, 15)
    assert decoded.shape == (4, 20, 3)
    assert all(not parameter.requires_grad for parameter in codec.parameters())


def test_piecewise_asinh_normalizer_round_trip():
    normalizer = PiecewiseAsinhNormalizer(
        ROOT
        / "cosmos3_joint_video_hand_pose/artifacts/cosmos3_action_contract/v2/normalizers/future_delta_normalizer.json"
    )
    values = torch.randn(32, 27)
    restored = normalizer.denormalize(normalizer.normalize(values))
    torch.testing.assert_close(restored, values, atol=2e-5, rtol=2e-5)


def test_default_action_normalizers_use_overfit_v0_0_contract():
    from cosmos3_joint_video_hand_pose.src.action import DEFAULT_FUTURE_NORMALIZER, DEFAULT_STATE_NORMALIZER

    assert DEFAULT_STATE_NORMALIZER.parts[-3:] == ("v2", "normalizers", "state_normalizer.json")
    assert DEFAULT_FUTURE_NORMALIZER.parts[-3:] == ("v2", "normalizers", "future_delta_normalizer.json")


def test_piecewise_asinh_normalizer_accepts_explicit_center_scale(tmp_path):
    path = tmp_path / "normalizer.json"
    center = torch.linspace(-0.2, 0.2, 27)
    scale = torch.linspace(0.01, 0.27, 27)
    path.write_text(
        json.dumps(
            {
                "method": "piecewise_asinh_rot",
                "beta": 1.0,
                "stats": {"center": center.tolist(), "scale": scale.tolist()},
            }
        ),
        encoding="utf-8",
    )
    normalizer = PiecewiseAsinhNormalizer(path)
    values = torch.randn(16, 27) * scale + center
    restored = normalizer.denormalize(normalizer.normalize(values))
    torch.testing.assert_close(normalizer.center, center)
    torch.testing.assert_close(normalizer.scale, scale)
    torch.testing.assert_close(restored, values, atol=2e-5, rtol=2e-5)
