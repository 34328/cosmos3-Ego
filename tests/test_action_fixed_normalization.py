import copy

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from pathlib import Path

from cosmos3_joint_video_hand_pose.src.action_fixed_normalization import (
    FixedCameraNormalizer, fit_fixed_normalizer, profile_sha256,
)


def profile(kind="future"):
    x = np.random.default_rng(5).normal(size=(100, 57))
    x[:, 18:33] *= 7
    if kind == "state":
        x[:, :9] = 0
    return x, fit_fixed_normalizer(x, kind=kind, codec_sha256=("a" * 64, "b" * 64), manifest_sha256="c" * 64)


@pytest.mark.parametrize("kind", ["state", "future"])
def test_roundtrip_and_full_channel_statistics(kind):
    x, p = profile(kind)
    n = FixedCameraNormalizer(p, kind=kind)
    source = torch.tensor(x).reshape(2, 50, 57)
    encoded = n.normalize(source)
    torch.testing.assert_close(n.denormalize(encoded), source, atol=1e-12, rtol=1e-12)
    assert n.codec_sha256 == ("a" * 64, "b" * 64)
    expected = (np.quantile(x, .99, axis=0) + np.quantile(x, .01, axis=0))/2
    np.testing.assert_allclose(p["stats"]["center"][18:33], expected[18:33])
    if kind == "state":
        assert encoded[..., :9].count_nonzero() == 0


@pytest.mark.parametrize("field,value", [("representation", "legacy"), ("representation", "fixed_camera_delta_latent_v1"), ("schema", "old"), ("kind", "bad"), ("chunk_size", 1), ("tokens_per_latent", 4), ("split", "heldout"), ("frozen", False), ("codec_sha256", ["old", "old"]), ("manifest_sha256", ""), ("method", "old")])
def test_reject_incompatible_profiles(field, value):
    _, p = profile()
    p[field] = value
    p["profile_sha256"] = profile_sha256(p)
    with pytest.raises(ValueError):
        FixedCameraNormalizer(p)


def test_integrity_and_pair_identity():
    _, p = profile()
    with pytest.raises(ValueError, match="kind"):
        FixedCameraNormalizer(p, kind="state")
    with pytest.raises(ValueError, match="identity"):
        FixedCameraNormalizer(p, codec_sha256=("b" * 64, "a" * 64))
    p["stats"]["center"][0] += 1
    with pytest.raises(ValueError, match="integrity"):
        FixedCameraNormalizer(p)


def test_state_rejects_nonzero_camera_and_future_accepts_it():
    x, p = profile("state")
    x[0, 0] = 1
    with pytest.raises(ValueError, match="zero"):
        FixedCameraNormalizer(p).normalize(torch.tensor(x))
    with pytest.raises(ValueError, match="zero"):
        fit_fixed_normalizer(x, kind="state", codec_sha256=("a"*64, "b"*64), manifest_sha256="c"*64)


def test_constant_channels_finite_and_gradients_preserved():
    p = fit_fixed_normalizer(np.zeros((2, 57)), kind="future", codec_sha256=("a"*64, "b"*64), manifest_sha256="c"*64)
    n = FixedCameraNormalizer(p)
    x = torch.zeros(3, 57, requires_grad=True)
    n.denormalize(n.normalize(x)).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_piecewise_tails_quantiles_and_unit_floors():
    p = fit_fixed_normalizer(np.zeros((5,57)), kind="future", codec_sha256=("a"*64,"b"*64), manifest_sha256="c"*64)
    assert p["method"] == "piecewise_asinh_per_channel_v1"
    assert p["scale_floor"][:9] == [.01]*3 + [.05]*6
    assert p["scale_floor"][18:33] == [.01]*15
    assert p["near_constant_channels"] == list(range(57))
    assert p["stats"]["q01"] == p["stats"]["q99"] == [0.]*57
    n = FixedCameraNormalizer(p)
    z = torch.tensor([-5., -1., 0., 1., 5.], dtype=torch.float64)[:,None].expand(-1,57)
    physical = z*n.scale + n.center
    expected = torch.where(z.abs() <= 1, z, z.sign()*(1+torch.asinh((z.abs()-1).clamp_min(0))))
    torch.testing.assert_close(n.normalize(physical), expected)
    torch.testing.assert_close(n.denormalize(expected), physical)


@pytest.mark.parametrize("sides", [("left", "right"), ("right", "right"), (None, "left"), ("right", None)])
def test_codec_loader_rejects_swapped_or_missing_side(monkeypatch, sides):
    from cosmos3_joint_video_hand_pose.src import codec_fixed_camera
    from cosmos3_joint_video_hand_pose.src.action_fixed_normalization import load_fixed_codecs
    codecs = {name: SimpleNamespace(metadata={} if side is None else {"side": side})
              for name, side in zip(("r.pt", "l.pt"), sides)}
    monkeypatch.setattr(codec_fixed_camera, "FrozenFixedCameraHandCodec", lambda path: codecs[path])
    with pytest.raises(ValueError, match="codec side mismatch"):
        load_fixed_codecs(("r.pt", "l.pt"))


def test_codec_loader_preserves_right_left_order(monkeypatch):
    from cosmos3_joint_video_hand_pose.src import codec_fixed_camera
    from cosmos3_joint_video_hand_pose.src.action_fixed_normalization import load_fixed_codecs
    codecs = {name: SimpleNamespace(metadata={"side": side}, checkpoint_sha256=digest*64)
              for name, side, digest in (("r.pt", "right", "a"), ("l.pt", "left", "b"))}
    monkeypatch.setattr(codec_fixed_camera, "FrozenFixedCameraHandCodec", lambda path: codecs[path])
    pair = load_fixed_codecs(("r.pt", "l.pt"))
    assert pair == (codecs["r.pt"], codecs["l.pt"])
    assert tuple(c.checkpoint_sha256 for c in pair) == ("a"*64, "b"*64)
    assert pair[0].checkpoint_path == str(Path("r.pt").resolve())
