import json

import pytest
import torch

from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import audited_frame_ranges, sha256
from cosmos3_joint_video_hand_pose.scripts.prepare_wrist_local_pca import fit_pca, passes
from cosmos3_joint_video_hand_pose.src.codec_fixed_camera import (
    FrozenFixedCameraHandCodec, FrozenFixedCameraHandAE15, INPUT_FRAME, REPRESENTATION,
)


def provenance(split="train"):
    return dict(split=split, episode_ids=[split], manifest_sha256="a"*64, episodes_sha256="b"*64,
                source_hashes=dict(train_episodes="b"*64, heldout_episodes="b"*64,
                                   train_segments="c"*64, heldout_segments="d"*64),
                representation=REPRESENTATION, input_frame=INPUT_FRAME,
                source_windows=[dict(episode=split, window="window", start=0, span=33)])


@pytest.fixture
def pca(tmp_path):
    rng = torch.Generator().manual_seed(7)
    basis = torch.linalg.qr(torch.randn(60, 15, generator=rng)).Q
    q = ((torch.randn(256, 15, generator=rng)*.02) @ basis.T + .03).reshape(-1, 20, 3)
    payload = fit_pca(q, provenance(), "right")
    path = tmp_path / "right_pca15.pt"
    torch.save(payload, path)
    return path, q, payload


def test_physical_pca_formula_reconstruction_and_deterministic_fit(pca):
    path, q, payload = pca
    codec = FrozenFixedCameraHandCodec(path, allow_unvalidated=True)
    flat = q.flatten(1)
    expected = ((flat-codec.input_mean) @ codec.components.T-codec.latent_mean)/codec.latent_std
    torch.testing.assert_close(codec.encode(q), expected)
    torch.testing.assert_close(codec.decode(expected), q, atol=1e-7, rtol=1e-5)
    z = torch.randn(10, 15)
    torch.testing.assert_close(codec.encode(codec.decode(z)), z, atol=3e-6, rtol=1e-5)
    assert torch.equal(codec.decode(z), codec.decode(z))
    assert payload["fitting"]["explained_variance_ratio_sum"] > .99999
    assert not list(codec.parameters())
    same = fit_pca(q, provenance(), "right")
    assert torch.equal(same["state_dict"]["components"], payload["state_dict"]["components"])
    assert payload["fit"]["data_sha256"] == same["fit"]["data_sha256"]
    with pytest.raises(ValueError, match="train-only"):
        fit_pca(q, provenance("heldout"), "right")
    with pytest.raises(ValueError, match="MLP AE"):
        FrozenFixedCameraHandAE15(path, allow_unvalidated=True)


def test_pca_components_tampering_rejected(pca):
    path, _, payload = pca
    payload["state_dict"]["components"][0] *= 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="orthonormal"):
        FrozenFixedCameraHandCodec(path, allow_unvalidated=True)


def test_pca_strict_sidecar_requires_repeat_decode_and_exact_hash(pca):
    path, _, payload = pca
    with pytest.raises(ValueError, match="sidecar"):
        FrozenFixedCameraHandCodec(path)
    metrics = dict(reconstruction=dict(mean_mm=1, p95_mm=2), trajectory_count=1,
                   repeat_decode=dict(passed=True, repeats=10, max_abs_difference_m=0),
                   reanchor_invariance=dict(passed=True, latent_exact=True, no_reencode=True,
                       full_chunks=100, tail_frames=8, max_point_error_m=1e-6,
                       nonzero_wrist_rotation=True, nonzero_camera_rotation=True))
    report = dict(schema_version=2, architecture=payload["architecture"], input_frame=INPUT_FRAME,
                  representation=REPRESENTATION, checkpoint_sha256=sha256(path), passed=True,
                  heldout_episode_ids=["heldout"], heldout_sample_count=33,
                  heldout_provenance=provenance("heldout"), metrics=metrics,
                  thresholds=dict(reconstruction_mean_mm=5, reconstruction_p95_mm=15))
    sidecar = path.with_suffix(".validation.json")
    sidecar.write_text(json.dumps(report))
    assert FrozenFixedCameraHandCodec(path).checkpoint_sha256 == sha256(path)
    metrics["repeat_decode"]["max_abs_difference_m"] = 1e-8
    sidecar.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="metrics"):
        FrozenFixedCameraHandCodec(path)
    metrics["repeat_decode"]["max_abs_difference_m"] = 0
    report["checkpoint_sha256"] = "0"*64
    sidecar.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="stale"):
        FrozenFixedCameraHandCodec(path)


def test_audited_union_never_duplicates_frames_or_bridges_gaps():
    result = audited_frame_ranges([
        ("a", {"starts":[0, 1, 2], "span":33}),
        ("b", {"starts":[30], "span":33}),
        ("c", {"starts":[100], "span":33}),
    ])
    assert [(x[0], x[1]) for x in result] == [(0, 63), (100, 133)]
    assert set(result[0][2]) == {"a", "b"}


@pytest.mark.parametrize("mean,p95,passed", [(5,15,True), (5.0001,15,False), (5,15.0001,False)])
def test_quality_thresholds_not_relaxed(mean, p95, passed):
    assert passes(dict(reconstruction=dict(mean_mm=mean,p95_mm=p95),
                       repeat_decode=dict(passed=True),reanchor_invariance=dict(passed=True))) == passed
