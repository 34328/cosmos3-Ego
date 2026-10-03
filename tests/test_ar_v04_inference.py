"""Real sampler orchestration with deterministic CPU network/VAE doubles."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout, ACTION
from cosmos3_joint_video_hand_pose.src.ar_v02_overlay import replay_timeline
from cosmos3_joint_video_hand_pose.src.ar_v04_codec import (
    VIDEO_LATENT_FORMAT, gather_joint_latents, continuous_video_latents,
    decode_continuous_video_chunks,
)
from cosmos3_joint_video_hand_pose.src.ar_v04_inference import ContinuousJointARSampler
from cosmos3_joint_video_hand_pose.src.ar_v04_archive import validate_metadata, save_rollout


@pytest.mark.parametrize("frames", [2, 5, 6, 9, 10, 69])
def test_continuous_latents_roundtrip_partial_tails_and_gradient(frames):
    layout = JointChunkLayout(frames, 1, 4)
    continuous = torch.arange(frames).float().reshape(1, 1, frames, 1, 1).requires_grad_()
    packed = gather_joint_latents(layout, continuous)
    got = continuous_video_latents(layout, packed, video_latent_format=VIDEO_LATENT_FORMAT, history="generated")
    assert torch.equal(got, continuous)
    got.sum().backward()
    assert torch.equal(continuous.grad, torch.ones_like(continuous))


def test_generated_rejects_mismatched_u_but_gt_targets_keep_their_prediction():
    layout = JointChunkLayout(9, 1, 4)
    continuous = torch.arange(9).float().reshape(1, 1, 9, 1, 1)
    packed = gather_joint_latents(layout, continuous)
    packed[:, :, layout.video_indexes(2, condition=True)] += 100
    with pytest.raises(ValueError, match="previous final latent"):
        continuous_video_latents(layout, packed, video_latent_format=VIDEO_LATENT_FORMAT, history="generated")
    got = continuous_video_latents(layout, packed, video_latent_format=VIDEO_LATENT_FORMAT, history="gt")
    assert torch.equal(got, continuous)


@pytest.mark.parametrize("frames", [6, 9, 10])
def test_single_native_decode_then_rgb_slices_share_exact_boundary(frames):
    layout = JointChunkLayout(frames, 1, 4)
    latent = torch.linspace(-.7, .7, frames).reshape(1, 1, frames, 1, 1)
    seen = []
    # Cumulative dependence deliberately makes blockwise decoding unequal.
    def decode(value):
        seen.append(value.clone())
        rgb = torch.cat((value[:, :, :1], value[:, :, 1:].repeat_interleave(4, dim=2)), dim=2)
        return rgb.cumsum(2).div(frames).expand(1, 3, -1, 1, 1)
    model = SimpleNamespace(config=SimpleNamespace(video_latent_format=VIDEO_LATENT_FORMAT),
                            tensor_kwargs={"dtype": torch.float32}, decode=decode)
    chunks = decode_continuous_video_chunks(model, layout, gather_joint_latents(layout, latent),
        video_latent_format=VIDEO_LATENT_FORMAT, history="generated")
    assert len(seen) == 1 and torch.equal(seen[0], latent)
    assert [len(x) for x in chunks] == [1 + b.action_count // 2 for b in layout.boundaries]
    assert all(np.array_equal(a[-1], b[0]) for a, b in zip(chunks, chunks[1:]))
    timeline = replay_timeline(layout)
    stitched = np.concatenate([chunks[0], *[x[1:] for x in chunks[1:]]])
    for ci, fi, source in zip(timeline['generated_chunk_ids'], timeline['generated_frame_indexes'],
                              timeline['generated_source_indexes']):
        assert np.array_equal(chunks[ci - 1][fi], stitched[source // 2])


@pytest.mark.parametrize("history", ["gt", "generated", "oracle", "pred_history"])
@pytest.mark.parametrize("sigma", [0., .02])
def test_sampler_real_loop_preserves_action_kv_refresh_and_uses_latent_boundary(monkeypatch, history, sigma):
    from test_ar_v03_inference import cache_sampler
    sampler = cache_sampler(monkeypatch, ContinuousJointARSampler)
    sampler.model.config = SimpleNamespace(video_latent_format=VIDEO_LATENT_FORMAT)
    continuous = torch.arange(sampler.layout.num_frames).float().reshape(1, 1, -1, 1, 1)
    sampler.gt_video = gather_joint_latents(sampler.layout, continuous)
    sampler.gen.x0_tokens_vision = [sampler.gt_video]
    def forbidden(*args):
        raise AssertionError("sampling must never decode/re-encode a boundary")
    sampler.model.encode = sampler.model.decode = forbidden
    video, action = sampler.sample(history=history, seed=42, sigma_small=sigma)
    assert action.shape == sampler.gt_action.shape and not action[:, 57:].count_nonzero()
    assert torch.isfinite(action).all()
    assert sampler.cache.forward_calls == 1 + 32 * len(sampler.layout.boundaries)
    for b in sampler.layout.boundaries:
        condition = [r for r in sampler.cache.records if r[:2] == (b.chunk_id, "condition")][0]
        assert not condition[4].numel() and not condition[5].numel()
        assert len([r for r in sampler.cache.records if r[:2] == (b.chunk_id, "noisy")]) == 30
        assert len([r for r in sampler.cache.records if r[:2] == (b.chunk_id, "refresh")]) == 1
        if b.chunk_id > 1 and history == "generated":
            previous = sampler.layout.video_indexes(b.chunk_id - 1, condition=False)[-1:]
            assert torch.equal(condition[2], video[:, :, previous])
        rows = (sampler.roles == ACTION) & (sampler.chunks == b.chunk_id)
        assert torch.isfinite(action[rows, :57]).all()
    assert all(r['video_latent_format'] == VIDEO_LATENT_FORMAT for r in sampler.chunk_reports)
    if sigma:
        for b in sampler.layout.boundaries:
            vi = sampler.layout.video_indexes(b.chunk_id, condition=False)
            expected = sampler.gt_video[:, :, vi] if history in ('gt', 'oracle') else video[:, :, vi]
            assert not torch.equal(sampler.cache.history[b.chunk_id][0], expected)


def test_boundary_condition_is_copy_not_vae_transform():
    sampler = object.__new__(ContinuousJointARSampler)
    block = torch.randn(1, 3, 5, 2, 2)
    got = sampler._next_condition_video(block)
    assert torch.equal(got, block[:, :, -1:])
    got.zero_()
    assert block[:, :, -1:].count_nonzero()


@pytest.mark.parametrize("format", [None, "joint_chunk_cond_v1", "block_reset"])
def test_legacy_format_rejected_before_model_or_decode(format):
    model = SimpleNamespace(config=SimpleNamespace(video_latent_format=format))
    with pytest.raises(ValueError, match="legacy block-reset"):
        ContinuousJointARSampler(model, {})
    with pytest.raises(ValueError, match="legacy block-reset"):
        continuous_video_latents(None, None, video_latent_format=format, history="generated")


def test_legacy_archive_cannot_enter_v04_output(tmp_path):
    with pytest.raises(ValueError, match="legacy block-reset"):
        save_rollout(tmp_path / 'bad.npz', metadata={'model_version':'ar_v0.3.1'})
    assert not (tmp_path / 'bad.npz').exists()
    validate_metadata(dict(video_latent_format=VIDEO_LATENT_FORMAT, model_version='ar_v0.4',
                           video_decode_mode='continuous_full_sequence'))


def test_eval_default_selects_only_new_v04_recipe():
    from cosmos3_joint_video_hand_pose.src.ar_v04_eval import parser
    args = parser().parse_args(['sample', '--ckpt', 'model', '--training-snapshot', 'config.yaml',
        '--episodes-manifest', 'e', '--segments-manifest', 's', '--eval-windows', 'w', '--output', 'o', '--split', 'test'])
    assert args.toml.name == 'ar_v0_4.toml'
    assert args.sigma_small == .02 and args.video_shift == 5
