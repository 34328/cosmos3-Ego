"""V0.3 joint refresh routing, RNG isolation and completed-block output order."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.teacher_forcing import make_teacher_forcing_clean_pack
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, STATE, VIDEO, JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence
from cosmos3_joint_video_hand_pose.src.ar_v03_inference import (
    DiffusionForcingJointARSampler, refresh_noise_pack,
)


def packed_fixture(c=4, device="cpu"):
    layouts = [JointChunkLayout(2 * c + 2, 2, c), JointChunkLayout(c + 1, 2, c)]
    data = GenerationDataClean(
        batch_size=len(layouts), is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 4, x.num_video_frames, 2, 4, device=device) for x in layouts],
        x0_tokens_action=[torch.cat((torch.randn(x.num_action_rows, 57, device=device),
                                    torch.zeros(x.num_action_rows, 7, device=device)), dim=1) for x in layouts],
        fps_vision=torch.tensor([7.5] * len(layouts)), fps_action=torch.tensor([15.] * len(layouts)),
    )
    pack = pack_joint_sequence(layout=layouts, gen_data_clean=data, text_ids=[[3, 4], [5, 6]],
                               special_tokens={"eos_token_id": 1, "start_of_generation": 2},
                               timesteps=[500, 600], latent_patch_size=2, condition_frames=[(), ()])
    return layouts, make_teacher_forcing_clean_pack(pack)


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_zero_refresh_is_identity_and_does_not_consume_rng(c):
    _, packed = packed_fixture(c)
    generator = torch.Generator().manual_seed(42)
    before = generator.get_state().clone()
    assert refresh_noise_pack(packed, sigma_small=0, generator=generator,
                              video_max_timestep=1000, action_max_timestep=2000) is packed
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_joint_refresh_formula_conditions_padding_and_per_token_timesteps(c):
    layouts, packed = packed_fixture(c)
    saved = copy.deepcopy(packed)
    generator = torch.Generator().manual_seed(42)
    expected_generator = torch.Generator().manual_seed(42)
    global_rng = torch.get_rng_state().clone()
    changed = refresh_noise_pack(packed, sigma_small=.02, generator=generator,
                                video_max_timestep=1000, action_max_timestep=2000)
    assert changed.uses_single_timestep is False
    assert torch.equal(global_rng, torch.get_rng_state())
    for modality_name, maximum in (("vision", 1000), ("action", 2000)):
        original, modified = getattr(packed, modality_name), getattr(changed, modality_name)
        expected_indexes = []
        cursor = 0
        for i, layout in enumerate(layouts):
            is_action = modality_name == "action"
            roles, _, _ = layout.action_metadata() if is_action else layout.video_metadata()
            active = roles == (ACTION if is_action else VIDEO)
            rows = torch.where(active)[0]
            clean = original.tokens[i][rows, :57] if is_action else original.tokens[i][:, :, rows]
            noise = torch.randn(clean.shape, generator=expected_generator)
            expected = .98 * clean + .02 * noise
            got = modified.tokens[i][rows, :57] if is_action else modified.tokens[i][:, :, rows]
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
            conditions = ~active
            old_conditions = original.tokens[i][conditions] if is_action else original.tokens[i][:, :, conditions]
            new_conditions = modified.tokens[i][conditions] if is_action else modified.tokens[i][:, :, conditions]
            assert torch.equal(old_conditions, new_conditions)
            assert torch.equal(original.tokens[i], getattr(saved, modality_name).tokens[i])
            if is_action:
                assert not torch.count_nonzero(modified.tokens[i][:, 57:])
            patches = 1 if is_action else layout.vision_tokens
            size = active.numel() * patches
            selected = original.sequence_indexes[cursor:cursor + size][active.repeat_interleave(patches)]
            expected_indexes.append(selected)
            assert torch.equal(modified.noisy_frame_indexes[i], rows)
            assert not modified.condition_mask[i][rows].count_nonzero()
            assert modified.condition_mask[i][conditions].all()
            cursor += size
        assert torch.equal(modified.mse_loss_indexes, torch.cat(expected_indexes))
        assert torch.equal(modified.timesteps, torch.full_like(modified.timesteps, .02 * maximum))
        assert not original.timesteps.numel()
    assert changed.text_ids is packed.text_ids


@pytest.mark.parametrize("sigma", [float("nan"), float("inf"), -.01, 1.01])
def test_invalid_refresh_sigma_fails(sigma):
    _, packed = packed_fixture()
    with pytest.raises(ValueError, match="sigma_small"):
        refresh_noise_pack(packed, sigma_small=sigma, generator=torch.Generator(),
                           video_max_timestep=1000, action_max_timestep=1000)


def cache_sampler(monkeypatch, sampler_class=DiffusionForcingJointARSampler):
    from test_ar_v02_inference import fixture
    from cosmos3_joint_video_hand_pose.src import ar_v02_inference

    sampler, _, _ = fixture(c=2, future_frames=4)
    sampler.__class__ = sampler_class
    data = GenerationDataClean(batch_size=1, is_image_batch=False,
                               x0_tokens_vision=[sampler.gt_video], x0_tokens_action=[sampler.gt_action],
                               fps_vision=torch.tensor([7.5]), fps_action=torch.tensor([15.]))
    pack = pack_joint_sequence(layout=sampler.layout, gen_data_clean=data, text_ids=[3, 4],
                               special_tokens={"eos_token_id": 1, "start_of_generation": 2},
                               timesteps=500, latent_patch_size=1, condition_frames=())

    class FakeCache:
        def __init__(self, *args, **kwargs):
            self.forward_calls, self.history, self.records = 0, {}, []

        def begin(self, indexes, *, chunk, capture, include_text):
            self.chunk, self.capture = chunk, capture

        def ensure_complete(self):
            pass

    monkeypatch.setattr(ar_v02_inference, "JointKVCache", FakeCache)
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    model = sampler.model
    model.net = SimpleNamespace(num_hidden_layers=1, num_kv_heads=1, head_dim=1)
    model.ar_context = lambda *args: nullcontext()
    model._pack_input_sequence = lambda *args, **kwargs: copy.deepcopy(pack)
    model._cast_generated_tokens_to_precision = lambda packed: None
    model.rectified_flow_video = SimpleNamespace(noise_scheduler=SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=1000)))
    model.rectified_flow_action = SimpleNamespace(noise_scheduler=SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=2000)))

    def denoise(*, data_batch_packed, memory):
        p, cache = data_batch_packed, memory
        cache.forward_calls += 1
        if p.vision is None:
            return {"preds_vision": [], "preds_action": []}
        v, a = p.vision.tokens[0], p.action.tokens[0]
        condition = bool(p.action_state_mask.all())
        phase = "condition" if condition else "refresh" if cache.capture else "noisy"
        cache.records.append((cache.chunk, phase, v.clone(), a.clone(),
                              p.vision.timesteps.clone(), p.action.timesteps.clone()))
        if phase == "refresh":
            cache.history[cache.chunk] = (v.clone(), a.clone())
        if cache.capture:
            return {"preds_vision": [torch.zeros_like(v)], "preds_action": [torch.zeros_like(a)]}
        rows = (sampler.roles == ACTION) & (sampler.chunks == cache.chunk)
        sigma_v = p.vision.timesteps[0] / 1000 if p.vision.timesteps.numel() else 0
        sigma_a = p.action.timesteps[0] / 2000
        past = sum(x.mean() + y[:, :57].mean() for x, y in cache.history.values()) * .01
        pv = (v - .75) / sigma_v.clamp_min(1e-6) + past
        pa = (a - sampler.gt_action[rows]) / sigma_a.clamp_min(1e-6) + past
        pa[:, 57:] = 0
        return {"preds_vision": [pv], "preds_action": [pa]}

    model.denoise = denoise
    sampler.text, sampler.plans, sampler.gen = [[3, 4]], [], data
    sampler.memory_info = {"initial_temporal_offset": 0.0}
    return sampler


@pytest.mark.parametrize("history", ["gt", "oracle", "pred_history", "generated"])
def test_sigma_zero_sampler_is_bitwise_v02(monkeypatch, history):
    from cosmos3_joint_video_hand_pose.src.ar_v02_inference import JointARSampler
    old = cache_sampler(monkeypatch, JointARSampler)
    new = cache_sampler(monkeypatch)
    expected = old.sample(history=history, seed=42)
    got = new.sample(history=history, seed=42, sigma_small=0)
    assert all(torch.equal(a, b) for a, b in zip(expected, got))


@pytest.mark.parametrize("history", ["gt", "oracle", "pred_history", "generated"])
def test_refresh_changes_only_history_write_and_later_denoising(monkeypatch, history):
    zero, positive = cache_sampler(monkeypatch), cache_sampler(monkeypatch)
    zv, za = zero.sample(history=history, seed=42, sigma_small=0)
    pv, pa = positive.sample(history=history, seed=42, sigma_small=.02)
    first_v = zero.layout.video_indexes(1)
    first_a = zero.chunks == 1
    assert torch.equal(zv[:, :, first_v], pv[:, :, first_v])
    assert torch.equal(za[first_a], pa[first_a])
    first_zero = [r for r in zero.cache.records if r[:2] == (1, "noisy")]
    first_positive = [r for r in positive.cache.records if r[:2] == (1, "noisy")]
    assert len(first_zero) == len(first_positive) == 30
    assert all(torch.equal(a[i], b[i]) for a, b in zip(first_zero, first_positive) for i in (2, 3, 4, 5))
    assert not torch.equal(zero.cache.history[1][0], positive.cache.history[1][0])
    assert not torch.equal(zero.cache.history[1][1], positive.cache.history[1][1])
    assert not torch.equal(za[zero.chunks == 2], pa[positive.chunks == 2])
    first_condition = [r for r in positive.cache.records if r[:2] == (1, "condition")][0]
    assert not first_condition[4].numel() and not first_condition[5].numel()
    for _, phase, _, action, vt, at in positive.cache.records:
        assert not torch.count_nonzero(action[:, 57:])
        if phase == "refresh":
            assert torch.equal(vt, torch.full_like(vt, 20.))
            assert torch.equal(at, torch.full_like(at, 40.))
    assert positive.chunk_reports[0]["noisy_refresh_calls"] == 1
    assert positive.chunk_reports[0]["clean_refresh_calls"] == 0


def test_eval_parser_default_and_reference_guard():
    from cosmos3_joint_video_hand_pose.src.ar_v03_eval import parser, validate_sample_args
    required = ["sample", "--ckpt", "model", "--episodes-manifest", "ep.csv", "--segments-manifest", "segments.csv",
                "--eval-windows", "windows.json", "--output", "out", "--split", "heldout"]
    args = parser().parse_args(required)
    assert args.sigma_small == .02 and args.toml.name == "ar_v0_3.toml"
    validate_sample_args(args)
    args.no_cache = True
    with pytest.raises(ValueError, match="persistent cache"):
        validate_sample_args(args)
    args.sigma_small = 0
    validate_sample_args(args)


def test_sample_metadata_keeps_joint_sigma_visible_to_existing_endpoint_metrics(monkeypatch, tmp_path):
    import json
    import numpy as np
    import zarr
    import cosmos_framework.configs.toml_config.sft_config as sft
    from cosmos_framework.utils import distributed, lazy_config, misc
    from cosmos3_joint_video_hand_pose.src import action, ar_inference, ar_v02_overlay, dataset
    from cosmos3_joint_video_hand_pose.src import ar_v03_eval, ar_v03_inference
    from cosmos3_joint_video_hand_pose.src.ar_v03_model import EgoVerseARV03Model

    window = dict(sample_id="e:0:0:600", start=0, frames=17, seed=42)
    frozen = tmp_path / "window.json"
    frozen.write_text(json.dumps([window]))
    profile = tmp_path / "normalizer.json"
    profile.write_text("{}")
    ds_cfg = SimpleNamespace(chunk_state_normalizer=profile, future_normalizer=profile)
    config = SimpleNamespace(validate=lambda: None, freeze=lambda: None,
        dataloader_train=SimpleNamespace(dataloader=SimpleNamespace(datasets=SimpleNamespace(
            egoverse=SimpleNamespace(dataset=ds_cfg)))))
    row = dict(episode_hash="e", span_index=0, start_idx=0, end_idx=600,
               _clip_frames=17, _valid_starts=[0])
    raw_item = dict(action_source_frame_indices=torch.arange(2),
                    ar_source_keypoints_world=np.zeros((2, 2, 21, 3)),
                    ar_source_poses=np.zeros((2, 1, 7)))
    raw = SimpleNamespace(chunk_camera_mode=True, fixed_camera_mode=True, rows=[row],
        episodes={"e": dict(fps=30, abs_zarr_path="fake.zarr")}, speed_factor=.5,
        chunk_state_normalizer=object(), future_normalizer=object(),
        get_item_at_window=lambda *args, **kwargs: raw_item)
    wrapped = SimpleNamespace(dataset=raw, get_item_at_window=lambda *args, **kwargs: {})
    model = object.__new__(EgoVerseARV03Model)
    torch.nn.Module.__init__(model)
    layout = JointChunkLayout(5, 1, 4)
    sampler = SimpleNamespace(layout=layout, gt_action=torch.zeros(layout.num_action_rows, 64),
        gt_states=torch.zeros(4, 64), action_adapter=SimpleNamespace(representation="fixed_camera_wrist_local_delta_latent_v1"),
        condition_reports=[], chunk_reports=[], cache_prefill_seconds=0,
        sample=lambda **kwargs: (torch.zeros(1, 1, layout.num_video_frames, 1, 1),
                                  torch.zeros(layout.num_action_rows, 64)))
    metadata = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sft, "load_experiment_from_toml", lambda *args: config)
    monkeypatch.setattr(distributed, "init", lambda: None)
    monkeypatch.setattr(lazy_config, "instantiate", lambda *args, **kwargs: wrapped)
    monkeypatch.setattr(misc, "to", lambda value, **kwargs: value)
    monkeypatch.setattr(ar_inference, "_training_layout_batch", lambda value: value)
    monkeypatch.setattr(ar_v03_eval, "_load_bound_model", lambda *args: model)
    monkeypatch.setattr(ar_v03_inference, "DiffusionForcingJointARSampler", lambda *args, **kwargs: sampler)
    monkeypatch.setattr(action, "pose_matrices", lambda value: np.repeat(np.eye(4)[None], len(value), axis=0))
    monkeypatch.setattr(dataset, "decode_rgb_video", lambda value: torch.zeros(3, len(value), 1, 1))
    # The group needs only read indexing; keep zarr/storage outside this metadata test.
    class Group:
        attrs = dict(intrinsics=dict(front_1=np.eye(3)))
        def __getitem__(self, name):
            return np.zeros((2, 1, 1, 3), dtype=np.uint8)
    monkeypatch.setattr(zarr, "open_group", lambda *args, **kwargs: Group())
    monkeypatch.setattr(ar_v02_overlay, "decode_video_chunks", lambda *args: [])
    def save(path, **kwargs):
        metadata.append(kwargs["metadata"])
        return path
    monkeypatch.setattr(ar_v03_eval, "save_rollout", save)
    args = ar_v03_eval.parser().parse_args([
        "sample", "--ckpt", str(tmp_path / "model"), "--episodes-manifest", "ep.csv",
        "--segments-manifest", "segments.csv", "--eval-windows", str(frozen),
        "--output", str(tmp_path / "out"), "--split", "heldout", "--sigma-small", ".05",
    ])
    ar_v03_eval.sample(args)
    assert len(metadata) == 1
    meta = metadata[0]
    assert meta["sigma_small"] == meta["history_video_sigma"] == meta["history_action_sigma"] == .05
    assert meta["model_version"] == "ar_v0.3.0"
