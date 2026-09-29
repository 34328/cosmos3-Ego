from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_chunk_state import (
    ChunkCameraState,
    ChunkCameraStateNormalizer,
    encode_chunk_camera_state,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_inference import JointARSampler, assert_numerically_close
from cosmos3_joint_video_hand_pose.src.ar_v02_evaluation import evaluate_joint_actions, decode_joint_actions
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, CONDITION_VIDEO, STATE, VIDEO, JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_overlay import (
    replay_timeline,
    image_pixel_transform,
    transformed_intrinsics,
    project_chunk_hands,
    render_joint_overlay,
)


def fixture(c=2, future_frames=5):
    # Real 18D transform in memory; file/profile validation is tested by the data owner.
    normalizer = object.__new__(ChunkCameraStateNormalizer)
    normalizer.center, normalizer.scale, normalizer.beta = torch.zeros(18), torch.ones(18), 1.0
    future_normalizer = SimpleNamespace(normalize=lambda x: x, denormalize=lambda x: x)
    layout = JointChunkLayout(1 + future_frames, 1, c)
    boundary_states = []
    for j in range(future_frames):
        rigid = torch.eye(4).repeat(3, 1, 1)
        rigid[1, 0, 3], rigid[2, 0, 3] = 0.3 + j * 8 * 0.01, -0.3 + j * 8 * 0.01
        rigid[1:, 2, 3] = 1
        state = ChunkCameraState(j * 8, rigid, torch.full((2, 15), j * 8 * 0.001))
        boundary_states.append(torch.nn.functional.pad(encode_chunk_camera_state(state, normalizer), (0, 7)))
    states = torch.stack(boundary_states)
    future = torch.zeros(future_frames * 8, 64)
    for start, translation in ((0, 0.01), (9, 0.02), (33, 0.02)):
        future[:, start] = translation
        future[:, start + 3] = future[:, start + 7] = 1
    future[:, 18:33] = future[:, 42:57] = torch.arange(1, len(future) + 1)[:, None] * 0.001
    payload, _ = layout.assemble_action(future, states, torch.ones(len(future), 2, dtype=torch.bool))
    sampler = object.__new__(JointARSampler)
    sampler.layout, sampler.chunk_size, sampler.source_fps = layout, c, 30.0
    sampler.gt_video = torch.arange(layout.num_video_frames).float()[None, None, :, None, None] + 2
    sampler.gt_action, sampler.gt_states = payload, states
    sampler.state_normalizer, sampler.future_normalizer = normalizer, future_normalizer
    from cosmos3_joint_video_hand_pose.src.action_representation import ActionRepresentationAdapter
    sampler.action_adapter = ActionRepresentationAdapter(normalizer, future_normalizer)
    sampler.roles, sampler.chunks, sampler.sources = layout.action_metadata()
    decoded_blocks = []

    def decode(block):
        decoded_blocks.append(block.clone())
        last = block.sum(2, keepdim=True)
        return last.expand(-1, -1, 1 + 4 * (block.shape[2] - 1), -1, -1).clone()

    sampler.model = SimpleNamespace(tensor_kwargs={"dtype": torch.float32}, decode=decode, encode=lambda x: x * 0.1)
    return sampler, future, decoded_blocks


def perfect_forward(sampler, desired_future, history, calls):
    desired = sampler.gt_action.clone()
    desired[sampler.roles == ACTION] = desired_future
    vr, vc, _ = sampler.layout.video_metadata()

    def forward(video, action, *, first_noisy, end, video_sigmas, action_sigmas):
        b = next(b for b in sampler.layout.boundaries if b.latent_start == first_noisy)
        prefix = JointChunkLayout(end, 1, sampler.chunk_size)
        nv, n = prefix.num_video_frames, prefix.num_action_rows
        u = sampler.layout.video_indexes(b.chunk_id, condition=True)
        if history != "generated" or b.chunk_id == 1:
            torch.testing.assert_close(video[:, :, u], sampler.gt_video[:, :, u], rtol=0, atol=0)
            states = (sampler.roles == STATE) & (sampler.chunks == b.chunk_id)
            torch.testing.assert_close(action[states], sampler.gt_action[states], rtol=0, atol=0)
        if history in ("gt", "oracle"):
            torch.testing.assert_close(video[:, :, vc < b.chunk_id], sampler.gt_video[:, :, vc < b.chunk_id])
            torch.testing.assert_close(
                action[sampler.chunks < b.chunk_id], sampler.gt_action[sampler.chunks < b.chunk_id]
            )
        assert not video_sigmas[vr == CONDITION_VIDEO].count_nonzero()
        assert not action_sigmas[sampler.roles == STATE].count_nonzero()
        assert not action[:, 57:].count_nonzero()
        pv = (video[:, :, :nv] - 0.75) / video_sigmas[:nv].clamp_min(1e-6)[None, None, :, None, None]
        pa = (action[:n] - desired[:n]) / action_sigmas[:n].clamp_min(1e-6)[:, None]
        calls.append((b.chunk_id, video.clone(), action.clone()))
        return pv, pa

    return forward


@pytest.mark.parametrize("c,tail", [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("history", ["gt", "oracle", "pred_history", "generated"])
def test_thirty_steps_roles_history_and_partial_tail(c, tail, history):
    sampler, future, decoded_blocks = fixture(c, c * 2 + tail)
    desired = future.clone()
    desired[:, 9] += 0.003
    desired[:, 33] -= 0.002
    calls = []
    sampler.forward = perfect_forward(sampler, desired, history, calls)
    sv = torch.linspace(1, 0, 31)
    sa = sv.square()
    video, action = sampler.sample(history=history, use_cache=False, video_schedule=sv, action_schedule=sa)
    assert len(calls) == 30 * len(sampler.layout.boundaries)
    _, output, indexes = sampler.layout.unpack_action(action)
    torch.testing.assert_close(output, desired, atol=1e-6, rtol=1e-5)
    assert torch.equal(indexes, torch.arange(1, len(future) + 1))
    vr, _, _ = sampler.layout.video_metadata()
    expected = (
        sampler.gt_video[:, :, vr == VIDEO] if history == "oracle" else torch.full_like(video[:, :, vr == VIDEO], 0.75)
    )
    torch.testing.assert_close(video[:, :, vr == VIDEO], expected, atol=1e-6, rtol=1e-5)
    assert not action[:, 57:].count_nonzero()
    if history == "generated":
        assert len(decoded_blocks) == len(sampler.layout.boundaries) - 1
        for block, b in zip(decoded_blocks, sampler.layout.boundaries[1:]):
            previous = sampler.layout.video_indexes(b.chunk_id - 1)
            torch.testing.assert_close(block, video[:, :, previous])
            u = sampler.layout.video_indexes(b.chunk_id, True)
            torch.testing.assert_close(video[:, :, u], block.sum(2, keepdim=True) * 0.1)
        # Every generated state has zero camera, including after camera motion.
        assert not action[sampler.roles == STATE, :9].count_nonzero()
    else:
        assert decoded_blocks == []
    assert sampler.condition_reports[-1]["boundary_time"] == sampler.layout.boundaries[-1].source_start / 30
    result = evaluate_joint_actions(
        sampler.layout,
        action,
        future,
        sampler.gt_states,
        state_normalizer=sampler.state_normalizer,
        future_normalizer=sampler.future_normalizer,
        history=history,
    )
    assert result["layout_version"] == "joint_chunk_cond_v1"
    assert result["cumulative_drift_reported"] == (history == "generated")
    assert result["local_metrics_include_history_drift"] == (history == "generated")
    assert result["boundary_state_reset_to_gt"] == (history != "generated")
    assert result["metric_frame"] == "gt_chunk_camera"
    if history != "generated":
        for b, record in zip(sampler.layout.boundaries, result["chunks"]):
            assert record["local_right_wrist_end_mm"] == pytest.approx(b.action_count * 3, abs=0.002)
            assert not any(k.startswith("cumulative_") for k in record)
    with pytest.raises(ValueError, match="exactly 30"):
        sampler.sample(steps=20, use_cache=False)


def test_generated_metrics_preserve_boundary_error_in_gt_coordinates():
    sampler, future, _ = fixture(c=2, future_frames=4)
    shifted = sampler.gt_action.clone()
    shifted[sampler.roles == STATE, 9] += 0.1  # A carried wrist-state error, not an action error.
    kwargs = dict(state_normalizer=sampler.state_normalizer, future_normalizer=sampler.future_normalizer)
    generated = evaluate_joint_actions(
        sampler.layout, shifted, future, sampler.gt_states, history="generated", **kwargs
    )
    gt = evaluate_joint_actions(sampler.layout, shifted, future, sampler.gt_states, history="gt", **kwargs)
    assert generated["local_metric_scope"] == "rollout_error_in_gt_chunk_camera_including_history_drift"
    assert gt["local_metric_scope"] == "chunk_error_from_gt_boundary_state"
    assert generated["chunks"][0]["local_right_wrist_end_mm"] == pytest.approx(100, abs=0.01)
    assert gt["chunks"][0]["local_right_wrist_end_mm"] < 0.001


def test_generated_rollout_does_not_read_future_gt():
    a, future, _ = fixture()
    b, _, _ = fixture()
    b.gt_video[:, :, 1:] = 9999
    b.gt_states[1:, 9] += 30
    b.gt_action[1:] = 777
    for sampler in (a, b):
        sampler.forward = perfect_forward(sampler, future, "generated", [])
    av, aa = a.sample(history="generated", use_cache=False, seed=72)
    bv, ba = b.sample(history="generated", use_cache=False, seed=72)
    torch.testing.assert_close(av, bv, rtol=0, atol=0)
    torch.testing.assert_close(aa, ba, rtol=0, atol=0)


def test_pred_history_keeps_gt_conditions_but_preserves_predictions():
    sampler, future, _ = fixture()
    desired = future.clone()
    desired[:, 9] += 0.004
    calls = []
    sampler.forward = perfect_forward(sampler, desired, "pred_history", calls)
    _, action = sampler.sample(history="pred_history", use_cache=False)
    b = sampler.layout.boundaries[1]
    _, _, second_action = calls[30]
    past = (sampler.roles == ACTION) & (sampler.chunks == 1)
    torch.testing.assert_close(second_action[past], action[past])
    assert not torch.allclose(second_action[past], sampler.gt_action[past])


@pytest.mark.parametrize("fp32", [True, False])
def test_numerical_threshold_zero_and_relative_limits(fp32):
    assert_numerically_close(torch.tensor([1e-7]), torch.zeros(1), fp32=fp32)
    with pytest.raises(AssertionError):
        assert_numerically_close(torch.tensor([float("nan")]), torch.zeros(1), fp32=fp32)
    with pytest.raises(AssertionError):
        assert_numerically_close(torch.tensor([2.0]), torch.ones(1), fp32=fp32)


def test_timeline_source_time_and_intrinsics():
    layout = JointChunkLayout(6, 1, 2)
    real = replay_timeline(layout, source_offset=90)
    model = replay_timeline(layout, source_offset=90, mode="model_time")
    assert real["output_fps"] == 30 and model["output_fps"] == 15
    assert real["fps_video"] == 7.5 and real["fps_action"] == 15
    np.testing.assert_array_equal(real["source_indexes"], np.arange(90, 131))
    np.testing.assert_array_equal(real["source_indexes"], model["source_indexes"])
    assert real["source_times"][0] == 3
    assert real["action_rows"].tolist() == list(range(-1, 40))
    # At b+1 retain the previous prediction; never replace it with the next GT U.
    assert real["chunk_ids"][17] == 2 and real["generated_frame_indexes"][17] == 8
    assert real["generated_chunk_ids"][17] == 1
    assert real["generated_source_indexes"][17] == 106 and real["held_background"][17]
    k = np.array([[100, 0, 320], [0, 100, 180], [0, 0, 1.0]])
    affine = image_pixel_transform(crop_xywh=(20, 10, 600, 340), resized_wh=(300, 170), pad_xy=(0, 4))
    adjusted = transformed_intrinsics(k, affine)
    np.testing.assert_allclose(adjusted[:2, 2], np.array([149.75, 88.75]))
    points = np.zeros((1, 2, 21, 3))
    points[..., 2] = 1
    camera = np.eye(4)[None]
    before, _ = project_chunk_hands(points, camera, k)
    after, _ = project_chunk_hands(points, camera, adjusted)
    homogeneous = np.concatenate((before, np.ones((*before.shape[:-1], 1))), -1)
    np.testing.assert_allclose(after, (homogeneous @ affine.T)[..., :2])
    camera[:, 0, 3] = 0.1
    moved, _ = project_chunk_hands(points, camera, k)
    np.testing.assert_allclose(before[..., 0] - moved[..., 0], 10)


def test_chunk_camera_reset_and_overlay_dense_actions():
    sampler, future, _ = fixture(c=2, future_frames=3)
    decoded, _ = decode_joint_actions(
        sampler.layout,
        sampler.gt_action,
        future,
        sampler.gt_states,
        state_normalizer=sampler.state_normalizer,
        future_normalizer=sampler.future_normalizer,
    )
    for item in decoded:
        torch.testing.assert_close(item["predicted_rigid"], item["gt_rigid"], atol=1e-6, rtol=1e-5)
        # Chunk 2 camera starts integrating from identity, not accumulated camera.
        assert item["gt_rigid"][0, 0, 0, 3] == pytest.approx(0.01)
    codec = SimpleNamespace(decode=lambda x: torch.zeros(len(x), 20, 3))
    gt = np.zeros((len(future) + 1, 48, 64, 3), dtype=np.uint8)
    rgb = [np.zeros((1 + b.action_count // 2, 48, 64, 3), dtype=np.uint8) for b in sampler.layout.boundaries]
    frames, meta = render_joint_overlay(
        sampler.layout,
        sampler.gt_action,
        future,
        sampler.gt_states,
        gt_rgb=gt,
        generated_rgb_chunks=rgb,
        intrinsics=np.array([[30, 0, 32], [0, 30, 24], [0, 0, 1.0]]),
        gt_pixel_transform=np.eye(3),
        generated_pixel_transform=np.eye(3),
        state_normalizer=sampler.state_normalizer,
        future_normalizer=sampler.future_normalizer,
        hand_codecs=(codec, codec),
    )
    assert frames.shape == (25, 48, 128, 3)
    assert int(meta["consistency_eligible"].sum()) == 12
    assert int(meta["held_background"].sum()) == 12


@pytest.mark.parametrize("history", ["gt", "oracle", "pred_history", "generated"])
def test_sampler_cache_phase_counts_and_refresh_policy(history):
    import contextlib

    sampler, future, _ = fixture()
    reference, _, _ = fixture()
    desired = future.clone()
    desired[:, 9] += 0.003
    desired_payload = sampler.gt_action.clone()
    desired_payload[sampler.roles == ACTION] = desired
    reference.forward = perfect_forward(reference, desired, history, [])
    expected_video, expected_action = reference.sample(history=history, use_cache=False, seed=31)
    sampler.model.net = SimpleNamespace(num_hidden_layers=2, num_kv_heads=1, head_dim=2)
    sampler.model.ar_context = lambda *args: contextlib.nullcontext()
    sampler.model._pack_input_sequence = lambda *args, **kwargs: SimpleNamespace()
    sampler.plans, sampler.text, sampler.gen = [], [], None
    sampler.memory_info = {"initial_temporal_offset": 0}
    phases = []

    def cached(video, action, indexes, *, chunk, phase, video_sigma=0.0, action_sigma=0.0):
        sampler.cache.forward_calls += 1
        phases.append((chunk, phase))
        if phase == "text":
            assert len(indexes) == 0 and chunk == 0
            return {}
        vi = sampler.layout.video_indexes(chunk, False)
        rows = torch.where((sampler.roles == ACTION) & (sampler.chunks == chunk))[0]
        if phase == "condition":
            assert torch.equal(indexes, sampler.layout.condition_prefill_indexes(chunk))
            return {}
        if phase == "refresh":
            expected_a = sampler.gt_action[rows] if history in ("gt", "oracle") else desired_payload[rows]
            torch.testing.assert_close(action[rows], expected_a, atol=1e-6, rtol=1e-5)
            expected_v = (
                sampler.gt_video[:, :, vi] if history in ("gt", "oracle") else torch.full_like(video[:, :, vi], 0.75)
            )
            torch.testing.assert_close(video[:, :, vi], expected_v, atol=1e-6, rtol=1e-5)
            return {}
        return {
            "preds_vision": [(video[:, :, vi] - 0.75) / max(float(video_sigma), 1e-6)],
            "preds_action": [(action[rows] - desired_payload[rows]) / float(action_sigma)],
        }

    sampler._cache_forward = cached
    video, action = sampler.sample(history=history, use_cache=True, seed=31)
    torch.testing.assert_close(video, expected_video, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(action, expected_action, atol=1e-6, rtol=1e-5)
    assert phases[0] == (0, "text")
    for b in sampler.layout.boundaries:
        current = [phase for chunk, phase in phases if chunk == b.chunk_id]
        assert current == ["condition"] + ["noisy"] * 30 + ["refresh"]
    assert all(r["forward_calls"] == 32 for r in sampler.chunk_reports)
    assert sampler.cache.forward_calls == 1 + 32 * len(sampler.layout.boundaries)


@pytest.fixture
def cli_rollout(tmp_path):
    import hashlib, json
    from cosmos3_joint_video_hand_pose.src.ar_chunk_state import state_profile_sha256
    from cosmos3_joint_video_hand_pose.src.ar_v02_eval import save_rollout

    sampler, future, _ = fixture(c=2, future_frames=3)
    state_path = tmp_path / "state.json"
    profile = dict(
        schema="ar_v02_chunk_camera_state_v2",
        layout_version="joint_chunk_cond_v1",
        split="train",
        frozen=True,
        method="piecewise_asinh_rot",
        camera_encoding="zero_identity",
        manifest_sha256="a" * 64,
        fit_samples_sha256="b" * 64,
        beta=1.0,
        stats=dict(center=[0.0] * 18, scale=[1.0] * 18),
    )
    profile["profile_sha256"] = state_profile_sha256(profile)
    state_path.write_text(json.dumps(profile))
    future_path = tmp_path / "future.json"
    future_path.write_text(
        json.dumps(dict(method="piecewise_asinh_rot", beta=1.0, stats=dict(center=[0.0] * 27, scale=[1.0] * 27)))
    )

    def artifact(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    metadata = dict(
        sample_id="episode:0:0:100",
        episode_id="episode",
        history="gt",
        seed=42,
        source_offset=90,
        source_fps=30.0,
        speed_factor=0.5,
        state_normalizer=artifact(state_path),
        future_normalizer=artifact(future_path),
    )
    gt = np.broadcast_to(np.arange(25, dtype=np.uint8)[:, None, None, None], (25, 48, 64, 3)).copy()
    rgb = [gt[b.source_start : b.source_stop + 1 : 2].copy() for b in sampler.layout.boundaries]
    archive = save_rollout(
        tmp_path / "rollout.npz",
        layout=sampler.layout,
        predicted_action=sampler.gt_action,
        gt_future=future,
        boundary_states=sampler.gt_states,
        metadata=metadata,
        gt_rgb=gt,
        generated_rgb_chunks=rgb,
        intrinsics=np.array([[30, 0, 32], [0, 30, 24], [0, 0, 1.0]]),
        gt_pixel_transform=np.eye(3),
        generated_pixel_transform=np.eye(3),
    )
    codec = tmp_path / "hand.pt"
    encoder = torch.nn.Sequential(
        torch.nn.Linear(60, 64), torch.nn.SiLU(), torch.nn.Linear(64, 32), torch.nn.SiLU(), torch.nn.Linear(32, 15)
    )
    decoder = torch.nn.Sequential(
        torch.nn.Linear(15, 32), torch.nn.SiLU(), torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 60)
    )
    weights = {"encoder." + k: v for k, v in encoder.state_dict().items()}
    weights.update({"decoder." + k: v for k, v in decoder.state_dict().items()})
    weights.update(mean=torch.zeros(60), std=torch.ones(60))
    torch.save(
        dict(
            architecture="60-64-SiLU-32-SiLU-15 / 15-32-SiLU-64-SiLU-60",
            state_dict=weights,
            latent_mean=torch.zeros(15),
            latent_std=torch.ones(15),
        ),
        codec,
    )
    return archive, codec


def _cli(*args):
    import os, subprocess, sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(root) + ":" + str(root / "packages/cosmos3"),
    )
    return subprocess.run(
        [sys.executable, "-m", "cosmos3_joint_video_hand_pose.src.ar_v02_eval", *map(str, args)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_real_process_metrics_and_mp4(cli_rollout, tmp_path):
    import json
    import imageio.v2 as imageio

    archive, codec = cli_rollout
    report = tmp_path / "metrics.json"
    run = _cli("evaluate", "--input", archive, "--output", report, "--right-codec", codec, "--left-codec", codec)
    assert run.returncode == 0, run.stdout + run.stderr
    results = json.loads(report.read_text())
    assert results["bounded_control_api"] is False
    result = results["samples"][0]
    assert result["metrics"]["layout_version"] == "joint_chunk_cond_v1"
    assert result["metrics"]["hand_metrics_available"]
    for item in result["metrics"]["chunks"]:
        assert item["local_right_wrist_end_mm"] < 0.001
        assert item["local_right_mpjpe_mm"] < 0.001
        assert item["video_exact_match"] and item["video_psnr_db"] is None
    mp4 = tmp_path / "overlay.mp4"
    run = _cli("overlay", "--input", archive, "--output", mp4, "--right-codec", codec, "--left-codec", codec)
    assert run.returncode == 0, run.stdout + run.stderr
    mapping = json.loads(mp4.with_suffix(".json").read_text())
    assert mapping["source_indexes"] == list(range(90, 115))
    assert mapping["output_fps"] == 30 and mapping["fps_video"] == 7.5
    assert sum(mapping["consistency_eligible"]) == 12
    assert mapping["right_overlay"] == "prediction_red_only_with_predicted_camera"
    assert mapping["initial_state_drawn"] and mapping["projection_nonfinite_policy"] == "raise_data_error"
    reader = imageio.get_reader(mp4)
    try:
        assert reader.count_frames() == 25
        assert reader.get_meta_data()["fps"] == 30
        assert reader.get_data(0).shape == (48, 128, 3)
    finally:
        reader.close()


def test_cli_rejects_hash_mismatch_legacy_and_existing_output(cli_rollout, tmp_path):
    import json
    from cosmos3_joint_video_hand_pose.src.ar_v02_eval import main, load_rollout

    archive, _ = cli_rollout
    fake = tmp_path / "wrong.json"
    fake.write_text("{}")
    out = tmp_path / "metrics.json"
    with pytest.raises(SystemExit, match="SHA256"):
        main(
            ["evaluate", "--input", str(archive), "--output", str(out), "--rigid-only", "--state-normalizer", str(fake)]
        )
    assert not out.exists()
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, metadata=json.dumps(dict(schema="legacy", layout_version="joint_state_single_v1")))
    with pytest.raises(ValueError, match="legacy"):
        load_rollout(legacy)
    out.write_text("keep me")
    with pytest.raises(SystemExit, match="already exists"):
        main(["evaluate", "--input", str(archive), "--output", str(out), "--rigid-only"])
    assert out.read_text() == "keep me"


def test_cli_missing_rgb_and_cpu_sample_fail_explicitly(cli_rollout, tmp_path):
    from cosmos3_joint_video_hand_pose.src.ar_v02_eval import main

    archive, _ = cli_rollout
    without = tmp_path / "no_rgb.npz"
    with np.load(archive, allow_pickle=False) as original:
        data = {k: original[k] for k in ("metadata", "predicted_action", "gt_future", "boundary_states")}
    np.savez(without, **data)
    with pytest.raises(SystemExit, match="overlay requires"):
        main(["overlay", "--input", str(without), "--output", str(tmp_path / "none.mp4")])
    run = _cli(
        "sample",
        "--ckpt",
        tmp_path / "checkpoint",
        "--episodes-manifest",
        tmp_path / "episodes.csv",
        "--segments-manifest",
        tmp_path / "segments.csv",
        "--eval-windows",
        tmp_path / "windows.json",
        "--split",
        "heldout",
        "--output",
        tmp_path / "no_gpu",
    )
    assert run.returncode != 0 and "allocated CUDA GPU" in run.stderr
    assert not (tmp_path / "no_gpu").exists()


def test_overlay_panel_colors_and_initial_state(monkeypatch):
    import cosmos3_joint_video_hand_pose.src.ar_overlay as old_overlay

    sampler, future, _ = fixture(c=2, future_frames=3)
    draws = []

    def record(frame, uv, valid, color, thickness):
        draws.append(color)

    monkeypatch.setattr(old_overlay, "draw_hand", record)
    codec = SimpleNamespace(decode=lambda x: torch.zeros(len(x), 20, 3))
    gt = np.zeros((25, 48, 64, 3), dtype=np.uint8)
    rgb = [np.zeros((1 + b.action_count // 2, 48, 64, 3), dtype=np.uint8) for b in sampler.layout.boundaries]
    _, metadata = render_joint_overlay(
        sampler.layout,
        sampler.gt_action,
        future,
        sampler.gt_states,
        gt_rgb=gt,
        generated_rgb_chunks=rgb,
        intrinsics=np.eye(3),
        gt_pixel_transform=np.eye(3),
        generated_pixel_transform=np.eye(3),
        state_normalizer=sampler.state_normalizer,
        future_normalizer=sampler.future_normalizer,
        hand_codecs=(codec, codec),
    )
    expected = [old_overlay.GT_COLOR] * 2 + [old_overlay.PRED_COLOR] * 4
    assert draws == expected * 25  # includes t=0, and no right-panel green skeleton.
    assert metadata["generated_chunk_ids"][17] == 1 and metadata["chunk_ids"][17] == 2
    bad = np.zeros((1, 2, 21, 3))
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        project_chunk_hands(bad, np.eye(4)[None], np.eye(3))
