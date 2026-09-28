import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from cosmos3_joint_video_hand_pose.src.action import _pose9_matrices, pose9, pose_matrices
from cosmos3_joint_video_hand_pose.src.ar_chunk_state import (
    BoundaryState,
    boundary_states_from_streams,
    chunk_boundaries,
    decode_action_chunk,
    encode_boundary_state,
    state_boundary_indices,
)
from cosmos3_joint_video_hand_pose.src.normalization import PiecewiseAsinhNormalizer


@pytest.fixture
def normalizer(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "method": "piecewise_asinh_rot",
                "beta": 1.0,
                "stats": {"center": np.linspace(-0.2, 0.2, 27).tolist(), "scale": [0.4] * 27},
            }
        )
    )
    return PiecewiseAsinhNormalizer(path)


def trajectory(length):
    rng = np.random.default_rng(42)
    streams = []
    for offset in ([1, 2, 3], [1.3, 2.1, 3.2], [0.7, 2.1, 3.2]):
        poses = np.zeros((length, 7))
        poses[:, :3] = np.asarray(offset) + np.cumsum(rng.normal(0, 0.01, (length, 3)), axis=0)
        angles = np.array([0.2, -0.4, 0.3]) + np.cumsum(rng.normal(0, 0.005, (length, 3)), axis=0)
        poses[:, 3:] = Rotation.from_euler("xyz", angles).as_quat()[:, [3, 0, 1, 2]]
        streams.append(poses)
    hands = torch.from_numpy(rng.normal(size=(length, 2, 15)).astype(np.float32))
    return streams, hands


def states(streams, hands, boundaries, enabled=True):
    return boundary_states_from_streams(
        head_pose=streams[0],
        right_wrist_pose=streams[1],
        left_wrist_pose=streams[2],
        hand_latents=hands,
        boundaries=boundaries,
        chunk_state_conditioning=enabled,
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
@pytest.mark.parametrize("latent_frames", [2, 6, 9, 17, 33])
def test_boundaries_cover_every_future_row_once(chunk_size, latent_frames):
    bounds = chunk_boundaries(latent_frames, chunk_size)
    rows = [row for b in bounds for row in range(b.source_start + 1, b.source_stop + 1)]
    assert rows == list(range(1, (latent_frames - 1) * 8 + 1))
    assert [b.chunk_id for b in bounds] == list(range(1, len(bounds) + 1))
    assert all(b.action_count == (b.latent_stop - b.latent_start) * 8 for b in bounds)
    assert state_boundary_indices(bounds, chunk_state_conditioning=False) == (0,)
    assert state_boundary_indices(bounds, chunk_state_conditioning=True) == tuple(b.source_start for b in bounds)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
def test_states_keep_original_camera_frame_and_normalizer_roundtrip(chunk_size, normalizer):
    bounds = chunk_boundaries(10, chunk_size)
    streams, hands = trajectory(73)
    result = states(streams, hands, bounds)
    f0_from_world = np.linalg.inv(pose_matrices(streams[0])[0])
    for b, state in zip(bounds, result, strict=True):
        expected = np.stack([f0_from_world @ pose_matrices(stream)[b.source_start] for stream in streams])
        torch.testing.assert_close(state.rigid_f0, torch.from_numpy(expected).float(), atol=1e-5, rtol=1e-4)
        encoded = encode_boundary_state(state, normalizer)
        assert encoded.shape == (57,)
        pose27 = normalizer.denormalize(torch.cat((encoded[:18], encoded[33:42])))
        torch.testing.assert_close(_pose9_matrices(pose27.reshape(3, 9)), state.rigid_f0, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(encoded[18:33], hands[b.source_start, 0])
        torch.testing.assert_close(encoded[42:57], hands[b.source_start, 1])
    if len(result) > 1:
        assert not torch.allclose(result[1].rigid_f0[0], torch.eye(4))
    assert len(states(streams, hands, bounds, enabled=False)) == 1


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
def test_streaming_integration_matches_full_sequence_without_dropping_rows(chunk_size, normalizer):
    bounds = chunk_boundaries(10, chunk_size)
    streams, hands = trajectory(73)
    gt_states = states(streams, hands, bounds)
    pose27 = torch.from_numpy(
        np.concatenate([pose9(np.linalg.inv(pose_matrices(p)[:-1]) @ pose_matrices(p)[1:]) for p in streams], axis=-1)
    ).float()
    normalized = normalizer.normalize(pose27)
    future = torch.cat((normalized[:, :18], hands[1:, 0], normalized[:, 18:], hands[1:, 1]), dim=-1)
    full = decode_action_chunk(gt_states[0], future, normalizer)
    current = gt_states[0]
    chunks = []
    for b, expected_start in zip(bounds, gt_states, strict=True):
        torch.testing.assert_close(current.rigid_f0, expected_start.rigid_f0, atol=1e-5, rtol=1e-4)
        output = decode_action_chunk(current, future[b.source_start : b.source_stop], normalizer)
        assert len(output.rigid_f0) == b.action_count
        assert output.end_state.source_index == b.source_stop
        current = output.end_state
        chunks.append(output.rigid_f0)
    torch.testing.assert_close(torch.cat(chunks), full.rigid_f0, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(current.hand_latents, hands[-1])
    f0_from_world = np.linalg.inv(pose_matrices(streams[0])[0])
    expected = np.stack([f0_from_world @ pose_matrices(p)[1:] for p in streams], axis=1)
    torch.testing.assert_close(full.rigid_f0, torch.from_numpy(expected).float(), atol=1e-5, rtol=1e-4)


def test_different_decode_anchor_is_not_a_change_to_predicted_increments(normalizer):
    streams, hands = trajectory(9)
    original = states(streams, hands, chunk_boundaries(2, 1))[0]
    moved = original.rigid_f0.clone()
    moved[:, :3, 3] += torch.tensor([0.3, -0.2, 0.1])
    observed = BoundaryState(0, moved, original.hand_latents)
    identity9 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0]).float()
    pose27 = normalizer.normalize(identity9.repeat(3)).expand(8, -1)
    actions = torch.cat((pose27[:, :18], hands[1:, 0], pose27[:, 18:], hands[1:, 1]), dim=-1)
    a = decode_action_chunk(original, actions, normalizer)
    b = decode_action_chunk(observed, actions, normalizer)
    torch.testing.assert_close(
        b.rigid_f0[:, :, :3, 3] - a.rigid_f0[:, :, :3, 3], torch.tensor([0.3, -0.2, 0.1]).expand(8, 3, 3)
    )
    torch.testing.assert_close(a.hand_latents, b.hand_latents)


def test_invalid_input_and_padding_fail_explicitly(normalizer):
    with pytest.raises(ValueError):
        chunk_boundaries(0, 4)
    with pytest.raises(ValueError):
        chunk_boundaries(5, 1.5)
    streams, hands = trajectory(9)
    state = states(streams, hands, chunk_boundaries(2, 1))[0]
    action = torch.zeros(8, 64)
    action[:, 57] = 1
    with pytest.raises(ValueError, match="padding"):
        decode_action_chunk(state, action, normalizer)
    action[:, 57] = 0
    action[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        decode_action_chunk(state, action, normalizer)
    with pytest.raises(ValueError, match="beyond"):
        states(streams, hands, chunk_boundaries(3, 1))


def test_future_gt_does_not_change_current_boundary():
    streams, hands = trajectory(65)
    bounds = chunk_boundaries(9, 4)
    before = states(streams, hands, bounds)
    modified = [p.copy() for p in streams]
    for p in modified:
        p[33:, :3] += 100
    modified_hands = hands.clone()
    modified_hands[33:] += 100
    after = states(modified, modified_hands, bounds)
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a.rigid_f0, b.rigid_f0)
        torch.testing.assert_close(a.hand_latents, b.hand_latents)


from cosmos3_joint_video_hand_pose.src.ar_chunk_state import (
    CHUNK_CAMERA_STATE_SCHEMA,
    CHUNK_CAMERA_LAYOUT_VERSION,
    ChunkCameraState,
    ChunkCameraStateNormalizer,
    chunk_camera_states_from_streams,
    encode_chunk_camera_state,
    decode_chunk_camera_state,
    decode_chunk_camera_action,
    state_profile_sha256,
)


@pytest.fixture
def camera_normalizer(tmp_path):
    profile = dict(
        schema=CHUNK_CAMERA_STATE_SCHEMA,
        layout_version=CHUNK_CAMERA_LAYOUT_VERSION,
        split="train",
        frozen=True,
        camera_encoding="zero_identity",
        method="piecewise_asinh_rot",
        manifest_sha256="a" * 64,
        fit_samples_sha256="b" * 64,
        beta=1.0,
        stats=dict(center=np.linspace(-0.2, 0.2, 18).tolist(), scale=[0.4] * 18),
    )
    profile["profile_sha256"] = state_profile_sha256(profile)
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(profile))
    return ChunkCameraStateNormalizer(path)


def camera_states(streams, hands, bounds):
    return chunk_camera_states_from_streams(
        head_pose=streams[0],
        right_wrist_pose=streams[1],
        left_wrist_pose=streams[2],
        hand_latents=hands,
        boundaries=bounds,
    )


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("latents", [6, 10])
def test_chunk_camera_roundtrip_and_increment_integration(c, latents, camera_normalizer, normalizer):
    length = (latents - 1) * 8 + 1
    streams, hands = trajectory(length)
    bounds = chunk_boundaries(latents, c)
    states = camera_states(streams, hands, bounds)
    matrices = np.stack([pose_matrices(p) for p in streams], axis=1)
    delta = torch.from_numpy(
        np.concatenate([pose9(np.linalg.inv(pose_matrices(p)[:-1]) @ pose_matrices(p)[1:]) for p in streams], axis=-1)
    ).float()
    values = normalizer.normalize(delta)
    future = torch.cat((values[:, :18], hands[1:, 0], values[:, 18:], hands[1:, 1]), dim=-1)
    for i, (b, state) in enumerate(zip(bounds, states, strict=True)):
        expected = np.linalg.inv(matrices[b.source_start, 0]) @ matrices[b.source_start]
        torch.testing.assert_close(state.rigid_camera, torch.from_numpy(expected).float(), atol=1e-5, rtol=1e-4)
        encoded = encode_chunk_camera_state(state, camera_normalizer)
        assert torch.count_nonzero(encoded[:9]) == 0
        recovered = decode_chunk_camera_state(
            torch.nn.functional.pad(encoded, (0, 7)), camera_normalizer, source_index=b.source_start
        )
        torch.testing.assert_close(recovered.rigid_camera, state.rigid_camera, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(recovered.hand_latents, hands[b.source_start])
        result = decode_chunk_camera_action(recovered, future[b.source_start : b.source_stop], normalizer)
        selected = matrices[b.source_start + 1 : b.source_stop + 1]
        physical = np.linalg.inv(matrices[b.source_start, 0]) @ selected
        torch.testing.assert_close(result.rigid_chunk, torch.from_numpy(physical).float(), atol=1e-5, rtol=1e-4)
        camera_wrists = np.linalg.inv(selected[:, 0])[:, None] @ selected[:, 1:]
        torch.testing.assert_close(result.wrist_camera, torch.from_numpy(camera_wrists).float(), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(result.hand_latents, hands[b.source_start + 1 : b.source_stop + 1])
        assert result.end_state.source_index == b.source_stop
        assert torch.equal(result.end_state.rigid_camera[0], torch.eye(4))
        if i + 1 < len(states):
            torch.testing.assert_close(result.end_state.rigid_camera, states[i + 1].rigid_camera, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(result.end_state.hand_latents, states[i + 1].hand_latents)


def test_candidate_is_independent_of_window_first_camera_and_future(camera_normalizer):
    streams, hands = trajectory(33)
    before = camera_states(streams, hands, chunk_boundaries(5, 1))
    modified = [p.copy() for p in streams]
    modified[0][0, :3] += 100
    for p in modified:
        p[25:, :3] += 200
    after = camera_states(modified, hands, chunk_boundaries(5, 1))
    for a, b in zip(before[1:], after[1:], strict=True):
        torch.testing.assert_close(a.rigid_camera, b.rigid_camera, atol=0, rtol=0)
    shifted = camera_states([p[8:] for p in streams], hands[8:], chunk_boundaries(4, 1))
    torch.testing.assert_close(before[1].rigid_camera, shifted[0].rigid_camera, atol=0, rtol=0)


def test_camera_state_rejects_invalid_inputs_and_legacy_profile(camera_normalizer, normalizer):
    streams, hands = trajectory(9)
    state = camera_states(streams, hands, chunk_boundaries(2, 1))[0]
    encoded = encode_chunk_camera_state(state, camera_normalizer)
    with pytest.raises(ValueError, match="18D"):
        encode_chunk_camera_state(state, normalizer)
    for column, message in [(0, "camera"), (57, "padding")]:
        bad = torch.nn.functional.pad(encoded, (0, 7))
        bad[column] = 1
        with pytest.raises(ValueError, match=message):
            decode_chunk_camera_state(bad, camera_normalizer, source_index=0)
    bad = state.rigid_camera.clone()
    bad[0, 0, 3] = 0.1
    with pytest.raises(ValueError, match="identity"):
        ChunkCameraState(0, bad, hands[0])
    streams[0][0, 3:] *= 2
    with pytest.raises(ValueError, match="quaternion"):
        camera_states(streams, hands, chunk_boundaries(2, 1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "ar_v02_boundary_state_f0_v1"),
        ("layout_version", "joint_state_single_v1"),
        ("split", "heldout"),
        ("frozen", False),
    ],
)
def test_camera_normalizer_rejects_versions(camera_normalizer, field, value):
    profile = json.loads(camera_normalizer.path.read_text())
    profile[field] = value
    profile["profile_sha256"] = state_profile_sha256(profile)
    camera_normalizer.path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        ChunkCameraStateNormalizer(camera_normalizer.path)


def test_camera_normalizer_detects_modified_statistics(camera_normalizer):
    with pytest.raises(ValueError, match="file hash"):
        ChunkCameraStateNormalizer(camera_normalizer.path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="manifest hash"):
        ChunkCameraStateNormalizer(camera_normalizer.path, expected_manifest_sha256="0" * 64)
    profile = json.loads(camera_normalizer.path.read_text())
    profile["stats"]["center"][0] += 1
    camera_normalizer.path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="profile hash"):
        ChunkCameraStateNormalizer(camera_normalizer.path)
