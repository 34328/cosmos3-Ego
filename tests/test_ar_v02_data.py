"""CPU data-contract tests; no VAE, codecs, or GPU are required."""

import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos3_joint_video_hand_pose.src import ar_dataset as module
from cosmos3_joint_video_hand_pose.src.action import pose27_from_streams, pose_matrices, pose9
from cosmos3_joint_video_hand_pose.src.ar_chunk_state import (
    CHUNK_CAMERA_STATE_SCHEMA,
    CHUNK_CAMERA_LAYOUT_VERSION,
    VALID_WINDOWS_SCHEMA,
    ChunkCameraStateNormalizer,
    decode_chunk_camera_state,
    state_profile_sha256,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_prepare_data import prepare, sha256, invalid_frames


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Group(dict):
    attrs = {"total_frames": 75}


class Builder:
    rigid_pose_frame_delta = True

    def build(self, *, head_pose, right_wrist_pose, left_wrist_pose, **kwargs):
        # Deliberately transparent pose normalization lets tests inspect deltas.
        pose = torch.from_numpy(
            pose27_from_streams(head_pose, right_wrist_pose, left_wrist_pose, rigid_pose_frame_delta=True)
        )
        hands = torch.from_numpy(head_pose[:, :1].astype(np.float32)).expand(-1, 15)
        return torch.cat((pose[:, :18], hands + 0.7, pose[:, 18:], hands - 0.4), dim=-1)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    groups = {}
    paths = {}
    for split in ("train", "heldout"):
        ep = split + "_ep"
        group = Group()
        for name, offset in (("obs_head_pose", 0), ("right.obs_wrist_pose", 0.3), ("left.obs_wrist_pose", -0.3)):
            poses = np.zeros((75, 7))
            poses[:, 0] = np.arange(75) * 0.01 + offset + (100 if split == "heldout" and offset else 0)
            angles = np.arange(75) * 0.005
            poses[:, 3] = np.cos(angles / 2)
            poses[:, 6] = np.sin(angles / 2)
            group[name] = poses
        for side in ("right", "left"):
            group[side + ".obs_keypoints"] = np.ones((75, 21, 3)) * 0.01
            group[side + ".obs_palm_in_fov_front_1"] = np.zeros(75, dtype=np.uint8)
        group["images.front_1"] = np.arange(75)
        groups[ep] = group
        episode_path, segment_path = tmp_path / (split + "_ep.csv"), tmp_path / (split + "_seg.csv")
        write_csv(episode_path, [dict(episode_hash=ep, split=split, fps=30, abs_zarr_path=ep, task_description="task")])
        write_csv(
            segment_path,
            [dict(episode_hash=ep, split=split, span_index=0, start_idx=4, end_idx=75, text_normalized="move hand")],
        )
        paths[split + "_episodes"] = str(episode_path)
        paths[split + "_segments"] = str(segment_path)
    monkeypatch.setattr(module.zarr, "open_group", lambda path, mode: groups[path])
    monkeypatch.setattr(
        module, "decode_rgb_video", lambda ids: torch.from_numpy(ids.copy()).view(1, -1, 1, 1).expand(3, -1, 368, 640)
    )
    args = SimpleNamespace(
        **paths, output=str(tmp_path / "data_v2"), seed=42, windows_per_segment=2, reference_data=None
    )
    prepare(args)
    return args, groups


def dataset(args, **kwargs):
    from pathlib import Path

    root = Path(args.output)
    options = dict(
        token_counter=lambda _: 20,
        prompt_formatter=lambda caption, frames, fps: json.dumps({"caption": caption}),
        action_builder=Builder(),
        random_window=False,
        chunk_state_normalizer=root / "chunk_state_normalizer.json",
        valid_windows_manifest=root / "valid_windows.json",
    )
    options.update(kwargs)
    return module.EgoVerseARSegmentDataset(args.train_episodes, args.train_segments, **options)


def test_future_actions_all_candidates_rgb_and_source_indexes(inputs):
    args, groups = inputs
    ds = dataset(args)
    sample = ds[0]
    assert sample["ar_layout_version"] == CHUNK_CAMERA_LAYOUT_VERSION
    assert sample["ar_state_schema"] == CHUNK_CAMERA_STATE_SCHEMA
    assert sample["action"].shape == (64, 57)
    assert sample["ar_boundary_states"].shape == (8, 64)
    assert torch.count_nonzero(sample["ar_boundary_states"][:, :9]) == 0
    assert torch.count_nonzero(sample["ar_boundary_states"][:, 57:]) == 0
    assert sample["source_frame_indices"].tolist() == list(range(4, 69, 2))
    assert sample["action_source_frame_indices"].tolist() == list(range(4, 69))
    assert sample["future_action_source_frame_indices"].tolist() == list(range(5, 69))
    assert sample["ar_boundary_source_indices"].tolist() == list(range(4, 68, 8))
    torch.testing.assert_close(sample["video"][0, :, 0, 0], sample["source_frame_indices"])
    torch.testing.assert_close(sample["ar_boundary_times"], sample["ar_boundary_source_indices"].double() / 30)
    group = groups["train_ep"]
    streams = [group[name][4:69] for name in ("obs_head_pose", "right.obs_wrist_pose", "left.obs_wrist_pose")]
    direct = Builder().build(head_pose=streams[0], right_wrist_pose=streams[1], left_wrist_pose=streams[2])
    torch.testing.assert_close(sample["action"], direct[1:], atol=0, rtol=0)
    assert not sample["hand_visibility"].any()  # Out of FOV is not missing tracking.
    for j, offset in enumerate(range(0, 64, 8)):
        state = decode_chunk_camera_state(
            sample["ar_boundary_states"][j], ds.chunk_state_normalizer, source_index=offset + 4
        )
        expected = np.stack(
            [np.linalg.inv(pose_matrices(streams[0])[offset]) @ pose_matrices(p)[offset] for p in streams]
        )
        torch.testing.assert_close(state.rigid_camera, torch.from_numpy(expected).float(), atol=1e-5, rtol=1e-4)
    for c in (1, 2, 3, 4):
        selected = sample["ar_boundary_source_indices"][::c]
        assert selected.tolist() == list(range(4, 68, 8 * c))
    old = dataset(args, chunk_state_normalizer=None, valid_windows_manifest=None)[0]
    torch.testing.assert_close(sample["action"], old["action"][8:], atol=0, rtol=0)
    torch.testing.assert_close(sample["video"], old["video"], atol=0, rtol=0)


def test_fit_is_train_only_18d_and_reference_eval_is_stable(inputs, tmp_path):
    from pathlib import Path

    args, _ = inputs
    root = Path(args.output)
    profile = json.loads((root / "chunk_state_normalizer.json").read_text())
    assert len(profile["stats"]["center"]) == 18
    assert max(abs(x) for x in profile["stats"]["maximum"]) < 2  # Heldout wrist is >100m away.
    assert min(profile["stats"]["scale"]) >= 0.01
    records = json.loads((root / "state_fit_samples.json").read_text())
    assert {r["C"] for r in records} == {1, 2, 3, 4}
    assert all(r["sample_id"].startswith("train_ep") for r in records)
    assert profile["fit_samples_sha256"] == sha256(root / "state_fit_samples.json")
    repeat = SimpleNamespace(**(vars(args) | dict(output=str(tmp_path / "repeat"), reference_data=str(root))))
    prepare(repeat)
    assert (Path(repeat.output) / "eval_windows.json").read_bytes() == (root / "eval_windows.json").read_bytes()
    assert json.loads((Path(repeat.output) / "chunk_state_normalizer.json").read_text())["stats"] == profile["stats"]


@pytest.mark.parametrize("target", ["normalizer", "manifest", "episodes", "segments"])
def test_dataset_rejects_hash_mismatch(inputs, target):
    from pathlib import Path

    args, _ = inputs
    root = Path(args.output)
    if target in ("normalizer", "manifest"):
        path = root / ("chunk_state_normalizer.json" if target == "normalizer" else "valid_windows.json")
        payload = json.loads(path.read_text())
        if target == "normalizer":
            payload["stats"]["center"][0] += 1
        else:
            payload["seed"] += 1
        path.write_text(json.dumps(payload))
    else:
        path = Path(getattr(args, "train_" + target))
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash|match|differs"):
        dataset(args)


def test_invalid_tracking_after_audit_fails_before_builder(inputs):
    args, groups = inputs
    ds = dataset(args)
    groups["train_ep"]["right.obs_wrist_pose"][12, 3:] *= 2
    with pytest.raises(ValueError, match="invalid_quaternion"):
        ds[0]


def test_reject_wrong_future_representation_and_budget_includes_conditions(inputs):
    args, _ = inputs
    builder = Builder()
    builder.rigid_pose_frame_delta = False
    with pytest.raises(ValueError, match="frame-delta"):
        dataset(args, action_builder=builder)
    # Old layout fits 3000 tokens; the per-chunk U/S maximum does not.
    with pytest.raises(ValueError, match="no 'train'"):
        dataset(args, max_sequence_length=3000)
    assert len(dataset(args, chunk_state_normalizer=None, valid_windows_manifest=None, max_sequence_length=3000)) == 1


def test_old_f0_normalizer_and_manifest_remain_supported(inputs):
    from pathlib import Path

    args, _ = inputs
    root = Path(args.output)
    manifest_path = root / "valid_windows.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "ar_v02_valid_windows_v1"
    manifest_path.write_text(json.dumps(manifest))
    profile = dict(
        schema="ar_v02_boundary_state_f0_v1",
        split="train",
        frozen=True,
        manifest_sha256=sha256(manifest_path),
        method="piecewise_asinh_rot",
        stats=dict(center=[0] * 27, scale=[1] * 27),
    )
    (root / "chunk_state_normalizer.json").write_text(json.dumps(profile))
    sample = dataset(args)[0]
    assert sample["ar_layout_version"] == "joint_state_single_v1"
    assert sample["action"].shape == (64, 57)
    assert sample["ar_boundary_states"][1, 0] != 0  # F0 camera moved; no reinterpretation as v2.


@pytest.mark.parametrize("frames", [33, 65, 129])
@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_double_pass_budget_matches_native_pack_and_bounds_any_c(frames, c):
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
    from cosmos_framework.model.generator.teacher_forcing import make_teacher_forcing_clean_pack
    from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout
    from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence
    from cosmos3_joint_video_hand_pose.src.ar_v02_dataloader import joint_training_token_budget

    layout = JointChunkLayout(1 + (frames - 1) // 4, 240, c)
    nt = 101
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, layout.num_video_frames, 24, 40)],
        x0_tokens_action=[torch.zeros(layout.num_action_rows, 64)],
        fps_vision=torch.tensor([7.5]),
        fps_action=torch.tensor([15.0]),
        raw_action_dim=[torch.tensor(57)],
        action_domain_id=[torch.tensor([3])],
    )
    pack = pack_joint_sequence(
        layout=[layout],
        gen_data_clean=data,
        text_ids=[[3] * nt],
        special_tokens={"eos_token_id": 5, "start_of_generation": 6},
        timesteps=[torch.tensor([0])],
        condition_frames=[()],
        latent_patch_size=2,
    )
    clean = make_teacher_forcing_clean_pack(pack)
    assert module.ar_v02_token_count(nt, frames, chunk_size=c) == pack.sequence_length + clean.sequence_length
    assert module.ar_v02_token_count(nt, frames) >= pack.sequence_length + clean.sequence_length
    assert module.ar_v02_token_count(nt, frames) == joint_training_token_budget(nt, frames, 368, 640)


def test_budget_refreshed_after_text_transform_and_keeps_sequence_plan(inputs):
    args, _ = inputs
    raw = dataset(args)[0]
    plan = object()

    def transform(sample, resolution=None):
        sample["text_token_ids"] = torch.arange(103)
        sample["sequence_plan"] = plan
        return sample

    result = module.ARV02BudgetTransform(transform, 70_000)(raw)
    assert result["ar_num_tokens"] == module.ar_v02_token_count(103, 33)
    assert result["ar_token_budget_version"] == module.AR_V02_TOKEN_BUDGET_VERSION
    assert result["sequence_plan"] is plan
    with pytest.raises(ValueError, match="double-pass"):
        module.ARV02BudgetTransform(transform, result["ar_num_tokens"])(raw)


def test_saved_window_rebuilds_all_payloads_without_random_draw(inputs, monkeypatch):
    import random

    args, _ = inputs
    ds = dataset(args, random_window=True)
    saved = ds.get_item_at_window(0, window_start=9)
    monkeypatch.setattr(ds, "window_start", lambda row: 4)
    wrong = ds[0]
    assert not torch.equal(wrong["video"], saved["video"])
    assert not torch.equal(wrong["action"], saved["action"])
    assert not torch.equal(wrong["ar_boundary_states"], saved["ar_boundary_states"])
    rng = random.getstate()
    restored = ds.get_item_at_window(0, source_frame_indices=saved["source_frame_indices"][None])
    assert random.getstate() == rng
    assert restored["window_start"] == 9
    for key in (
        "video",
        "action",
        "hand_visibility",
        "ar_boundary_states",
        "ar_source_poses",
        "ar_hand_latents",
        "source_frame_indices",
        "action_source_frame_indices",
        "future_action_source_frame_indices",
        "ar_boundary_source_indices",
        "ar_boundary_times",
    ):
        torch.testing.assert_close(restored[key], saved[key], atol=0, rtol=0)

    def transform(sample, resolution=None):
        sample["action"] = torch.nn.functional.pad(sample["action"], (0, 7))
        sample["raw_action_dim"] = 57
        sample["text_token_ids"] = torch.arange(11)
        return sample

    wrapper = module.EgoVerseARCosmosDataset(ds, module.ARV02BudgetTransform(transform, 70_000))
    item = wrapper.get_item_at_window(0, window_start=torch.tensor([9]))
    assert item["dataset_index"] == 0 and item["window_start"] == 9
    torch.testing.assert_close(item["action"][:, :57], saved["action"], atol=0, rtol=0)
    torch.testing.assert_close(item["ar_boundary_states"], saved["ar_boundary_states"], atol=0, rtol=0)


def test_saved_window_rejects_shifted_incomplete_or_unaudited_metadata(inputs):
    args, _ = inputs
    ds = dataset(args)
    indexes = torch.arange(4, 69, 2)
    with pytest.raises(ValueError, match="disagree"):
        ds.get_item_at_window(0, window_start=5, source_frame_indices=indexes)
    with pytest.raises(ValueError, match="full integer"):
        ds.get_item_at_window(0, source_frame_indices=indexes[:-1])
    bad = indexes.clone()
    bad[3] += 1
    with pytest.raises(ValueError, match="stride"):
        ds.get_item_at_window(0, source_frame_indices=bad)
    with pytest.raises(ValueError, match="exceeds"):
        ds.get_item_at_window(0, window_start=3)
    with pytest.raises(ValueError, match="integer"):
        ds.get_item_at_window(0)
    ds.rows[0]["_valid_starts"].remove(9)
    with pytest.raises(ValueError, match="absent"):
        ds.get_item_at_window(0, window_start=9)
