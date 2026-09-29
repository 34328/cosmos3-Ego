"""Fixed C4 data/fit integration. Synthetic nonlinear codecs, no model/GPU."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos3_joint_video_hand_pose.src import ar_dataset as ds
from cosmos3_joint_video_hand_pose.src import ar_v02_prepare_data as prep
from cosmos3_joint_video_hand_pose.src import action_fixed_normalization as norm
from cosmos3_joint_video_hand_pose.src.action import pose_matrices


class Codec:
    coordinate_system = "fixed-camera"
    representation = norm.REPRESENTATION
    input_frame = "current_frame_wrist_local"
    metadata = {"fit": {"episode_ids": ["tr"]}}

    def __init__(self, tag):
        self.checkpoint_sha256 = tag * 64
        self.checkpoint_path = "/synthetic/" + tag

    def encode(self, q):
        x = q.flatten(-2)[..., :15]
        return x + 3*x.square()  # E(dq) is observably different from delta E(q)

    def decode(self, z):
        return torch.nn.functional.pad(z, (0, 45)).reshape(*z.shape[:-1], 20, 3)


def streams(n):
    t = np.arange(n, dtype=np.float32)
    head = np.zeros((n, 7), dtype=np.float32)
    head[:, 0] = t*.01
    head[:, 3], head[:, 6] = np.cos(t*.01), np.sin(t*.01)
    wrists = []
    points = []
    for side in (1, -1):
        w = head.copy()
        w[:, :3] += [side*.2, .3, 1]
        w[:, 1] += t*.002
        q = np.ones((n, 21, 3), dtype=np.float32)*.01
        q[:, 0] = 0
        q[:, 1:, 0] += t[:, None]*.001
        wrists.append(w)
        points.append(w[:, None, :3] + q)
    return (head, *wrists, *points)


@pytest.mark.parametrize("future_count", [8, 16, 24, 32, 40, 48, 56, 64, 72])
def test_c4_partial_chunks_exact_frame_and_latent_difference(future_count):
    raw = streams(future_count+1)
    codecs = (Codec("a"), Codec("b"))
    states, actions = norm.encode_fixed_window(*raw, codecs)
    assert states.shape == (future_count//8, 57)
    assert actions.shape == (future_count, 57)
    assert states[:, :9].count_nonzero() == 0
    # Independently form CURRENT wrist-local q, then difference E(q) at each C4 boundary.
    head, right, left, rp, lp = raw
    for b in range(0, future_count, 32):
        e = min(b+32, future_count)
        for w, pts, c, slot in ((right, rp, codecs[0], slice(18,33)),
                                (left, lp, codecs[1], slice(42,57))):
            q_world = torch.tensor(pts[b:e+1, 1:]-w[b:e+1,None,:3])
            wrist_rotation = torch.tensor(pose_matrices(w[b:e+1])[:, :3, :3], dtype=torch.float32)
            q = torch.einsum("tji,tnj->tni", wrist_rotation, q_world)
            z = c.encode(q)
            torch.testing.assert_close(actions[b:e,slot], z[1:]-z[:-1], atol=2e-6, rtol=1e-4)
            torch.testing.assert_close(states[b//8,slot], z[0], atol=2e-6, rtol=1e-4)


class Group(dict):
    attrs = {"total_frames": 65}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    raw = streams(65)
    names = ("obs_head_pose", "right.obs_wrist_pose", "left.obs_wrist_pose",
             "right.obs_keypoints", "left.obs_keypoints")
    group = Group(zip(names, raw))
    group["images.front_1"] = np.zeros(65)
    for side in ("right", "left"):
        group[side+".obs_palm_in_fov_front_1"] = np.ones(65, dtype=np.uint8)
    monkeypatch.setattr(prep.zarr, "open_group", lambda *a, **k: group)
    monkeypatch.setattr(ds, "decode_rgb_video", lambda x: torch.zeros(3,len(x),368,640))
    codecs = (Codec("a"), Codec("b"))
    for side, codec in zip(("right", "left"), codecs):
        path = tmp_path / (side + "_source.pt")
        path.write_bytes((side + " synthetic codec").encode())
        codec.checkpoint_path = str(path)
        codec.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".validation.json").write_text(json.dumps(dict(
            checkpoint_sha256=codec.checkpoint_sha256, passed=True)))
    monkeypatch.setattr(norm, "load_fixed_codecs", lambda paths: codecs)
    paths = {}
    for split, eid in (("train","tr"), ("heldout","ho")):
        rows = {
            "episodes": dict(episode_hash=eid, split=split, abs_zarr_path=eid, fps=30, task_description="test"),
            "segments": dict(episode_hash=eid, split=split, span_index=0, start_idx=0, end_idx=65, text_normalized="test"),
        }
        for kind, row in rows.items():
            p = tmp_path / (split+"_"+kind+".csv")
            with p.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader(); writer.writerow(row)
            paths[split+"_"+kind] = str(p)
    args = SimpleNamespace(**paths, output=str(tmp_path/"prepared"), seed=42,
                           windows_per_segment=1, right_codec="a", left_codec="b",
                           representation=norm.REPRESENTATION, reference_data=None)
    # Real CLI dispatch, audit and fitting. Only raw storage/video and codecs mocked.
    argv = ["prepare"]
    for key, value in vars(args).items():
        if value is not None:
            argv.extend(("--"+key.replace("_","-"), str(value)))
    monkeypatch.setattr(sys, "argv", argv)
    prep.main()
    return args, raw, codecs


def make_dataset(args, **kwargs):
    root = Path(args.output)
    return ds.EgoVerseARSegmentDataset(
        args.train_episodes, args.train_segments, random_window=False,
        token_counter=lambda text: 2, prompt_formatter=lambda *a: "{}",
        chunk_state_normalizer=root/"chunk_state_normalizer.json",
        future_normalizer=root/"future_normalizer.json",
        valid_windows_manifest=root/"valid_windows.json",
        right_codec="a", left_codec="b", **kwargs)


def test_codec_bootstrap_audit_requires_no_codec_or_statistics(prepared, tmp_path, monkeypatch):
    args, _, _ = prepared
    def forbidden(*args, **kwargs):
        raise AssertionError("source audit must not load an AE")
    monkeypatch.setattr(norm, "load_fixed_codecs", forbidden)
    args.output = str(tmp_path / "source_audit")
    args.clip_frames = [33]
    summary = prep.audit_only(args)
    manifest = json.loads((Path(args.output) / "valid_windows.json").read_text())
    assert manifest["schema"] == "ar_v02_codec_source_windows_v1"
    assert manifest["representation"] == norm.REPRESENTATION
    assert manifest["tracking_validation"]["scope"] == "all_source_frames"
    assert len(manifest["windows"]) == 2
    assert summary["train"]["retained_windows"] == 1
    assert summary["heldout"]["retained_windows"] == 1
    assert summary["statistics_fitted"] is False
    assert not (Path(args.output) / "chunk_state_normalizer.json").exists()
    assert "codec_sha256" not in manifest


def test_codec_snapshot_rejects_source_changed_after_validation(prepared, tmp_path):
    _, _, codecs = prepared
    destination = tmp_path / "tampered_bundle"
    destination.mkdir()
    Path(codecs[0].checkpoint_path).write_bytes(b"changed after validation")
    with pytest.raises(ValueError, match="changed while preparing"):
        prep.snapshot_fixed_codecs(codecs, destination)


def test_cli_dataset_layout_stats_and_metadata(prepared):
    args, raw, codecs = prepared
    dataset = make_dataset(args)
    item = dataset[0]
    s, a = norm.encode_fixed_window(*raw, codecs)
    assert dataset.fixed_hand_codecs is codecs
    for side, codec in zip(("right", "left"), codecs):
        bundled = Path(args.output) / (side + "_mlp15.pt")
        assert hashlib.sha256(bundled.read_bytes()).hexdigest() == codec.checkpoint_sha256
        assert bundled.with_suffix(".validation.json").is_file()
    assert item["ar_action_representation"] == norm.REPRESENTATION
    expected_points = np.stack((raw[3].reshape(65,21,3), raw[4].reshape(65,21,3)), axis=1)
    np.testing.assert_array_equal(item["ar_source_keypoints_world"], expected_points)
    assert item["ar_right_hand_codec_sha256"] == dataset.chunk_state_normalizer.codec_sha256[0] == codecs[0].checkpoint_sha256
    assert item["ar_left_hand_codec_sha256"] == dataset.chunk_state_normalizer.codec_sha256[1] == codecs[1].checkpoint_sha256
    from torch.utils.data import default_collate
    metadata = {k: item[k] for k in ("ar_right_hand_codec_sha256", "ar_left_hand_codec_sha256")}
    collated = default_collate([metadata, metadata])
    assert collated["ar_right_hand_codec_sha256"] == [codecs[0].checkpoint_sha256]*2
    assert collated["ar_left_hand_codec_sha256"] == [codecs[1].checkpoint_sha256]*2
    assert item["action"].shape == (64,57)
    assert item["ar_boundary_states"].shape == (8,64)
    assert item["future_action_source_frame_indices"].tolist() == list(range(1,65))
    assert item["ar_boundary_source_offsets"].tolist() == list(range(0,64,8))
    assert item["ar_boundary_states"][:,57:].count_nonzero() == 0
    torch.testing.assert_close(dataset.future_normalizer.denormalize(item["action"]), a, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(dataset.chunk_state_normalizer.denormalize(item["ar_boundary_states"][:,:57]), s)
    # Heldout never contributes; state fit excludes unused candidate boundaries.
    for kind, rows, file in (("state", s[::4], "chunk_state_normalizer.json"),
                             ("future", a, "future_normalizer.json")):
        p = json.loads((Path(args.output)/file).read_text())
        assert p["count"] == len(rows)
        assert p["split"] == "train"
        assert p["codec_sha256"] == [c.checkpoint_sha256 for c in codecs]
        expected = (torch.quantile(rows.double(), .01, dim=0) + torch.quantile(rows.double(), .99, dim=0))/2
        torch.testing.assert_close(torch.tensor(p["stats"]["center"], dtype=torch.float64), expected)
    records = json.loads((Path(args.output)/"fit_samples.json").read_text())
    assert all(r["sample_id"].startswith("tr:") and r["C"] == 4 for r in records)


def test_reject_wrong_schema_manifest_and_codec(prepared):
    args, _, codecs = prepared
    with pytest.raises(ValueError, match="schema"):
        make_dataset(args, action_representation="legacy_local_delta_absolute_hand_v1")
    saved_hash = codecs[0].checkpoint_sha256
    codecs[0].checkpoint_sha256 = "c"*64
    with pytest.raises(ValueError, match="codec identity"):
        make_dataset(args)
    codecs[0].checkpoint_sha256 = saved_hash
    p = Path(args.output)/"future_normalizer.json"
    profile = json.loads(p.read_text())
    profile["manifest_sha256"] = "d"*64
    profile["profile_sha256"] = norm.profile_sha256(profile)
    p.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="manifest hash"):
        make_dataset(args)


def test_codec_fit_leakage_rejected_before_output(prepared, tmp_path):
    args, _, codecs = prepared
    codecs[0].metadata = {"fit": {"episode_ids": ["ho"]}}
    args.output = str(tmp_path/"must_not_exist")
    with pytest.raises(ValueError, match="training split"):
        prep.prepare_fixed(args)
    assert not Path(args.output).exists()


def test_strict_tracking_rejects_old_tolerance_and_zero_pose():
    head, _, _, rp, _ = streams(4)
    head[1,3:] *= 1.0004  # accepted by old 1e-3, rejected before new runtime
    head[2] = 0
    rp[3] = 0
    raw = {"obs_head_pose": head, "right.obs_keypoints": rp}
    old = prep.invalid_frames(raw)
    new = prep.invalid_frames(raw, fixed_camera=True)
    assert not old["obs_head_pose:invalid_quaternion"][1]
    assert new["obs_head_pose:invalid_quaternion"].tolist() == [False,True,True,False]
    assert new["right.obs_keypoints:all_zero_keypoints"].tolist() == [False,False,False,True]
    assert not new["obs_head_pose:invalid_so3"].any()


def test_float32_overflow_is_not_a_valid_window():
    head, *_ = streams(2)
    head = head.astype(np.float64)
    head[1,0] = 1e100
    assert prep.invalid_frames({"obs_head_pose":head}, fixed_camera=True)["obs_head_pose:nonfinite"].tolist() == [False,True]


def test_flat_keypoints_match_audited_storage_shape():
    h, r, l, rp, lp = streams(41)
    codecs = (Codec("a"), Codec("b"))
    first = norm.encode_fixed_window(h,r,l,rp,lp,codecs)
    second = norm.encode_fixed_window(h,r,l,rp.reshape(41,63),lp.reshape(41,63),codecs)
    for a,b in zip(first,second):
        torch.testing.assert_close(a,b)


def test_out_of_fov_keeps_future_rows_but_invalid_tracking_rejects_window(prepared):
    args, _, _ = prepared
    group = prep.zarr.open_group("tr", mode="r")
    group["right.obs_palm_in_fov_front_1"][:] = 0
    group["left.obs_palm_in_fov_front_1"][:] = 0
    dataset = make_dataset(args)
    item = dataset[0]
    assert item["hand_visibility"].shape == (64, 2)
    assert not item["hand_visibility"].any()
    assert item["action"].shape == (64, 57)
    assert torch.isfinite(item["action"]).all()
    assert item["future_action_source_frame_indices"].tolist() == list(range(1, 65))
    # Even an invisible source frame between RGB samples must invalidate the
    # entire window. FOV must not turn corrupt tracking into a masked loss row.
    group["right.obs_keypoints"][17, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="audited window tracking"):
        dataset[0]


def test_full_codec_collection_is_deduplicated_and_split_bound(prepared):
    from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import collect
    args, _, _ = prepared
    manifest = Path(args.output) / "valid_windows.json"
    for split, episodes in (("train", args.train_episodes), ("heldout", args.heldout_episodes)):
        data, trajectories, provenance = collect(episodes, manifest, split, 1000, 1, 42, all_valid_frames=True)
        assert len(data["right"]) == len(data["left"]) == 65
        assert provenance["sampling"] is False
        assert provenance["episode_ids"] == (["tr"] if split == "train" else ["ho"])
        assert provenance["source_windows"][0]["sample_count"] == 65
        assert provenance["source_windows"][0]["sample_offset"] == 0
        assert trajectories["right"]


def test_explicit_preparation_tiers_get_matching_audit_and_refit(prepared, tmp_path):
    args, _, _ = prepared
    original=(Path(args.output)/"valid_windows.json").read_bytes()
    args.output=str(tmp_path/"different_tiers")
    args.clip_frames=[17]
    result=prep.prepare_fixed(args)
    manifest=json.loads((Path(args.output)/"valid_windows.json").read_text())
    assert {w["frames"] for w in manifest["windows"].values()}=={17}
    assert result["state_rows"]>0 and result["future_rows"]>0
    profile=json.loads((Path(args.output)/"future_normalizer.json").read_text())
    assert profile["manifest_sha256"]==hashlib.sha256((Path(args.output)/"valid_windows.json").read_bytes()).hexdigest()
    assert original != (Path(args.output)/"valid_windows.json").read_bytes()


def test_parallel_fixed_preparation_is_byte_identical(prepared, tmp_path):
    args, _, _ = prepared
    serial = Path(args.output)
    args.output = str(tmp_path / "parallel")
    args.workers = 2
    prep.prepare_fixed(args)
    for name in ("valid_windows.json", "fit_samples.json", "chunk_state_normalizer.json",
                 "future_normalizer.json", "eval_windows.json", "summary.json"):
        assert (serial / name).read_bytes() == (Path(args.output) / name).read_bytes(), name
