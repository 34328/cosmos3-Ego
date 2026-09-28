"""Audit joint windows and fit train-only boundary-state statistics before training."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import zarr

from .action import pose9, pose_matrices
from .ar_chunk_state import (
    chunk_boundaries,
    ChunkCameraStateNormalizer,
    state_profile_sha256,
    CHUNK_CAMERA_STATE_SCHEMA,
    CHUNK_CAMERA_LAYOUT_VERSION,
    VALID_WINDOWS_SCHEMA,
)
from .ar_dataset import ar_latent_frames, clip_span, select_clip_frames


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segment_id(row):
    return f"{row['episode_hash']}:{row['span_index']}:{row['start_idx']}:{row['end_idx']}"


def invalid_frames(streams):
    """Invalid tracking is distinct from FOV; visibility is not inspected here."""
    invalid = {}
    for name, values in streams.items():
        x = np.asarray(values)
        finite = np.isfinite(x).reshape(len(x), -1).all(1)
        invalid[name + ":nonfinite"] = ~finite
        if name.endswith("pose"):
            if x.ndim != 2 or x.shape[1] != 7:
                raise ValueError(f"{name} must have [T,7] pose values")
            norms = np.linalg.norm(x[:, 3:], axis=1)
            invalid[name + ":invalid_quaternion"] = finite & (np.abs(norms - 1) > 1e-3)
        else:
            if x.reshape(len(x), -1).shape[1] != 63:
                raise ValueError(f"{name} must have 21 3D keypoints")
            invalid[name + ":all_zero_keypoints"] = finite & (x.reshape(len(x), -1) == 0).all(1)
    return invalid


def audit(episodes_path, segments_path, split):
    episodes = {r["episode_hash"]: r for r in csv.DictReader(Path(episodes_path).open()) if r["split"] == split}
    segments = [r for r in csv.DictReader(Path(segments_path).open()) if r["split"] == split]
    if not episodes or not segments:
        raise ValueError(f"empty {split} manifest")
    fields = (
        "obs_head_pose",
        "right.obs_wrist_pose",
        "left.obs_wrist_pose",
        "right.obs_keypoints",
        "left.obs_keypoints",
    )
    windows, reports, streams_by_episode = {}, [], {}
    for episode_id, episode in episodes.items():
        group = zarr.open_group(episode["abs_zarr_path"], mode="r")
        streams = {name: np.asarray(group[name][:]) for name in fields}
        length = int(group.attrs["total_frames"])
        if any(len(x) != length for x in streams.values()):
            raise ValueError(f"{episode_id}: pose/keypoint length differs from episode")
        streams_by_episode[episode_id] = streams
        reasons = invalid_frames(streams)
        bad = np.logical_or.reduce(list(reasons.values()))
        prefix = np.r_[0, np.cumsum(bad)]
        for row in (r for r in segments if r["episode_hash"] == episode_id):
            start, end = int(row["start_idx"]), int(row["end_idx"])
            if start < 0 or end > length or start >= end:
                raise ValueError(f"{segment_id(row)} exceeds episode bounds")
            frames = select_clip_frames(end - start, 2)
            if frames is None:
                reports.append(
                    dict(
                        sample_id=segment_id(row),
                        episode=episode_id,
                        split=split,
                        reason="shorter_than_33",
                        candidates=0,
                    )
                )
                continue
            span = clip_span(frames, 2)
            starts = np.arange(start, end - span + 1)
            original_bad = prefix[starts + span] != prefix[starts]
            valid = starts[~original_bad]
            windows[segment_id(row)] = dict(
                episode=episode_id, split=split, frames=frames, span=span, starts=valid.tolist()
            )
            for c in (1, 2, 3, 4):
                boundary_offsets = np.array([b.source_start for b in chunk_boundaries(ar_latent_frames(frames), c)])
                boundary_bad = bad[starts[:, None] + boundary_offsets].any(1)
                extra = (~original_bad) & boundary_bad
                reason_counts = {}
                for name, flags in reasons.items():
                    p = np.r_[0, np.cumsum(flags)]
                    reason_counts[name] = int((p[starts + span] != p[starts]).sum())
                reports.append(
                    dict(
                        sample_id=segment_id(row),
                        episode=episode_id,
                        split=split,
                        C=c,
                        frames=frames,
                        candidates=len(starts),
                        original_excluded=int(original_bad.sum()),
                        state_extra_excluded=int(extra.sum()),
                        retained=int((~original_bad & ~boundary_bad).sum()),
                        reasons=reason_counts,
                    )
                )
                if extra.any():
                    raise AssertionError("state validation and full input validation disagree")
    covered = {w["episode"] for w in windows.values() if w["starts"]}
    if covered != set(episodes):
        raise ValueError(f"{split} has no valid windows for episodes {sorted(set(episodes)-covered)}")
    return windows, reports, streams_by_episode


def weighted_quantile(values, weights, quantile):
    result = []
    for dim in range(values.shape[1]):
        order = np.argsort(values[:, dim], kind="stable")
        x, w = values[order, dim], weights[order]
        result.append(np.interp(quantile, (np.cumsum(w) - 0.5 * w) / w.sum(), x))
    return np.array(result)


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    train, tr_report, train_streams = audit(args.train_episodes, args.train_segments, "train")
    held, ho_report, _ = audit(args.heldout_episodes, args.heldout_segments, "heldout")
    if {v["episode"] for v in train.values()} & {v["episode"] for v in held.values()}:
        raise ValueError("train and held-out episodes overlap")
    manifest = dict(
        schema=VALID_WINDOWS_SCHEMA,
        state_schema=CHUNK_CAMERA_STATE_SCHEMA,
        layout_version=CHUNK_CAMERA_LAYOUT_VERSION,
        frame_stride=2,
        seed=args.seed,
        source_hashes={
            name: sha256(getattr(args, name))
            for name in ("train_episodes", "train_segments", "heldout_episodes", "heldout_segments")
        },
        windows=train | held,
    )
    reference = getattr(args, "reference_data", None)
    if reference is not None:
        old = json.loads((Path(reference) / "valid_windows.json").read_text())
        if old["windows"] != manifest["windows"] or old["source_hashes"] != manifest["source_hashes"]:
            raise ValueError("reference windows/source manifests changed; refusing to shift evaluation sources")
        manifest["reference_manifest_sha256"] = sha256(Path(reference) / "valid_windows.json")
    manifest_path = output / "valid_windows.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "data_audit.json").write_text(json.dumps(tr_report + ho_report, indent=2) + "\n")
    rng = random.Random(args.seed)
    records, values, weights = [], [], []
    for sid, window in train.items():
        if not window["starts"]:
            continue
        raw = train_streams[window["episode"]]
        # Eight independent draws from the actual retained uniform window distribution.
        for draw in range(args.windows_per_segment):
            start = rng.choice(window["starts"])
            selected = slice(start, start + window["span"])
            poses = [
                pose_matrices(raw[name][selected])
                for name in ("obs_head_pose", "right.obs_wrist_pose", "left.obs_wrist_pose")
            ]
            # Every candidate uses its own camera, not the sampled window's F0.
            from_world = np.linalg.inv(poses[0])
            wrists = np.concatenate([pose9(from_world @ p) for p in poses[1:]], axis=1)
            for c in (1, 2, 3, 4):
                indexes = [b.source_start for b in chunk_boundaries(ar_latent_frames(window["frames"]), c)]
                records.append(dict(sample_id=sid, draw=draw, start=start, C=c, boundaries=indexes))
                values.append(wrists[indexes])
                weights.extend([1 / (args.windows_per_segment * 4 * len(indexes))] * len(indexes))
    values, weights = np.concatenate(values), np.array(weights)
    q01, q99 = weighted_quantile(values, weights, 0.01), weighted_quantile(values, weights, 0.99)
    floor = np.tile([0.01] * 3 + [0.05] * 6, 2)
    raw_scale = (q99 - q01) / 2
    samples_path = output / "state_fit_samples.json"
    samples_path.write_text(json.dumps(records, indent=2) + "\n")
    profile = dict(
        schema=CHUNK_CAMERA_STATE_SCHEMA,
        layout_version=CHUNK_CAMERA_LAYOUT_VERSION,
        camera_encoding="zero_identity",
        channels="right_pose9,left_pose9",
        split="train",
        frozen=False,
        fit_samples_sha256=sha256(samples_path),
        method="piecewise_asinh_rot",
        beta=1.0,
        manifest_sha256=sha256(manifest_path),
        seed=args.seed,
        windows_per_segment=args.windows_per_segment,
        num_boundary_rows=len(values),
        weighting="segment/clip equal; C=1..4 equal; boundaries within C equal",
        scale_floor=floor.tolist(),
        floor_reason="translation: 1 cm; rotation-column entries: 0.05 dimensionless",
        floor_channels=np.flatnonzero(raw_scale < floor).tolist(),
        stats=dict(
            center=((q01 + q99) / 2).tolist(),
            scale=np.maximum(raw_scale, floor).tolist(),
            q01=q01.tolist(),
            q99=q99.tolist(),
            minimum=values.min(0).tolist(),
            maximum=values.max(0).tolist(),
        ),
    )
    profile["profile_sha256"] = state_profile_sha256(profile)
    path = output / "chunk_state_normalizer.json"
    path.write_text(json.dumps(profile, indent=2) + "\n")
    normalizer = ChunkCameraStateNormalizer(path, require_frozen=False)
    source = torch.tensor(values, dtype=torch.float32)
    decoded = normalizer.denormalize(normalizer.normalize(source))
    torch.testing.assert_close(decoded, source, atol=1e-5, rtol=1e-4)
    profile["frozen"] = True
    profile["profile_sha256"] = state_profile_sha256(profile)
    path.write_text(json.dumps(profile, indent=2) + "\n")
    # Fixed evaluation draws cover every retained segment, with identical seeds for both models.
    evaluation = []
    for sid, window in held.items():
        if not window["starts"]:
            continue
        picks = sorted(set([window["starts"][0], window["starts"][-1], window["starts"][len(window["starts"]) // 2]]))
        evaluation.extend(
            dict(sample_id=sid, start=start, frames=window["frames"], seed=seed)
            for start in picks
            for seed in (42, 43, 44)
        )
    if reference is not None and evaluation != json.loads((Path(reference) / "eval_windows.json").read_text()):
        raise ValueError("evaluation window/seed selection changed")
    (output / "eval_windows.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    summary = dict(
        schema=CHUNK_CAMERA_STATE_SCHEMA,
        layout_version=CHUNK_CAMERA_LAYOUT_VERSION,
        fitted_channels=18,
        fit_samples_sha256=sha256(samples_path),
        eval_windows_sha256=sha256(output / "eval_windows.json"),
        train_segments=sum(bool(w["starts"]) for w in train.values()),
        heldout_segments=sum(bool(w["starts"]) for w in held.values()),
        eval_items=len(evaluation),
        boundary_rows=len(values),
        state_extra_excluded=sum(r.get("state_extra_excluded", 0) for r in tr_report + ho_report),
        max_roundtrip_error=float((decoded - source).abs().max()),
        normalizer_sha256=sha256(path),
        manifest_sha256=sha256(manifest_path),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("train-episodes", "train-segments", "heldout-episodes", "heldout-segments", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--reference-data", help="Verify existing valid/eval windows stay exactly unchanged")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows-per-segment", type=int, default=8)
    args = parser.parse_args()
    if args.windows_per_segment < 1:
        parser.error("windows-per-segment must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
