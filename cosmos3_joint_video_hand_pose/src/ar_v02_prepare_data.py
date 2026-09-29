"""Audit joint windows and fit train-only boundary-state statistics before training."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import multiprocessing as mp
import time

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


def invalid_frames(streams, *, fixed_camera=False):
    """Invalid tracking is distinct from FOV; visibility is not inspected here."""
    invalid = {}
    for name, values in streams.items():
        # Match runtime conversion before geometry; float64-finite values can
        # overflow in float32 and must not enter the audited fixed-camera set.
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.asarray(values, dtype=np.float32 if fixed_camera else None)
        finite = np.isfinite(x).reshape(len(x), -1).all(1)
        invalid[name + ":nonfinite"] = ~finite
        if name.endswith("pose"):
            if x.ndim != 2 or x.shape[1] != 7:
                raise ValueError(f"{name} must have [T,7] pose values")
            norms = np.linalg.norm(x[:, 3:], axis=1)
            invalid[name + ":invalid_quaternion"] = finite & (np.abs(norms - 1) > (1e-4 if fixed_camera else 1e-3))
            if fixed_camera:
                good = finite & ~invalid[name + ":invalid_quaternion"]
                invalid_so3 = np.zeros(len(x), dtype=bool)
                if good.any():
                    # pose_matrices uses the same conversion as geometry._stream.
                    r = pose_matrices(x[good])[:, :3, :3].astype(np.float32)
                    orthogonal = np.isclose(r.transpose(0,2,1) @ r, np.eye(3), atol=1e-5, rtol=1e-5).all((1,2))
                    proper = np.isclose(np.linalg.det(r), 1, atol=1e-5, rtol=1e-5)
                    invalid_so3[good] = ~(orthogonal & proper)
                invalid[name + ":invalid_so3"] = invalid_so3
        else:
            if x.reshape(len(x), -1).shape[1] != 63:
                raise ValueError(f"{name} must have 21 3D keypoints")
            invalid[name + ":all_zero_keypoints"] = finite & (x.reshape(len(x), -1) == 0).all(1)
    return invalid


def visibility_window_report(visibility, starts, span):
    """Audit existing current-frame FOV masks; never change tracking validity.

    Exposure counts weight each retained window once. Run lengths are over the
    union of retained future frames, so overlapping windows do not duplicate runs.
    """
    v = np.asarray(visibility)
    starts = np.asarray(starts, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 2 or not np.isin(v, (0, 1)).all():
        raise ValueError("visibility must be binary [source_frames, right/left]")
    if starts.ndim != 1 or span < 2 or (starts < 0).any() or (starts + span > len(v)).any():
        raise ValueError("visibility windows exceed source frames")
    coverage_delta = np.zeros(len(v) + 1, dtype=np.int64)
    np.add.at(coverage_delta, starts + 1, 1)
    np.add.at(coverage_delta, starts + span, -1)
    coverage = np.cumsum(coverage_delta[:-1])
    report = dict(scope="retained_windows_future_rows", window_count=len(starts),
                  future_row_exposures=int(coverage.sum()),
                  unique_future_frames=int((coverage > 0).sum()), hands={})
    for i, side in enumerate(("right", "left")):
        visible = v[:, i].astype(bool)
        recovery = visible & np.r_[False, ~visible[:-1]]
        hidden = ~visible & (coverage > 0)
        edges = np.diff(np.r_[False, hidden, False].astype(np.int8))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        hidden_prefix = np.r_[0, np.cumsum(~visible)]
        per_window = hidden_prefix[starts + span] - hidden_prefix[starts + 1]
        report["hands"][side] = dict(
            windows_with_masked_future=int((per_window > 0).sum()),
            masked_future_exposures=int(coverage[~visible].sum()),
            supervised_recovery_exposures=int(coverage[recovery].sum()),
            unique_supervised_recoveries=int(((coverage > 0) & recovery).sum()),
            max_masked_run_in_covered_union=int(lengths.max(initial=0)),
            masked_run_histogram={str(k): int(n) for k, n in sorted(Counter(lengths.tolist()).items())})
    return report


def audit(episodes_path, segments_path, split, *, chunk_sizes=(1, 2, 3, 4), fixed_camera=False,
          clip_frame_tiers=(129, 65, 33), include_visibility=False, retain_streams=True,
          require_all_episodes=True):
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
        if retain_streams:
            streams_by_episode[episode_id] = streams
        visibility = None
        if include_visibility:
            from .materialize_visibility import compute_visibility
            visibility = np.stack([
                np.asarray(group[side + ".obs_palm_in_fov_front_1"][:])
                if side + ".obs_palm_in_fov_front_1" in group
                else compute_visibility(group, length, side)
                for side in ("right", "left")], axis=-1)
            if visibility.shape != (length, 2):
                raise ValueError(f"{episode_id}: visibility length differs from episode")
        reasons = invalid_frames(streams, fixed_camera=fixed_camera)
        bad = np.logical_or.reduce(list(reasons.values()))
        prefix = np.r_[0, np.cumsum(bad)]
        for row in (r for r in segments if r["episode_hash"] == episode_id):
            start, end = int(row["start_idx"]), int(row["end_idx"])
            if start < 0 or end > length or start >= end:
                raise ValueError(f"{segment_id(row)} exceeds episode bounds")
            frames = select_clip_frames(end - start, 2, tiers=clip_frame_tiers)
            if frames is None:
                reports.append(
                    dict(
                        sample_id=segment_id(row),
                        episode=episode_id,
                        split=split,
                        reason="shorter_than_minimum_tier",
                        minimum_model_frames=min(clip_frame_tiers),
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
            visibility_report = (visibility_window_report(visibility, valid, span)
                                 if visibility is not None else None)
            for c in chunk_sizes:
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
                        visibility=visibility_report,
                    )
                )
                if extra.any():
                    raise AssertionError("state validation and full input validation disagree")
    covered = {w["episode"] for w in windows.values() if w["starts"]}
    if require_all_episodes and covered != set(episodes):
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
    if getattr(args, "representation", None) == "fixed_camera_wrist_local_delta_latent_v1":
        return prepare_fixed(args)
    if getattr(args, "representation", None) not in (None, "legacy_local_delta_absolute_hand_v1"):
        raise ValueError("unsupported or retired action representation; no implicit migration")
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


def snapshot_fixed_codecs(codecs, output):
    """Bundle validated weights/sidecars with statistics in the recipe directory.

    AE fitting and normalization use separate new output directories. Snapshot
    exact bytes here so the generated directory is directly consumable by config.
    """
    import shutil
    from .action_fixed_normalization import load_fixed_codecs
    destinations = []
    hashes = tuple(c.checkpoint_sha256 for c in codecs)
    for side, codec in zip(("right", "left"), codecs, strict=True):
        source = Path(codec.checkpoint_path)
        from .codec_fixed_camera import PCA_ARCHITECTURE
        suffix = "_pca15.pt" if codec.metadata.get("architecture") == PCA_ARCHITECTURE else "_mlp15.pt"
        destination = output / (side + suffix)
        shutil.copyfile(source, destination)
        if sha256(destination) != codec.checkpoint_sha256:
            raise ValueError("codec changed while preparing statistics")
        shutil.copyfile(source.with_suffix(".validation.json"), destination.with_suffix(".validation.json"))
        destinations.append(destination)
    frozen = load_fixed_codecs(destinations)
    if tuple(c.checkpoint_sha256 for c in frozen) != hashes:
        raise ValueError("bundled codec hashes differ from prepared source")
    return frozen


_FIXED_WORKER_INPUTS = None


def _init_fixed_worker(streams, codecs):
    global _FIXED_WORKER_INPUTS
    torch.set_num_threads(1)
    _FIXED_WORKER_INPUTS = (streams, codecs)


def _encode_fixed_draw(job, streams, codecs):
    from .action_fixed_normalization import encode_fixed_window
    sid, episode, start, span, draw = job
    raw = streams[episode]
    sl = slice(start, start + span)
    state, action = encode_fixed_window(*(raw[name][sl] for name in (
        "obs_head_pose", "right.obs_wrist_pose", "left.obs_wrist_pose",
        "right.obs_keypoints", "left.obs_keypoints")), codecs)
    return state[::4].numpy().copy(), action.numpy(), dict(
        sample_id=sid, split="train", start=start, draw=draw,
        C=4, K=8, future_rows=len(action), state_rows=len(state[::4]))


def _encode_fixed_worker(job):
    return _encode_fixed_draw(job, *_FIXED_WORKER_INPUTS)


def iter_fixed_draws(jobs, streams, codecs, *, workers=1):
    """Ordered CPU map: sampling happens in the parent, never in workers.

    Linux fork shares the already audited read-only source arrays and codecs.
    One Torch thread per worker avoids nested BLAS/OpenMP oversubscription.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for job in jobs:
            yield _encode_fixed_draw(job, streams, codecs)
        return
    if torch.cuda.is_initialized():
        raise RuntimeError("parallel preparation must run in a CPU-only process")
    with mp.get_context("fork").Pool(workers, initializer=_init_fixed_worker,
                                     initargs=(streams, codecs)) as pool:
        yield from pool.imap(_encode_fixed_worker, jobs, chunksize=4)


def prepare_fixed(args):
    """Audit both splits; fit *only train* with the exact runtime C4 encoder."""
    from .action_fixed_normalization import (
        REPRESENTATION, NORMALIZER_SCHEMA, VALID_WINDOWS_SCHEMA as FIXED_WINDOWS,
        load_fixed_codecs, encode_fixed_window, fit_fixed_normalizer, FixedCameraNormalizer,
    )
    if args.windows_per_segment < 1:
        raise ValueError("windows_per_segment must be positive")
    codecs = load_fixed_codecs((args.right_codec, args.left_codec))
    hashes = tuple(c.checkpoint_sha256 for c in codecs)
    tiers = tuple(getattr(args, "clip_frames", None) or (129, 65, 33))
    train, tr_report, streams = audit(args.train_episodes, args.train_segments, "train", chunk_sizes=(4,), fixed_camera=True, include_visibility=True, clip_frame_tiers=tiers)
    held, ho_report, _ = audit(args.heldout_episodes, args.heldout_segments, "heldout", chunk_sizes=(4,), fixed_camera=True, include_visibility=True, clip_frame_tiers=tiers)
    train_ids = {w["episode"] for w in train.values()}
    held_ids = {w["episode"] for w in held.values()}
    if train_ids & held_ids:
        raise ValueError("train and heldout episodes overlap")
    # Codec fitting must also be train-only with respect to THIS split.
    for codec in codecs:
        fitted = set(codec.metadata["fit"]["episode_ids"])
        if not fitted or not fitted <= train_ids or fitted & held_ids:
            raise ValueError("codec fit episodes are not a subset of this training split")
    manifest = dict(schema=FIXED_WINDOWS, state_schema=NORMALIZER_SCHEMA,
                    representation=REPRESENTATION, layout_version=CHUNK_CAMERA_LAYOUT_VERSION,
                    frame_stride=2, chunk_size=4, tokens_per_latent=8, codec_sha256=list(hashes),
                    tracking_validation=dict(version="fixed_camera_float32_v1", quaternion_norm_atol=1e-4,
                                             so3_atol=1e-5, so3_rtol=1e-5, missing_hand="exclude_entire_window",
                                             finite_dtype="float32", scope="all_source_frames"),
                    seed=args.seed, source_hashes={name: sha256(getattr(args, name)) for name in
                    ("train_episodes", "train_segments", "heldout_episodes", "heldout_segments")},
                    windows=train | held)
    reference = getattr(args, "reference_data", None)
    if reference:
        old = json.loads((Path(reference) / "valid_windows.json").read_text())
        if old["windows"] != manifest["windows"] or old["source_hashes"] != manifest["source_hashes"]:
            raise ValueError("reference windows/source manifests changed")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    codecs = snapshot_fixed_codecs(codecs, output)
    manifest_path = output / "valid_windows.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "data_audit.json").write_text(json.dumps(tr_report + ho_report, indent=2) + "\n")
    rng = random.Random(args.seed)
    states, futures, records = [], [], []
    jobs = []
    for sid, window in train.items():
        if not window["starts"]:
            continue
        for draw in range(args.windows_per_segment):
            start = rng.choice(window["starts"])
            jobs.append((sid, window["episode"], start, window["span"], draw))
    workers = getattr(args, "workers", 1)
    started = time.monotonic()
    for done, (state, action, record) in enumerate(
            iter_fixed_draws(jobs, streams, codecs, workers=workers), 1):
        states.append(state)
        futures.append(action)
        records.append(record)
        if done % 512 == 0 or done == len(jobs):
            elapsed = time.monotonic() - started
            print(json.dumps(dict(stage="encode_windows", completed=done, total=len(jobs),
                                  workers=workers, elapsed_seconds=round(elapsed, 2),
                                  windows_per_second=round(done / max(elapsed, 1e-6), 2))), flush=True)
    if not states:
        raise ValueError("no retained train rows for fixed-camera statistics")
    samples = output / "fit_samples.json"
    samples.write_text(json.dumps(records, indent=2) + "\n")
    summary = dict(representation=REPRESENTATION, chunk_size=4, tokens_per_latent=8,
                   codec_sha256=list(hashes), manifest_sha256=sha256(manifest_path),
                   fit_samples_sha256=sha256(samples), train_segments=sum(bool(w["starts"]) for w in train.values()),
                   heldout_segments=sum(bool(w["starts"]) for w in held.values()))
    for kind, rows, name in (("state", states, "chunk_state_normalizer.json"),
                              ("future", futures, "future_normalizer.json")):
        values = np.concatenate(rows)
        profile = fit_fixed_normalizer(values, kind=kind, codec_sha256=hashes,
                                       manifest_sha256=sha256(manifest_path))
        from .action_fixed_normalization import profile_sha256
        profile.update(fit_samples_sha256=sha256(samples), seed=args.seed,
                       windows_per_segment=args.windows_per_segment,
                       weighting="uniform train-window draws; equal rows within state/future separately")
        profile["profile_sha256"] = profile_sha256(profile)
        norm = FixedCameraNormalizer(profile, kind=kind, codec_sha256=hashes)
        x = torch.from_numpy(values)
        torch.testing.assert_close(norm.denormalize(norm.normalize(x)), x, atol=1e-5, rtol=1e-5)
        path = output / name
        path.write_text(json.dumps(profile, indent=2) + "\n")
        summary[kind + "_rows"] = len(values)
        summary[kind + "_normalizer_sha256"] = sha256(path)
    evaluation = []
    for sid, w in held.items():
        if w["starts"]:
            picks = sorted(set((w["starts"][0], w["starts"][-1], w["starts"][len(w["starts"])//2])))
            evaluation.extend(dict(sample_id=sid, start=s, frames=w["frames"], seed=seed)
                              for s in picks for seed in (42, 43, 44))
    if reference and evaluation != json.loads((Path(reference) / "eval_windows.json").read_text()):
        raise ValueError("reference evaluation windows changed")
    (output / "eval_windows.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def audit_only(args):
    tiers = tuple(args.clip_frames or (129, 65, 33))
    if any(t < 5 or (t - 1) % 4 for t in tiers):
        raise ValueError("model video tiers must be 1+4N frames, N>=1")
    from .action_fixed_normalization import REPRESENTATION
    if args.representation != REPRESENTATION:
        raise ValueError("audit-only requires the explicit current action representation")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    reports, summary, split_ids, audited_windows = [], {}, {}, {}
    for split, ep, seg in (("train", args.train_episodes, args.train_segments),
                           ("heldout", args.heldout_episodes, args.heldout_segments)):
        windows, rows, _ = audit(ep, seg, split, chunk_sizes=(4,), fixed_camera=True,
                                clip_frame_tiers=tiers, include_visibility=True,
                                retain_streams=False, require_all_episodes=False)
        split_ids[split] = {r["episode_hash"] for r in csv.DictReader(Path(ep).open()) if r["split"] == split}
        reports.extend(rows)
        audited_windows.update(windows)
        summary[split] = dict(segments=len(rows), retained_segments=sum(bool(w["starts"]) for w in windows.values()),
                             retained_windows=sum(len(w["starts"]) for w in windows.values()),
                             retained_episodes=len({w["episode"] for w in windows.values() if w["starts"]}),
                             excluded_windows=sum(r.get("original_excluded", 0) for r in rows))
    if split_ids["train"] & split_ids["heldout"]:
        raise ValueError("train and heldout episodes overlap")
    summary.update(representation=REPRESENTATION, clip_frame_tiers=list(tiers), frame_stride=2,
                   chunk_size=4, tokens_per_latent=8, statistics_fitted=False,
                   visibility_policy="report_only_current_frame_FOV_mask_unchanged",
                   source_hashes={name: sha256(getattr(args, name)) for name in
                    ("train_episodes", "train_segments", "heldout_episodes", "heldout_segments")})
    # Bootstrap AE fitting without depending on an AE-bound normalizer manifest.
    # This schema is source-audit-only and is deliberately rejected by the trainer.
    source_manifest = dict(schema="ar_v02_codec_source_windows_v1", representation=REPRESENTATION,
                           frame_stride=2, chunk_size=4, tokens_per_latent=8,
                           clip_frame_tiers=list(tiers), source_hashes=summary["source_hashes"],
                           tracking_validation=dict(version="fixed_camera_float32_v1", quaternion_norm_atol=1e-4,
                               so3_atol=1e-5, so3_rtol=1e-5, missing_hand="exclude_entire_window",
                               finite_dtype="float32", scope="all_source_frames"), windows=audited_windows)
    (output / "valid_windows.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    summary["valid_windows_sha256"] = sha256(output / "valid_windows.json")
    (output / "data_audit.json").write_text(json.dumps(reports, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    for name in ("train-episodes", "train-segments", "heldout-episodes", "heldout-segments", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--reference-data", help="Verify existing valid/eval windows stay exactly unchanged")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows-per-segment", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1, help="CPU processes for fixed-camera window encoding; one thread each")
    parser.add_argument("--representation", choices=("legacy_local_delta_absolute_hand_v1",
                        "fixed_camera_wrist_local_delta_latent_v1"), default="legacy_local_delta_absolute_hand_v1")
    parser.add_argument("--audit-only", action="store_true", help="Audit valid windows/FOV gaps without codecs or fitting")
    parser.add_argument("--clip-frames", type=int, nargs="+", help="Explicit audited tiers; fixed-camera preparation refits statistics in a new output directory")
    parser.add_argument("--right-codec")
    parser.add_argument("--left-codec")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.audit_only:
        audit_only(args)
        return
    if args.clip_frames is not None and args.representation != "fixed_camera_wrist_local_delta_latent_v1":
        parser.error("custom preparation tiers require the fixed-camera representation")
    if args.windows_per_segment < 1:
        parser.error("windows-per-segment must be positive")
    if args.representation == "fixed_camera_wrist_local_delta_latent_v1" and not (args.right_codec and args.left_codec):
        parser.error("fixed-camera preparation requires --right-codec and --left-codec")
    prepare(args)


if __name__ == "__main__":
    main()
