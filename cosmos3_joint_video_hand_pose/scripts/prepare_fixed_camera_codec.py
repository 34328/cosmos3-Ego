"""Fit wrist-local AE15 with train-only statistics (bounded standalone codec job).

Reuses the existing FrozenHandMLPAE15 encoder/decoder architecture and checkpoint
keys. No dependency on retired experiment launchers or WAM training lifecycle.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import zarr
from cosmos3_joint_video_hand_pose.src.action import pose_matrices
from cosmos3_joint_video_hand_pose.src.codec_fixed_camera import (
    ARCHITECTURE, INPUT_FRAME, REPRESENTATION, stats_hash, tensor_hash, validate_source_provenance,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary_indices(span):
    """C=4, K=8: 32 source-frame transitions, plus a possible short tail."""
    if span < 2:
        raise ValueError('window must include boundary and at least one future frame')
    return list(range(0, span - 1, 32)) + [span - 1]


def validate_window_manifest(manifest, episodes_digest, split):
    """Accept current source audits without requiring an already trained AE."""
    from cosmos3_joint_video_hand_pose.src.action_fixed_normalization import VALID_WINDOWS_SCHEMA
    schema = manifest.get("schema")
    if schema not in ("ar_v02_codec_source_windows_v1", VALID_WINDOWS_SCHEMA):
        raise ValueError("codec windows require a current source-audit or fixed-camera schema")
    expected = dict(representation=REPRESENTATION, frame_stride=2, chunk_size=4, tokens_per_latent=8)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"codec window manifest incompatible {key}")
    tracking = dict(version="fixed_camera_float32_v1", quaternion_norm_atol=1e-4,
                    so3_atol=1e-5, so3_rtol=1e-5, missing_hand="exclude_entire_window",
                    finite_dtype="float32", scope="all_source_frames")
    if any(manifest.get("tracking_validation", {}).get(k) != v for k,v in tracking.items()):
        raise ValueError("codec window tracking_validation differs from fixed-camera audit")
    sources = manifest.get("source_hashes", {})
    for key in ("train_episodes", "train_segments", "heldout_episodes", "heldout_segments"):
        digest = sources.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"codec window manifest missing source hash: {key}")
    if sources[split + "_episodes"] != episodes_digest:
        raise ValueError(f"codec source {split}_episodes hash mismatch; rerun source audit")
    if schema == VALID_WINDOWS_SCHEMA:
        hashes = manifest.get("codec_sha256", [])
        if not isinstance(hashes, list) or len(hashes) != 2 or any(
                not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
            raise ValueError("formally prepared windows require both codec hashes")
    if not isinstance(manifest.get("windows"), dict) or not manifest["windows"]:
        raise ValueError("codec window manifest has no audited windows")
    return sources


def fixed_offsets(head, wrist, keypoints, boundary=0):
    """Current wrist-local q=Rw.T@(kp-pw); head/boundary never affect shape.

    Keep the collector call signature stable, but use every frame's wrist pose.
    """
    wrist = np.asarray(wrist)
    if wrist.ndim != 2 or wrist.shape[-1] != 7 or not np.isfinite(wrist).all():
        raise ValueError("wrist poses must be finite [N,7]")
    if not np.allclose(np.linalg.norm(wrist[:, 3:], axis=-1), 1, atol=1e-4, rtol=1e-4):
        raise ValueError("invalid wrist quaternion")
    rotation = pose_matrices(wrist)[:, :3, :3]
    points = np.asarray(keypoints)
    if points.ndim == 2 and points.shape[-1] == 63:
        points = points.reshape(-1, 21, 3)
    if points.shape[-2:] != (21, 3):
        raise ValueError("keypoints must have 21 world-space joints")
    if len(points) != len(wrist):
        raise ValueError("wrist/keypoint frame count mismatch")
    offset_world = points[:, 1:] - np.asarray(wrist)[:, None, :3]
    q = np.einsum("tji,tkj->tki", rotation, offset_world)
    if not np.isfinite(q).all():
        raise ValueError("non-finite wrist-local hand offsets")
    return q.astype(np.float32)


def audited_frame_ranges(choices):
    """Union audited source windows; count each source frame once per episode."""
    ranges = []
    for sid, window in choices:
        local = []
        for start in sorted(set(window["starts"])):
            end = start + window["span"]
            if local and start <= local[-1][1]:
                local[-1][1] = max(local[-1][1], end)
            else:
                local.append([start, end])
        ranges.extend((start, end, [sid]) for start, end in local)
    merged = []
    for start, end, ids in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].extend(ids)
        else:
            merged.append([start, end, ids])
    return merged


def collect(episodes_path, windows_path, split, max_episodes, windows_per_episode, seed, *, all_valid_frames=False):
    """Deterministic sample of audited windows; retain complete 32-frame chunks."""
    if min(max_episodes, windows_per_episode) < 1:
        raise ValueError("sample bounds must be positive")
    if split not in ("train", "heldout"):
        raise ValueError("split must be train or heldout")
    source_hashes = (sha256(episodes_path), sha256(windows_path))
    rows = list(csv.DictReader(Path(episodes_path).open()))
    ids = [r["episode_hash"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode IDs (potential split leakage)")
    episodes = {r["episode_hash"]: r for r in rows}
    manifest = json.loads(Path(windows_path).read_text())
    audited_sources = validate_window_manifest(manifest, source_hashes[0], split)
    split_ids = {name: set() for name in ("train", "heldout")}
    by_episode = {}
    for sid, w in manifest["windows"].items():
        eid = w["episode"]
        if w.get("split") not in split_ids:
            raise ValueError("unknown audited window split")
        split_ids[w["split"]].add(eid)
        if w["split"] != split:
            continue  # train/heldout CSVs may be separate source files
        if eid not in episodes or w["split"] != episodes[eid]["split"]:
            raise ValueError("window/episode split mismatch")
        if (type(w.get("span")) is not int or w["span"] < 2 or not isinstance(w.get("starts"), list)
                or any(type(s) is not int or s < 0 for s in w["starts"])):
            raise ValueError("invalid audited window span/starts")
        if w["starts"]:
            by_episode.setdefault(eid, []).append((sid, w))
    if split_ids["train"] & split_ids["heldout"]:
        raise ValueError("audited train/heldout episode overlap")
    rng = np.random.default_rng(seed)
    selected = sorted(by_episode)
    if not all_valid_frames:
        rng.shuffle(selected)
    selected = selected[:max_episodes]
    if not selected:
        raise ValueError("no audited windows for requested split")
    samples = {s: [] for s in ("right", "left")}
    trajectories = {s: [] for s in samples}
    sources = []
    sample_offset = 0
    for episode_index, eid in enumerate(selected):
        group = zarr.open_group(episodes[eid]["abs_zarr_path"], mode="r")
        choices = by_episode[eid]
        if all_valid_frames:
            plans = [("audited-union", start, end-start, ids)
                     for start, end, ids in audited_frame_ranges(choices)]
        else:
            plans = []
            for _ in range(windows_per_episode):
                sid, w = choices[int(rng.integers(len(choices)))]
                plans.append((sid, int(rng.choice(w["starts"])), int(w["span"]), [sid]))
        for sid, start, span, window_ids in plans:
            stop = start + span
            fields = ("obs_head_pose", "right.obs_wrist_pose", "left.obs_wrist_pose",
                      "right.obs_keypoints", "left.obs_keypoints")
            length = int(group.attrs["total_frames"])
            if stop > length or any(group[name].shape[0] != length for name in fields):
                raise ValueError("audited window outside source")
            streams = {name: np.asarray(group[name][start:stop]) for name in fields}
            if any(len(x) != span for x in streams.values()):
                raise ValueError("audited window source stream length mismatch")
            from cosmos3_joint_video_hand_pose.src.ar_v02_prepare_data import invalid_frames
            reasons = invalid_frames(streams, fixed_camera=True)
            failed = [name for name, flags in reasons.items() if flags.any()]
            if failed:
                raise ValueError(f"source tracking changed/invalid in {sid} at {start}: {failed}")
            # Use the exact arrays just validated, with the runtime FP32 conversion.
            streams = {name: x.astype(np.float32) for name,x in streams.items()}
            head = streams["obs_head_pose"]
            rotations = pose_matrices(head)[:, :3, :3]
            trajectory_indices = boundary_indices(span)
            boundaries = trajectory_indices[:-1]
            # Preserve camera trajectory for geometric validation, never E/D rebasing.
            sources.append({"episode": eid, "window": sid, "start": start, "span": span,
                            "input_sha256": tensor_hash({name:torch.from_numpy(x) for name,x in streams.items()})})
            if all_valid_frames:
                sources[-1].update(window_ids=window_ids, sample_offset=sample_offset, sample_count=span)
                sample_offset += span
            for side in samples:
                wrist = streams[side + ".obs_wrist_pose"]
                points = streams[side + ".obs_keypoints"]
                if all_valid_frames:
                    samples[side].append(torch.from_numpy(fixed_offsets(head, wrist, points)))
                else:
                    for b in boundaries:
                        end = min(b + 33, span)
                        q = fixed_offsets(head, wrist[b:end], points[b:end], b)
                        samples[side].append(torch.from_numpy(q))
                q0 = fixed_offsets(head, wrist[:1], points[:1])[0]
                # Real camera rotations remain diagnostic inputs, not AE inputs.
                rebase = np.einsum("tji,jk->tik", rotations[trajectory_indices], rotations[0])
                trajectories[side].append((q0, rebase.astype(np.float32)))
        if all_valid_frames and ((episode_index+1) % 50 == 0 or episode_index+1 == len(selected)):
            print(f"collect {split}: {episode_index+1}/{len(selected)} episodes, {sample_offset} unique frames", flush=True)
    data = {side: torch.cat(parts) for side, parts in samples.items()}
    if source_hashes != (sha256(episodes_path), sha256(windows_path)):
        raise ValueError("source manifests changed while collecting codec data")
    provenance = dict(split=split, episode_ids=selected, source_windows=sources,
                      source_hashes=audited_sources, window_schema=manifest["schema"],
                      representation=REPRESENTATION, input_frame=INPUT_FRAME,
                      episodes_path=str(Path(episodes_path).resolve()), manifest_path=str(Path(windows_path).resolve()),
                      manifest_sha256=sha256(windows_path), episodes_sha256=sha256(episodes_path),
                      sampling_seed=seed, windows_per_episode=windows_per_episode)
    if all_valid_frames:
        provenance.update(coverage="all unique source frames covered by audited windows, including boundaries",
                          windows_per_episode=None, sampling=False)
    return data, trajectories, provenance


def ae_losses(encoder, decoder, target, consistency_weight=0.):
    """Standardized-coordinate objectives; consistency target/input are detached."""
    if not np.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError('consistency weight must be finite and nonnegative')
    reconstruction = decoder(encoder(target))
    reconstruction_loss = (reconstruction-target).square().mean()
    consistency_loss = reconstruction_loss.new_zeros(())
    if consistency_weight:
        fixed_target = reconstruction.detach()
        consistency_loss = (decoder(encoder(fixed_target))-fixed_target).square().mean()
    return reconstruction_loss + consistency_weight*consistency_loss, reconstruction_loss, consistency_loss


def train_side(x, provenance, side, steps, batch_size, lr, seed, device, consistency_weight=0.):
    """Same MLP shape as the existing AE; statistics fitted only on x (train)."""
    if provenance["split"] != "train" or len(x) < 2:
        raise ValueError("fit must use train-only samples")
    validate_source_provenance(provenance, "train")
    if x.ndim != 3 or x.shape[-2:] != (20,3) or not torch.isfinite(x).all():
        raise ValueError("fit requires finite wrist-local [N,20,3] samples")
    if not np.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError('consistency weight must be finite and nonnegative')
    torch.manual_seed(seed)
    encoder = nn.Sequential(nn.Linear(60,64), nn.SiLU(), nn.Linear(64,32), nn.SiLU(), nn.Linear(32,15)).to(device)
    decoder = nn.Sequential(nn.Linear(15,32), nn.SiLU(), nn.Linear(32,64), nn.SiLU(), nn.Linear(64,60)).to(device)
    flat = x.reshape(-1, 60).to(device)
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
    normalized = (flat - mean) / std
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    history = []
    for step in range(steps):
        idx = torch.randint(len(flat), (min(batch_size, len(flat)),), device=device)
        loss, reconstruction_loss, consistency_loss = ae_losses(encoder, decoder, normalized[idx], consistency_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == steps:
            history.append({"step": step+1, "standardized_coordinate_mse": float(reconstruction_loss.detach()),
                            "ae_reconstruction_loss": float(reconstruction_loss.detach()),
                            "ae_consistency_loss": float(consistency_loss.detach()), "ae_total_loss": float(loss.detach())})
            print(side, history[-1], flush=True)
    with torch.no_grad():
        latent = torch.cat([encoder(v) for v in normalized.split(8192)])
        latent_mean, latent_std = latent.mean(0), latent.std(0).clamp_min(1e-6)
    state = {"encoder."+k:v.detach().cpu() for k,v in encoder.state_dict().items()}
    state.update({"decoder."+k:v.detach().cpu() for k,v in decoder.state_dict().items()})
    state.update(mean=mean.cpu(), std=std.cpu())
    data_hash = tensor_hash({"q": x})
    state["wrist_local_data_binding"] = torch.tensor(list(bytes.fromhex(data_hash)), dtype=torch.uint8)
    payload = dict(schema_version=2, representation=REPRESENTATION,
        coordinate_system="fixed-camera", input_frame=INPUT_FRAME, architecture=ARCHITECTURE,
        chunk_size=4, action_tokens_per_latent=8, units="m", side=side,
        state_dict=state, latent_mean=latent_mean.cpu(), latent_std=latent_std.cpu(),
        fit={**provenance, "data_sha256":data_hash, "sample_count":len(x)},
        training=dict(steps=steps,batch_size=batch_size,learning_rate=lr,seed=seed,
                      objective="standardized coordinate reconstruction MSE + weight * detached E/D consistency MSE",
                      consistency_weight=consistency_weight, consistency_target="D(E(q)).detach() in standardized input coordinates",
                      initialization="random",
                      scope="bounded codec adaptation, not WAM training", history=history))
    payload["stats_sha256"] = stats_hash(payload)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", required=True)
    p.add_argument("--valid-windows", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-episodes", type=int, default=128)
    p.add_argument("--windows-per-episode", type=int, default=2)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--consistency-weight", type=float, default=0., help="AE-only detached E/D consistency weight; no WAM loss changes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if min(a.steps,a.batch_size) <= 0 or not np.isfinite(a.lr) or a.lr <= 0:
        p.error("positive steps/batch size/lr required")
    if not np.isfinite(a.consistency_weight) or a.consistency_weight < 0:
        p.error('consistency weight must be finite and nonnegative')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    data, _, provenance = collect(a.episodes,a.valid_windows,"train",a.max_episodes,a.windows_per_episode,a.seed)
    paths = {}
    for i, side in enumerate(("right","left")):
        payload = train_side(data[side],provenance,side,a.steps,a.batch_size,a.lr,a.seed+i,a.device,a.consistency_weight)
        path = out / (side + "_mlp15.pt")
        torch.save(payload,path)
        paths[side] = {"path":str(path.resolve()),"sha256":sha256(path)}
    (out/"checkpoints.json").write_text(json.dumps(paths,indent=2)+"\n")
    print(json.dumps(paths),flush=True)


if __name__ == "__main__":
    main()
