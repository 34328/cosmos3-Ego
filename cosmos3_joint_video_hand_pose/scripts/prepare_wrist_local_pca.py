"""Fit independent PCA15 codecs on audited train frames; no neural-network training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np
import torch

from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import collect, sha256
from cosmos3_joint_video_hand_pose.scripts.validate_fixed_camera_codec import reanchor_invariance
from cosmos3_joint_video_hand_pose.src.codec_fixed_camera import (
    FrozenFixedCameraHandCodec, PCA_ARCHITECTURE, INPUT_FRAME, REPRESENTATION,
    stats_hash, tensor_hash, validate_source_provenance, validate_source_pair,
)


def fit_pca(x, provenance, side, seed=42):
    validate_source_provenance(provenance, "train")
    if x.ndim != 3 or x.shape[1:] != (20, 3) or len(x) < 16 or not torch.isfinite(x).all():
        raise ValueError("PCA requires at least 16 finite wrist-local hand samples")
    flat = x.reshape(-1, 60).cpu()
    # Exact covariance eigensolver, no randomized SVD and no per-coordinate scaling.
    mean = flat.double().mean(0)
    covariance = torch.zeros(60, 60, dtype=torch.float64)
    for part in flat.split(16384):
        centered = part.double() - mean
        covariance += centered.T @ centered
    covariance /= len(flat) - 1
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    components = vectors.flip(1)[:, :15].T.contiguous()
    # Deterministic sign convention: largest-magnitude entry positive.
    pivots = components.abs().argmax(1)
    signs = torch.sign(components[torch.arange(15), pivots])
    components *= signs[:, None]
    if eigenvalues.sum() <= 0:
        raise ValueError("zero-variance training set")
    data_hash = tensor_hash({"q": x})
    state = dict(mean=mean.float(), std=torch.ones(60), components=components.float(),
                 wrist_local_data_binding=torch.tensor(list(bytes.fromhex(data_hash)), dtype=torch.uint8))
    payload = dict(
        schema_version=2, representation=REPRESENTATION, coordinate_system="fixed-camera",
        input_frame=INPUT_FRAME, architecture=PCA_ARCHITECTURE, chunk_size=4,
        action_tokens_per_latent=8, units="m", side=side, state_dict=state,
        latent_mean=torch.zeros(15), latent_std=eigenvalues[:15].sqrt().clamp_min(1e-8).float(),
        fit={**provenance, "data_sha256": data_hash, "sample_count": len(x)},
        fitting=dict(method="float64 centered physical covariance eigh; deterministic component signs",
                     seed=seed, randomized=False, input_coordinate_scaling=False,
                     explained_variance=eigenvalues[:15].tolist(),
                     explained_variance_ratio=(eigenvalues[:15]/eigenvalues.sum()).tolist(),
                     explained_variance_ratio_sum=float(eigenvalues[:15].sum()/eigenvalues.sum()),
                     total_variance_m2=float(eigenvalues.sum())),
    )
    payload["stats_sha256"] = stats_hash(payload)
    return payload


def error_summary(error):
    values = np.asarray(error, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("empty/non-finite reconstruction error")
    return dict(mean_mm=float(values.mean()), p50_mm=float(np.quantile(values, .5)),
                p90_mm=float(np.quantile(values, .9)), p95_mm=float(np.quantile(values, .95)),
                p99_mm=float(np.quantile(values, .99)), max_mm=float(values.max()))


@torch.no_grad()
def evaluate_pca(codec, q, trajectories, provenance):
    errors = []
    for part in q.split(8192):
        errors.append((codec.decode(codec.encode(part)) - part).norm(dim=-1).numpy()*1000)
    errors = np.concatenate(errors)
    per_episode = {}
    for record in provenance["source_windows"]:
        a, n = record["sample_offset"], record["sample_count"]
        per_episode.setdefault(record["episode"], []).append(errors[a:a+n])
    episodes = {eid: dict(frames=sum(len(x) for x in parts),
                          reconstruction=error_summary(np.concatenate(parts)))
                for eid, parts in per_episode.items()}
    z = codec.encode(q[:min(len(q), 1024)])
    first = codec.decode(z)
    differences = [float((codec.decode(z) - first).abs().max()) for _ in range(10)]
    repeat = dict(passed=max(differences) == 0, repeats=10, latent_count=len(z),
                  max_abs_difference_m=max(differences), reencoding=False)
    histogram_edges = [0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 100, float("inf")]
    hist, _ = np.histogram(errors, bins=histogram_edges)
    return dict(
        reconstruction=error_summary(errors), frame_mean=error_summary(errors.mean(-1)),
        error_scope="Euclidean mm per non-wrist joint, all unique audited frames, no FOV mask",
        histogram=dict(edges_mm=histogram_edges[:-1]+["infinity"], counts=hist.tolist()),
        per_episode=episodes, repeat_decode=repeat,
        reanchor_invariance=reanchor_invariance(codec, *trajectories[0]),
        trajectory_count=len(trajectories),
    )


def passes(metrics):
    return (metrics["reconstruction"]["mean_mm"] <= 5
            and metrics["reconstruction"]["p95_mm"] <= 15
            and metrics["repeat_decode"]["passed"]
            and metrics["reanchor_invariance"]["passed"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", required=True)
    p.add_argument("--valid-windows", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(a.seed)
    rows = list(csv.DictReader(Path(a.episodes).open()))
    expected = {split: {r["episode_hash"] for r in rows if r["split"] == split}
                for split in ("train", "heldout")}
    if len(expected["train"]) != 744 or len(expected["heldout"]) != 119 or expected["train"] & expected["heldout"]:
        raise ValueError("expected current disjoint train744/heldout119 episode split")
    manifest = dict(schema_version=1, status="fitting", architecture=PCA_ARCHITECTURE,
                    representation=REPRESENTATION, input_frame=INPUT_FRAME, seed=a.seed,
                    train_episode_ids=sorted(expected["train"]), heldout_episode_ids=sorted(expected["heldout"]),
                    thresholds=dict(mean_mm=5, p95_mm=15),
                    environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                    checkpoints={})
    manifest_path = out / "manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    save()
    import wandb
    run = wandb.init(entity="alexlzh431564", project="joint_video_hand_pose", mode="online",
                     name="v3_wrist_local_pca15_train744", job_type="pca-fit-validation",
                     dir=str(out), config=dict(codec="PCA15", input_dim=60, latent_dim=15,
                     train_episodes=744, heldout_episodes=119, seed=a.seed, max_mean_mm=5, max_p95_mm=15),
                     settings=wandb.Settings(disable_git=True, save_code=False))
    if run.offline:
        raise RuntimeError("PCA record must be online")
    manifest["wandb_url"], manifest["wandb_id"] = run.url, run.id
    save()
    print(json.dumps(dict(wandb_url=run.url, stage="collect train")), flush=True)
    try:
        train, _, provenance = collect(a.episodes, a.valid_windows, "train", 744, 1, a.seed, all_valid_frames=True)
        if set(provenance["episode_ids"]) != expected["train"]:
            raise ValueError("audit does not cover all 744 train episodes")
        manifest["source"] = {key: provenance[key] for key in (
            "episodes_path", "episodes_sha256", "manifest_path", "manifest_sha256", "source_hashes", "coverage")}
        root = Path(__file__).resolve().parents[2]
        manifest["source_code_sha256"] = {name: sha256(root/name) for name in (
            "cosmos3_joint_video_hand_pose/scripts/prepare_wrist_local_pca.py",
            "cosmos3_joint_video_hand_pose/scripts/prepare_fixed_camera_codec.py",
            "cosmos3_joint_video_hand_pose/src/codec_fixed_camera.py")}
        for side in ("right", "left"):
            payload = fit_pca(train[side], provenance, side, a.seed)
            path = out / (side+"_pca15.pt")
            torch.save(payload, path)
            manifest["checkpoints"][side] = dict(path=path.name, sha256=sha256(path),
                data_sha256=payload["fit"]["data_sha256"], sample_count=len(train[side]),
                fitting=payload["fitting"])
            run.log({f"{side}/explained_variance_ratio": payload["fitting"]["explained_variance_ratio_sum"],
                     f"{side}/fit_frames": len(train[side])})
            save()
            print(json.dumps(dict(side=side, fit_frames=len(train[side]), fitting=payload["fitting"])), flush=True)
        del train
        manifest["status"] = "validating"
        save()
        held, trajectories, held_provenance = collect(a.episodes, a.valid_windows, "heldout", 119, 1, a.seed,
                                                     all_valid_frames=True)
        if set(held_provenance["episode_ids"]) != expected["heldout"]:
            raise ValueError("audit does not cover all 119 heldout episodes")
        reports = {}
        for side in ("right", "left"):
            path = out / (side+"_pca15.pt")
            codec = FrozenFixedCameraHandCodec(path, allow_unvalidated=True)
            validate_source_pair(codec.metadata["fit"], held_provenance)
            metrics = evaluate_pca(codec, held[side], trajectories[side], held_provenance)
            report = dict(schema_version=2, architecture=PCA_ARCHITECTURE, input_frame=INPUT_FRAME,
                representation=REPRESENTATION, checkpoint_sha256=sha256(path),
                heldout_episode_ids=held_provenance["episode_ids"], heldout_provenance=held_provenance,
                heldout_sample_count=len(held[side]), heldout_data_sha256=tensor_hash({"q":held[side]}),
                metrics=metrics, thresholds=dict(reconstruction_mean_mm=5, reconstruction_p95_mm=15),
                passed=passes(metrics), gate_scope="AE/PCA quality only; WAM untested")
            (out/(side+"_pca15.evaluation.json")).write_text(json.dumps(report, indent=2)+"\n")
            reports[side] = report
            manifest["checkpoints"][side]["heldout_data_sha256"] = report["heldout_data_sha256"]
            manifest["checkpoints"][side]["validation"] = dict(passed=report["passed"],
                metrics=metrics["reconstruction"], heldout_frames=len(held[side]),
                repeat_decode=metrics["repeat_decode"], reanchor_invariance=metrics["reanchor_invariance"])
            run.log({f"{side}/heldout_{k}":v for k,v in metrics["reconstruction"].items()}
                    | {f"{side}/passed":int(report["passed"])})
            save()
            print(json.dumps(dict(side=side, passed=report["passed"], reconstruction=metrics["reconstruction"],
                                   repeat_decode=metrics["repeat_decode"])), flush=True)
        passed = all(report["passed"] for report in reports.values())
        if passed:
            for side, report in reports.items():
                path = out / (side+"_pca15.validation.json")
                path.write_text(json.dumps(report, indent=2)+"\n")
                manifest["checkpoints"][side]["sidecar_sha256"] = sha256(path)
                FrozenFixedCameraHandCodec(out/(side+"_pca15.pt"))  # strict production loader
        manifest["status"] = "passed" if passed else "failed_quality_gate"
        run.summary["passed"] = passed
        save()
        run.finish(exit_code=0 if passed else 2)
        if not passed:
            raise SystemExit(2)
    except Exception as error:
        manifest["status"] = "error"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save()
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
