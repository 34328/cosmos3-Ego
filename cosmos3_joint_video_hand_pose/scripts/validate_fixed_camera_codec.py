"""Heldout wrist-local reconstruction and production reanchor invariance gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import collect, sha256
from cosmos3_joint_video_hand_pose.src.codec_fixed_camera import FrozenFixedCameraHandAE15, REPRESENTATION, INPUT_FRAME, validate_source_pair


def summary(error):
    error = error.detach().cpu().float().flatten()
    if not len(error) or not torch.isfinite(error).all():
        raise ValueError("non-finite/empty error metric")
    return {"mean_mm":float(error.mean()),"p95_mm":float(torch.quantile(error,.95)),"max_mm":float(error.max())}


@torch.no_grad()
def shape_measurements(pred, target):
    """Scale ratio and all-pairs shape error, including the origin wrist (metres)."""
    pred, target = pred.reshape(-1,20,3), target.reshape(-1,20,3)
    radius = lambda x: x.square().sum(-1).mean(-1).sqrt()
    ratio = radius(pred) / radius(target).clamp_min(1e-8)
    wrist = pred.new_zeros((len(pred),1,3))
    pred, target = torch.cat((wrist,pred),1), torch.cat((wrist,target),1)
    i,j = torch.triu_indices(21,21,offset=1,device=pred.device)
    pd = (pred[:,i]-pred[:,j]).norm(dim=-1)
    td = (target[:,i]-target[:,j]).norm(dim=-1)
    return torch.stack((ratio,(pd-td).abs().mean(-1)*1000),-1).cpu()


def shape_summary(values):
    x = torch.cat(values)
    if not torch.isfinite(x).all():
        raise ValueError('non-finite shape diagnostics')
    return {'wrist_radius_ratio_mean':float(x[:,0].mean()),
            'wrist_radius_ratio_p05':float(torch.quantile(x[:,0],.05)),
            'wrist_radius_ratio_p95':float(torch.quantile(x[:,0],.95)),
            'pairwise_distance_mae_mm':float(x[:,1].mean()),
            'scope':'diagnostic only; ratio 1 preserves hand scale, 0 is collapsed; 210 pairs including wrist'}


@torch.no_grad()
def reanchor_invariance(codec, q, camera_rotations):
    """Exercise the production integration/reanchor path: 100 C4 blocks + tail.

    Wrist and camera increments are nonzero. Empirical held-out camera axes
    initialize the test; this is a geometric stress test, not AE quality evidence.
    """
    from cosmos3_joint_video_hand_pose.src import action_fixed_camera as geometry
    device = codec.input_mean.device
    q = torch.as_tensor(q, device=device, dtype=torch.float32)
    z = codec.encode(q)
    if z.shape != (15,):
        raise ValueError("reanchor test requires a 15D wrist-local codec")
    local = codec.decode(z)
    class CountedCodec:
        coordinate_system = "fixed-camera"
        representation = REPRESENTATION
        input_frame = INPUT_FRAME
        calls = 0
        def encode(self, x):
            raise AssertionError("reanchor must never encode")
        def decode(self, x):
            self.calls += 1
            return codec.decode(x)
    counted = CountedCodec()
    pair = (counted, counted)
    rigid = torch.eye(4, device=device).repeat(3, 1, 1)
    # Nonzero wrist orientation, independent from camera axes.
    skew = q.new_tensor([[0., -.31, -.17], [.31, 0., -.23], [.17, .23, 0.]])
    rigid[1:, :3, :3] = torch.matrix_exp(skew)
    rigid[1:, :3, 3] = q.new_tensor([[.2, -.1, .5], [-.2, .1, .5]])
    state = geometry.FixedCameraState(0, rigid, z.repeat(2, 1))
    camera_rotations = torch.as_tensor(camera_rotations, device=device, dtype=torch.float32)
    if camera_rotations.ndim != 3 or camera_rotations.shape[-2:] != (3, 3) or len(camera_rotations) < 2:
        raise ValueError("need an empirical camera trajectory")
    # Calibrated finite rotations, including a nonzero camera increment even
    # when a held-out camera happens to remain stationary.
    delta = torch.eye(4, device=device).repeat(3, 1, 1)
    delta[0, :3, :3] = torch.matrix_exp(skew * .01)
    delta[1:, :3, :3] = torch.matrix_exp(skew * -.007)
    camera = camera_rotations[1] @ camera_rotations[0].T
    delta[0, :3, :3] = geometry._matrices(geometry._pose9(
        torch.block_diag(camera, q.new_ones(1))))[:3, :3] @ delta[0, :3, :3]
    rows = geometry._assemble(geometry._pose9(delta), z.new_zeros(2, 15))
    max_error, exact = 0., True
    for length in [32] * 100 + [8]:
        result = geometry.decode_future_physical(state, rows.repeat(length, 1), pair)
        exact = exact and torch.equal(result.end_state.hand_latents, state.hand_latents)
        inv = torch.linalg.inv(result.rigid_chunk[-1, 0])
        expected = torch.einsum("ij,hnj->hni", inv[:3, :3], result.keypoints_chunk[-1]) + inv[:3, 3]
        wrist = result.end_state.rigid_camera[1:]
        points = wrist[:, None, :3, 3] + torch.einsum("hij,nj->hni", wrist[:, :3, :3], local)
        reconstructed = torch.cat((wrist[:, None, :3, 3], points), dim=1)
        max_error = max(max_error, float((expected - reconstructed).abs().max()))
        state = result.end_state
    no_reencode = counted.calls == 202  # two hands, one decode per block
    return dict(passed=exact and no_reencode and max_error <= 1e-5,
                latent_exact=exact, full_chunks=100, tail_frames=8,
                max_point_error_m=max_error, no_reencode=no_reencode,
                nonzero_wrist_rotation=True, nonzero_camera_rotation=True,
                protocol="production decode_future_physical/reanchor_state; 3200 frames plus 8-frame tail")


@torch.no_grad()
def evaluate(codec, q, trajectories):
    device = codec.input_mean.device
    errors, reconstruction_shapes = [], []
    for part in q.split(8192):
        part = part.to(device)
        reconstructed = codec.decode(codec.encode(part))
        errors.append(torch.linalg.vector_norm(reconstructed-part, dim=-1).cpu()*1000)
        reconstruction_shapes.append(shape_measurements(reconstructed, part))
    if not trajectories:
        raise ValueError("heldout camera trajectories required")
    invariance = reanchor_invariance(codec, *trajectories[0])
    return {"reconstruction": summary(torch.cat(errors)),
            "reanchor_invariance": invariance,
            "trajectory_count": len(trajectories),
            "shape_diagnostics": {"reconstruction": shape_summary(reconstruction_shapes)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes",required=True)
    p.add_argument("--valid-windows",required=True)
    p.add_argument("--codec-dir",required=True)
    p.add_argument("--report-dir",help="Independent diagnostic output; never replace existing sidecars")
    p.add_argument("--max-episodes",type=int,default=32)
    p.add_argument("--windows-per-episode",type=int,default=2)
    p.add_argument("--seed",type=int,default=73)
    p.add_argument("--device",default="cpu")
    # Explicit diagnostic gates, not a claim these suffice for deployment.
    p.add_argument("--max-mean-mm",type=float,default=5.)
    p.add_argument("--max-p95-mm",type=float,default=15.)
    a = p.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in (a.max_mean_mm,a.max_p95_mm)) or a.max_mean_mm > 5 or a.max_p95_mm > 15:
        p.error("thresholds must be positive and no looser than mean 5mm / p95 15mm")
    data, trajectories, provenance = collect(a.episodes,a.valid_windows,"heldout",a.max_episodes,a.windows_per_episode,a.seed)
    reports = {}
    report_dir = Path(a.report_dir) if a.report_dir else Path(a.codec_dir)
    report_dir.mkdir(parents=True,exist_ok=True)
    if any((report_dir/(s+'_mlp15.validation.json')).exists() for s in ('right','left')):
        raise FileExistsError('validation reports already exist; choose a new --report-dir')
    for side in ("right","left"):
        path = Path(a.codec_dir)/(side+"_mlp15.pt")
        codec = FrozenFixedCameraHandAE15(path,allow_unvalidated=True).to(a.device)
        if codec.metadata["side"] != side:
            raise ValueError("left/right checkpoint mismatch")
        if set(provenance["episode_ids"]) & set(codec.metadata["fit"]["episode_ids"]):
            raise ValueError("heldout/train episode overlap")
        validate_source_pair(codec.metadata["fit"], provenance)
        metrics = evaluate(codec,data[side],trajectories[side])
        gates = {"reconstruction_mean_mm":a.max_mean_mm,"reconstruction_p95_mm":a.max_p95_mm}
        passed = (metrics["reconstruction"]["mean_mm"] <= a.max_mean_mm and
                  metrics["reconstruction"]["p95_mm"] <= a.max_p95_mm and
                  metrics["reanchor_invariance"]["passed"])
        report = dict(schema_version=2,input_frame=INPUT_FRAME,representation=REPRESENTATION,checkpoint_sha256=sha256(path),
            heldout_episode_ids=provenance["episode_ids"],heldout_provenance=provenance,
            heldout_sample_count=len(data[side]),metrics=metrics,thresholds=gates,passed=passed,
            gate_scope="explicit diagnostic gates; downstream WAM validation still required")
        (report_dir/(side+'_mlp15.validation.json')).write_text(json.dumps(report,indent=2)+"\n")
        reports[side]=report
    print(json.dumps(reports,indent=2),flush=True)
    if not all(r["passed"] for r in reports.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
