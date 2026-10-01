"""Post-process native rollout RGB; never feed diagnostics back into sampling.

Positive lag d means predicted frame at t best matches GT at t-d (prediction
lags GT). Flow is a diagnostic Farneback estimate, not measured scene motion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np

FLOW_PARAMETERS = dict(pyr_scale=.5, levels=3, winsize=15, iterations=3,
                       poly_n=5, poly_sigma=1.2, flags=0)
SUM_KEYS = ("gt_magnitude_sum", "pred_magnitude_sum", "moving_pixels",
            "direction_cosine_sum", "direction_pixels", "all_pixels",
            "all_gt_magnitude_sum", "all_pred_magnitude_sum", "frame_pairs")


def best_time_offset(pred, gt, *, sample_seconds, max_lag=16):
    """Equal support for every candidate; ties prefer zero/nearest-zero lag."""
    if pred.shape != gt.shape or pred.ndim != 4 or pred.shape[-1] != 3:
        raise ValueError("RGB sequences must have the same [T,H,W,3] shape")
    if max_lag < 0 or len(pred) <= 2*max_lag or not np.isfinite(sample_seconds) or sample_seconds <= 0:
        raise ValueError("invalid lag search or insufficient common support")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("nonfinite RGB")
    # Fixed area downsample keeps the offline search inexpensive and identical
    # across models. Valid-image cropping happens before this function.
    p = np.stack([cv2.resize(x, (160, 90), interpolation=cv2.INTER_AREA) for x in pred]).astype(np.float64)
    g = np.stack([cv2.resize(x, (160, 90), interpolation=cv2.INTER_AREA) for x in gt]).astype(np.float64)
    start, stop = max_lag, len(p)-max_lag
    curve = []
    for lag in range(-max_lag, max_lag+1):
        mse = float(np.square(p[start:stop]-g[start-lag:stop-lag]).mean())
        curve.append(dict(lag_samples=lag, lag_seconds=lag*sample_seconds, mse=mse))
    best = min(curve, key=lambda x: (x["mse"], abs(x["lag_samples"]), x["lag_samples"]))
    zero = curve[max_lag]["mse"]
    scores = np.asarray([x["mse"] for x in curve])
    return dict(**best, zero_lag_mse=zero,
                relative_mse_improvement=(zero-best["mse"])/zero if zero else 0.,
                max_lag_samples=max_lag, common_frames=stop-start,
                pixels_per_frame=160*90*3, spatial_size=[160,90],
                search_boundary_hit=bool(max_lag and abs(best["lag_samples"])==max_lag),
                flat_curve=bool(np.ptp(scores) <= 1e-12),
                tied_minima=int(np.count_nonzero(np.abs(scores-best["mse"]) <= 1e-12)),
                positive_lag_meaning="prediction_lags_gt", curve=curve)


def flow_sums(pred_flow, gt_flow, *, motion_threshold=.1, direction_epsilon=1e-4):
    """Additive statistics; stationary GT and zero predicted flow are explicit."""
    p, g = np.asarray(pred_flow, dtype=np.float64), np.asarray(gt_flow, dtype=np.float64)
    if p.shape != g.shape or p.shape[-1] != 2 or not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("finite equal-shaped flows [...,2] required")
    pn, gn = np.linalg.norm(p, axis=-1), np.linalg.norm(g, axis=-1)
    moving = gn > motion_threshold
    directional = moving & (pn > direction_epsilon)
    cosine = np.sum(p[directional]*g[directional], axis=-1)/(pn[directional]*gn[directional])
    return dict(gt_magnitude_sum=float(gn[moving].sum()),
                pred_magnitude_sum=float(pn[moving].sum()), moving_pixels=int(moving.sum()),
                direction_cosine_sum=float(np.clip(cosine,-1,1).sum()),
                direction_pixels=int(directional.sum()), all_pixels=int(gn.size),
                all_gt_magnitude_sum=float(gn.sum()), all_pred_magnitude_sum=float(pn.sum()),
                frame_pairs=1)


def summarize_flow(rows):
    sums = {key: sum(r[key] for r in rows) for key in SUM_KEYS}
    return dict(**sums,
                magnitude_ratio=(sums["pred_magnitude_sum"]/sums["gt_magnitude_sum"]
                                 if sums["gt_magnitude_sum"] else None),
                direction_cosine=(sums["direction_cosine_sum"]/sums["direction_pixels"]
                                  if sums["direction_pixels"] else None),
                direction_coverage=(sums["direction_pixels"]/sums["moving_pixels"]
                                    if sums["moving_pixels"] else None),
                gt_moving_fraction=(sums["moving_pixels"]/sums["all_pixels"] if sums["all_pixels"] else None))


def future_rgb(layout, meta, arrays):
    """Drop every U, and pair predicted future frames with dense raw GT timestamps."""
    x,y,w,h = meta["valid_image_rect"]
    if min(x,y)<0 or min(w,h)<=0:
        raise ValueError("invalid image crop")
    for key in ("gt_pixel_transform", "generated_pixel_transform"):
        if not np.array_equal(arrays[key], np.eye(3)):
            raise ValueError("diagnostics require matching identity image transforms")
    if y+h>arrays["gt_rgb"].shape[1] or x+w>arrays["gt_rgb"].shape[2]:
        raise ValueError("crop exceeds GT")
    pred, gt, sources, chunks = [], [], [], []
    offsets = arrays["generated_offsets"]
    for i,b in enumerate(layout.boundaries):
        rgb=arrays["generated_rgb"][offsets[i]:offsets[i+1]]
        index=np.arange(b.source_start+2,b.source_stop+1,2)
        block=rgb[1:,y:y+h,x:x+w]
        target=arrays["gt_rgb"][index,y:y+h,x:x+w]
        if block.shape != target.shape or block.shape[1:3] != (h,w):
            raise ValueError("generated/GT future frame alignment mismatch")
        pred.append(block);gt.append(target);sources.extend(index.tolist())
        chunks.extend([b.chunk_id]*len(index))
    sources=np.asarray(sources,dtype=np.int64)
    if len(sources)<2 or not np.all(np.diff(sources)==2):
        raise ValueError("expected dense stride-two future sampling")
    return np.concatenate(pred),np.concatenate(gt),sources,np.asarray(chunks)


def sequence_flow(pred, gt, chunk_ids):
    """No condition frame or cross-chunk edge enters the flow comparison."""
    if pred.shape!=gt.shape or len(pred)!=len(chunk_ids):
        raise ValueError("RGB/chunk sequence mismatch")
    rows=[]
    for i in range(1,len(pred)):
        if chunk_ids[i]!=chunk_ids[i-1]:
            continue
        images=[cv2.cvtColor(v,cv2.COLOR_RGB2GRAY) for v in (pred[i-1],pred[i],gt[i-1],gt[i])]
        pf=cv2.calcOpticalFlowFarneback(images[0],images[1],None,**FLOW_PARAMETERS)
        gf=cv2.calcOpticalFlowFarneback(images[2],images[3],None,**FLOW_PARAMETERS)
        rows.append(dict(chunk=int(chunk_ids[i]),**flow_sums(pf,gf)))
    return dict(all=summarize_flow(rows),
                chunk17plus=summarize_flow([r for r in rows if r["chunk"]>=17]),
                chunks=[dict(chunk=int(k),**summarize_flow([r for r in rows if r["chunk"]==k]))
                        for k in np.unique(chunk_ids)])


def evaluate(path):
    from .ar_v02_eval import load_rollout
    layout,meta,arrays=load_rollout(path)
    pred,gt,sources,chunks=future_rgb(layout,meta,arrays)
    interval=2/float(meta["source_fps"])
    return dict(input=str(Path(path).resolve()),sample_id=meta["sample_id"],
                history=meta["history"],checkpoint=meta["checkpoint"],seed=meta["seed"],
                source_fps=meta["source_fps"],sample_interval_seconds=interval,
                future_frames=len(pred),source_first=int(sources[0]+meta["source_offset"]),
                source_last=int(sources[-1]+meta["source_offset"]),
                lag=best_time_offset(pred,gt,sample_seconds=interval),
                flow=sequence_flow(pred,gt,chunks))


def summarize(samples):
    if not samples or len({(s['history'],s['checkpoint']) for s in samples})!=1:
        raise ValueError("aggregate one checkpoint and history mode at a time")
    if len({s['sample_id'] for s in samples})!=len(samples):
        raise ValueError("duplicate sample identities in aggregation")
    intervals={s["sample_interval_seconds"] for s in samples}
    if len(intervals)!=1:
        raise ValueError("cannot pool lag curves with different source sample intervals")
    curves=[s["lag"]["curve"] for s in samples]
    weights=[s["lag"]["common_frames"]*s["lag"]["pixels_per_frame"] for s in samples]
    curve=[]
    for j in range(len(curves[0])):
        assert all(c[j]["lag_samples"]==curves[0][j]["lag_samples"] for c in curves)
        curve.append(dict(lag_samples=curves[0][j]["lag_samples"],
                          lag_seconds=curves[0][j]["lag_seconds"],
                          mse=sum(c[j]["mse"]*w for c,w in zip(curves,weights))/sum(weights)))
    best=min(curve,key=lambda x:(x["mse"],abs(x["lag_samples"]),x["lag_samples"]))
    ids=sorted({r["chunk"] for s in samples for r in s["flow"]["chunks"]})
    return dict(windows=len(samples),pooled_best_lag=best,pooled_lag_curve=curve,
                median_window_lag_seconds=float(np.median([s["lag"]["lag_seconds"] for s in samples])),
                flow={scope:summarize_flow([s["flow"][scope] for s in samples])
                      for scope in ("all","chunk17plus")},
                flow_by_chunk=[dict(chunk=k,**summarize_flow([r for s in samples for r in s["flow"]["chunks"] if r["chunk"]==k])) for k in ids])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=1)
    args=p.parse_args()
    if args.output.exists():raise ValueError("refuse overwriting diagnostics")
    if args.workers<1:raise ValueError("workers must be positive")
    cv2.setNumThreads(1)
    if args.workers==1:
        samples=[evaluate(x) for x in args.input]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.workers,len(args.input)),initializer=cv2.setNumThreads,initargs=(1,)) as pool:
            samples=pool.map(evaluate,args.input)
    report=dict(schema="ar_v02_video_diagnostics_v1",flow_algorithm="OpenCV Farneback",
                opencv_version=cv2.__version__,flow_parameters=FLOW_PARAMETERS,
                flow_motion_threshold_pixels_per_sample=.1,direction_epsilon=1e-4,
                flow_scope="within_chunk_future_pairs_only",
                lag_metric="RGB MSE after 160x90 area resize; identical interior support for all lags",
                samples=samples,summary=summarize(samples))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.output),windows=len(samples))))


if __name__=="__main__":
    main()
