"""Endpoint video diagnostics at 320x180, with whole-vector flow cosine.

Offline only. Raw GT shifts retain the existing dense-frame RGB MSE definition.
Historical metric reports are not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .ar_v02_noise_grid_metrics import GT_SHIFTS, pool_shift, source_shift_mse
from .ar_v02_video_diagnostics import FLOW_PARAMETERS, future_rgb

FLOW_SIZE = (320, 180)
SUM_KEYS = ("pred_magnitude_sum", "gt_magnitude_sum", "dot_sum",
            "pred_squared_sum", "gt_squared_sum", "pixels", "pairs")


def flow_sums(pred, gt):
    p, g = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape or p.ndim < 2 or p.shape[-1] != 2:
        raise ValueError("equal-shaped [...,2] flow required")
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("finite flow required")
    return dict(pred_magnitude_sum=float(np.linalg.norm(p, axis=-1).sum()),
                gt_magnitude_sum=float(np.linalg.norm(g, axis=-1).sum()),
                dot_sum=float((p*g).sum()),
                pred_squared_sum=float((p*p).sum()),
                gt_squared_sum=float((g*g).sum()),
                pixels=int(np.prod(p.shape[:-1])), pairs=1)


def pool_flow(rows):
    sums = {k: sum(r[k] for r in rows) for k in SUM_KEYS}
    denominator = np.sqrt(sums["pred_squared_sum"] * sums["gt_squared_sum"])
    return dict(**sums,
                pred_mean_magnitude=(sums["pred_magnitude_sum"]/sums["pixels"]
                                     if sums["pixels"] else None),
                gt_mean_magnitude=(sums["gt_magnitude_sum"]/sums["pixels"]
                                   if sums["pixels"] else None),
                magnitude_ratio=(sums["pred_magnitude_sum"]/sums["gt_magnitude_sum"]
                                 if sums["gt_magnitude_sum"] else None),
                direction_cosine=(float(np.clip(sums["dot_sum"]/denominator, -1, 1))
                                  if denominator else None))


def endpoint_flow(layout, meta, arrays):
    x, y, w, h = meta["valid_image_rect"]
    offsets, gt, rows = arrays["generated_offsets"], arrays["gt_rgb"], []
    if len(offsets) != len(layout.boundaries)+1:
        raise ValueError("chunk offsets mismatch")
    for i, boundary in enumerate(layout.boundaries):
        block = arrays["generated_rgb"][offsets[i]:offsets[i+1], y:y+h, x:x+w]
        if len(block) < 2:
            raise ValueError("condition and future frame required")
        rgb = (block[0], block[-1],
               gt[boundary.source_start, y:y+h, x:x+w],
               gt[boundary.source_stop, y:y+h, x:x+w])
        images = [cv2.cvtColor(cv2.resize(v, FLOW_SIZE, interpolation=cv2.INTER_AREA),
                               cv2.COLOR_RGB2GRAY) for v in rgb]
        pf = cv2.calcOpticalFlowFarneback(images[0], images[1], None, **FLOW_PARAMETERS)
        gf = cv2.calcOpticalFlowFarneback(images[2], images[3], None, **FLOW_PARAMETERS)
        rows.append(dict(chunk=boundary.chunk_id,
                         source_start=boundary.source_start,
                         source_stop=boundary.source_stop, **flow_sums(pf, gf)))
    return dict(all=pool_flow(rows), chunks=rows,
                chunk17plus=pool_flow([r for r in rows if r["chunk"] >= 17]))


def evaluate(path):
    from .ar_v02_eval import load_rollout
    cv2.setNumThreads(1)
    layout, meta, arrays = load_rollout(path)
    pred, _, sources, chunks = future_rgb(layout, meta, arrays)
    x, y, w, h = meta["valid_image_rect"]
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            sha.update(block)
    return dict(input=str(Path(path).resolve()), archive_sha256=sha.hexdigest(),
                sample_id=meta["sample_id"], source_offset=meta["source_offset"],
                checkpoint=meta["checkpoint"], history=meta["history"],
                seed=meta["seed"], steps=meta["steps"],
                history_video_sigma=meta.get("history_video_sigma", 0.),
                source_fps=meta["source_fps"],
                shift=source_shift_mse(pred, arrays["gt_rgb"][:, y:y+h, x:x+w],
                                       sources, chunks),
                flow=endpoint_flow(layout, meta, arrays))


def summarize(samples):
    identities = [(s["sample_id"], s["source_offset"]) for s in samples]
    groups = {(s["checkpoint"], s["history"], s["history_video_sigma"],
               s["seed"], s["steps"], s["source_fps"]) for s in samples}
    if not samples or len(set(identities)) != len(identities) or len(groups) != 1:
        raise ValueError("distinct windows in one checkpoint/history/sigma required")
    shifts = [r for s in samples for r in s["shift"]["chunks"]]
    flows = [r for s in samples for r in s["flow"]["chunks"]]
    def scope(ids):
        return dict(shift=pool_shift([r for r in shifts if r["chunk"] in ids]),
                    flow=pool_flow([r for r in flows if r["chunk"] in ids]))
    ids = sorted({r["chunk"] for r in flows})
    return dict(windows=len(samples), **scope(ids),
                chunks=[dict(chunk=k, **scope([k])) for k in ids],
                chunk17plus=scope([k for k in ids if k >= 17]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1:
        raise ValueError("positive worker count required")
    if args.workers == 1:
        samples = list(map(evaluate, args.input))
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.input)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            samples = list(pool.map(evaluate, args.input))
    report = dict(schema="ar_v02_endpoint_video_metrics_v1",
                  samples=samples, summary=summarize(samples),
                  flow_geometry=FLOW_SIZE, flow_parameters=FLOW_PARAMETERS,
                  flow_scope="chunk_condition_start_to_last_future",
                  magnitude_scope="ratio_of_all_pixel_mean_magnitudes",
                  cosine_scope="pooled_dot_over_product_of_global_l2_norms",
                  gt_shift_source_frames=GT_SHIFTS,
                  gt_shift_meaning="pred(t) versus GT(t+shift)",
                  rgb_shift_scope="valid_image_rect, full resolution, common inner support",
                  opencv_version=cv2.__version__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), windows=len(samples))))


if __name__ == "__main__":
    main()
