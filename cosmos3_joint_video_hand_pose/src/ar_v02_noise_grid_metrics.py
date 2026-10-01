"""Offline metrics for the frozen history-noise grid; no sampling changes.

GT shift is in dense source-frame units: compare prediction at source t with
GT at t+shift. Positive shift means prediction matches later GT. First-to-last
flow includes each chunk's condition image, exactly as specified for this grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .ar_v02_video_diagnostics import (
    FLOW_PARAMETERS, flow_sums, future_rgb, summarize_flow,
)

GT_SHIFTS = (-4, -2, 0, 2, 4)


def source_shift_mse(pred, gt_dense, sources, chunk_ids):
    """Full-resolution RGB MSE with identical support for five GT shifts."""
    pred, gt_dense = np.asarray(pred), np.asarray(gt_dense)
    sources, chunk_ids = np.asarray(sources), np.asarray(chunk_ids)
    if pred.ndim != 4 or pred.shape[-1] != 3 or pred.shape[1:] != gt_dense.shape[1:]:
        raise ValueError("matching RGB geometry required")
    if len(pred) != len(sources) or len(pred) != len(chunk_ids):
        raise ValueError("source/chunk alignment mismatch")
    if not np.isfinite(pred).all() or not np.isfinite(gt_dense).all():
        raise ValueError("nonfinite RGB")
    valid = (sources >= 4) & (sources + 4 < len(gt_dense))
    if not valid.any():
        raise ValueError("no common support for GT shifts")
    selected = np.flatnonzero(valid)
    rows = []
    for chunk in np.unique(chunk_ids[selected]):
        ix = selected[chunk_ids[selected] == chunk]
        curve = []
        for shift in GT_SHIFTS:
            total = sum(float(np.square(
                pred[i].astype(np.float64) -
                gt_dense[sources[i] + shift].astype(np.float64)
            ).sum()) for i in ix)
            curve.append(dict(gt_shift_frames=shift, squared_error_sum=total,
                              values=int(len(ix) * np.prod(pred.shape[1:]))))
        rows.append(dict(chunk=int(chunk), common_frames=len(ix), curve=curve))
    return dict(all=pool_shift(rows), chunks=rows)


def pool_shift(rows):
    curve = []
    for shift in GT_SHIFTS:
        terms = [next(x for x in row["curve"] if x["gt_shift_frames"] == shift)
                 for row in rows]
        total = sum(x["squared_error_sum"] for x in terms)
        count = sum(x["values"] for x in terms)
        curve.append(dict(gt_shift_frames=shift, squared_error_sum=total,
                          values=count, mse=total/count if count else None))
    finite = [x for x in curve if x["mse"] is not None]
    if not finite:
        return dict(best_gt_shift_frames=None, curve=curve)
    best = min(finite, key=lambda x: (x["mse"], abs(x["gt_shift_frames"]),
                                     x["gt_shift_frames"]))
    zero = next(x["mse"] for x in curve if x["gt_shift_frames"] == 0)
    scores = np.asarray([x["mse"] for x in finite])
    return dict(best_gt_shift_frames=best["gt_shift_frames"], best_mse=best["mse"],
                zero_mse=zero, relative_improvement=(zero-best["mse"])/zero if zero else 0.,
                flat_curve=bool(np.ptp(scores) <= 1e-12),
                boundary_hit=abs(best["gt_shift_frames"]) == 4,
                curve=curve)


def endpoint_flow(layout, meta, arrays):
    x, y, w, h = meta["valid_image_rect"]
    offsets, rows = arrays["generated_offsets"], []
    for i, boundary in enumerate(layout.boundaries):
        block = arrays["generated_rgb"][offsets[i]:offsets[i+1], y:y+h, x:x+w]
        gt = arrays["gt_rgb"]
        images = [cv2.cvtColor(v, cv2.COLOR_RGB2GRAY) for v in (
            block[0], block[-1],
            gt[boundary.source_start, y:y+h, x:x+w],
            gt[boundary.source_stop, y:y+h, x:x+w],
        )]
        pf = cv2.calcOpticalFlowFarneback(images[0], images[1], None, **FLOW_PARAMETERS)
        gf = cv2.calcOpticalFlowFarneback(images[2], images[3], None, **FLOW_PARAMETERS)
        rows.append(dict(chunk=boundary.chunk_id, **flow_sums(pf, gf)))
    return dict(all=summarize_flow(rows), chunks=rows,
                chunk17plus=summarize_flow([x for x in rows if x["chunk"] >= 17]))


def evaluate(path):
    from .ar_v02_eval import load_rollout
    cv2.setNumThreads(1)
    layout, meta, arrays = load_rollout(path)
    pred, _, sources, chunks = future_rgb(layout, meta, arrays)
    x, y, w, h = meta["valid_image_rect"]
    gt = arrays["gt_rgb"][:, y:y+h, x:x+w]
    return dict(input=str(Path(path).resolve()), sample_id=meta["sample_id"],
                source_offset=meta["source_offset"], checkpoint=meta["checkpoint"],
                history=meta["history"], seed=meta["seed"],
                history_video_sigma=meta.get("history_video_sigma", 0.),
                source_fps=meta["source_fps"],
                shift=source_shift_mse(pred, gt, sources, chunks),
                flow=endpoint_flow(layout, meta, arrays))


def summarize(samples):
    identities = [(s["sample_id"], s["source_offset"]) for s in samples]
    if not samples or len(set(identities)) != len(identities):
        raise ValueError("nonempty distinct windows required")
    groups = {(s["checkpoint"], s["history"], s["history_video_sigma"]) for s in samples}
    if len(groups) != 1:
        raise ValueError("one checkpoint/history/sigma per aggregate required")
    shifts = [r for s in samples for r in s["shift"]["chunks"]]
    flows = [r for s in samples for r in s["flow"]["chunks"]]
    return dict(shift=pool_shift(shifts), flow=summarize_flow(flows),
                chunks=[dict(chunk=k,
                             shift=pool_shift([r for r in shifts if r["chunk"] == k]),
                             flow=summarize_flow([r for r in flows if r["chunk"] == k]))
                        for k in sorted({r["chunk"] for r in flows})],
                chunk17plus=dict(shift=pool_shift([r for r in shifts if r["chunk"] >= 17]),
                                 flow=summarize_flow([r for r in flows if r["chunk"] >= 17])))


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
    report = dict(schema="ar_v02_noise_grid_metrics_v1", samples=samples,
                  summary=summarize(samples), gt_shift_source_frames=GT_SHIFTS,
                  gt_shift_meaning="compare_pred_at_t_to_GT_at_t_plus_shift",
                  flow_scope="chunk_condition_start_to_last_future_frame",
                  flow_parameters=FLOW_PARAMETERS, opencv_version=cv2.__version__,
                  valid_crop="archive valid_image_rect; full resolution",
                  moving_threshold_pixels=.1, direction_epsilon=1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), windows=len(samples))))


if __name__ == "__main__":
    main()
