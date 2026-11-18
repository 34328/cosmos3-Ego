"""Measure completed full-segment MP4s on CPU; never load a model or change media."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path

# The offline workers use one CPU thread each and cannot initialize CUDA.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from cosmos_framework.inference.metrics import compute_psnr
from .build_page import sha256_file

cv2.setNumThreads(1)
torch.set_num_threads(1)

PARTS = ('full', 'early', 'middle', 'late')
FLOW_SCOPES = ('total', 'within_chunk', 'boundary', 'condition_transition')
METHOD = {
    'version': 1,
    'pixels': 'uint8 RGB decoded from existing compressed MP4s; compression error is included',
    'time': 'all original frames at manifest fps; no temporal resampling; VAE padding excluded',
    'psnr': 'sum RGB squared errors and RGB value counts, then 20*log10(255/sqrt(MSE)); first conditioned RGB excluded',
    'frame_psnr': 'cosmos_framework.inference.metrics.compute_psnr',
    'flow': 'OpenCV CPU Farneback on grayscale resized to exactly 320x180; all pixels, including camera/background motion',
    'flow_size_wh': [320, 180],
    'flow_parameters': {'pyr_scale': 0.5, 'levels': 3, 'winsize': 15, 'iterations': 3,
                        'poly_n': 5, 'poly_sigma': 1.2, 'flags': 0},
    'flow_units': '320x180 image pixels per original frame',
    'direction_cosine': 'sum dot products / sqrt(sum GT squared norms * sum prediction squared norms)',
    'motion_ratio': 'RMS magnitude ratio = sqrt(sum prediction squared norms / sum GT squared norms)',
    'total': 'all adjacent frame pairs, including the initial condition transition and actual montage splices',
    'within_chunk': 'adjacent pairs inside one prediction block; excludes the first condition transition and splices',
    'boundary': 'last predicted RGB of one block to first predicted RGB of the next block; excludes 0->1',
    'condition_transition': 'the first RGB condition to the first predicted RGB, 0->1',
    'temporal_parts': 'three equal portions of true predicted RGB frames; a flow pair belongs to its destination frame',
    'boundary_jump_ratio': 'RMS adjacent RGB difference at prediction-block splices, prediction / GT; not optical flow',
    'zero_motion': 'zero norm gives null direction, zero GT norm gives null ratio; pairs remain counted',
}


def empty_flow_totals():
    return dict(pairs=0, pixel_vectors=0, dot=0.0, gt_squared_norm=0.0,
                pred_squared_norm=0.0, endpoint_error_sum=0.0,
                zero_gt_norm_pairs=0, zero_pred_norm_pairs=0)


def add_flow_totals(total, other):
    for name in total:
        total[name] += other[name]


def flow_pair_totals(gt_flow, pred_flow):
    """Reuse the accepted diagnostic's pooled dot/norm primitive, without filtering."""
    gt64 = gt_flow.astype(np.float64)
    pred64 = pred_flow.astype(np.float64)
    gt_norm = float(np.sum(gt64 ** 2))
    pred_norm = float(np.sum(pred64 ** 2))
    return dict(pairs=1, pixel_vectors=int(gt64.shape[0] * gt64.shape[1]),
                dot=float(np.sum(gt64 * pred64)), gt_squared_norm=gt_norm,
                pred_squared_norm=pred_norm,
                endpoint_error_sum=float(np.linalg.norm(gt64 - pred64, axis=-1).sum()),
                zero_gt_norm_pairs=int(gt_norm == 0), zero_pred_norm_pairs=int(pred_norm == 0))


def flow_summary(total):
    count = total['pixel_vectors']
    gt_norm, pred_norm = total['gt_squared_norm'], total['pred_squared_norm']
    return dict(pairs=total['pairs'], pixel_vectors=count,
                direction_cosine=(float(np.clip(total['dot'] / math.sqrt(gt_norm * pred_norm), -1, 1))
                                  if gt_norm > 0 and pred_norm > 0 else None),
                motion_ratio=math.sqrt(pred_norm / gt_norm) if gt_norm > 0 else None,
                gt_rms_px_per_frame=math.sqrt(gt_norm / count) if count else None,
                pred_rms_px_per_frame=math.sqrt(pred_norm / count) if count else None,
                endpoint_error_px_per_frame=total['endpoint_error_sum'] / count if count else None,
                zero_gt_norm_pairs=total['zero_gt_norm_pairs'],
                zero_pred_norm_pairs=total['zero_pred_norm_pairs'])


def optical_flow(first, second):
    return cv2.calcOpticalFlowFarneback(first, second, None, **METHOD['flow_parameters'])


def gray_frame(rgb):
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (320, 180), interpolation=cv2.INTER_AREA)


def empty_totals():
    return dict(rgb_sse=0, rgb_values=0, predicted_frames=0,
                boundary_gt_rgb_sse=0, boundary_pred_rgb_sse=0,
                **{key: empty_flow_totals() for key in FLOW_SCOPES})


def add_totals(total, other):
    for name in ('rgb_sse', 'rgb_values', 'predicted_frames', 'boundary_gt_rgb_sse', 'boundary_pred_rgb_sse'):
        total[name] += other[name]
    for name in FLOW_SCOPES:
        add_flow_totals(total[name], other[name])


def summarize(total):
    mse = total['rgb_sse'] / total['rgb_values'] if total['rgb_values'] else None
    return dict(predicted_frames=total['predicted_frames'], mse_uint8=mse,
                psnr_db=20 * math.log10(255 / math.sqrt(mse)) if mse else None,
                psnr_identical=mse == 0,
                boundary_jump_ratio=(math.sqrt(total['boundary_pred_rgb_sse'] / total['boundary_gt_rgb_sse'])
                                     if total['boundary_gt_rgb_sse'] else None),
                **{key: flow_summary(total[key]) for key in FLOW_SCOPES})


def rgb_sse(first, second):
    difference = first.astype(np.int16) - second.astype(np.int16)
    return int(np.square(difference, dtype=np.int32).sum(dtype=np.int64))


def measure_case(job):
    """Decode a pair once and retain only the preceding frame in each worker."""
    manifest = job['manifest']
    count = int(manifest['frames'])
    fps = float(manifest['fps'])
    block_rgb_frames = 4 * int(manifest['frames_per_chunk'])
    totals = {part: empty_totals() for part in PARTS}
    frame_psnr = []
    readers = []
    try:
        for name in ('gt', 'prediction'):
            reader = imageio.get_reader(job['source'][name]['path'],
                                       input_params=['-threads', '1'], output_params=['-threads', '1'])
            readers.append(reader)
            actual_fps = float(reader.get_meta_data()['fps'])
            if not math.isclose(actual_fps, fps, rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError(f"{job['id']}: {name} MP4 fps {actual_fps} differs from manifest {fps}")
        previous = previous_gray = None
        frames_read = 0
        missing = object()
        for frame_index, (gt, pred) in enumerate(zip_longest(*readers, fillvalue=missing)):
            if gt is missing or pred is missing or frame_index >= count:
                raise ValueError(f"{job['id']}: decoded video lengths differ from the true manifest length")
            if (gt.shape != pred.shape or gt.ndim != 3 or gt.shape[-1] != 3
                    or gt.dtype != np.uint8 or pred.dtype != np.uint8):
                raise ValueError(f"{job['id']}: expected matching uint8 RGB frames")
            if previous is not None and gt.shape != previous[0].shape:
                raise ValueError(f"{job['id']}: decoded spatial geometry changed")
            gray = (gray_frame(gt), gray_frame(pred))
            if frame_index:
                part = PARTS[1 + min(2, (frame_index - 1) * 3 // (count - 1))]
                scopes = (totals['full'], totals[part])
                error = rgb_sse(gt, pred)
                psnr = compute_psnr(gt.transpose(2, 0, 1)[:, None], pred.transpose(2, 0, 1)[:, None])
                frame_psnr.append(None if math.isinf(psnr) else psnr)
                flows = flow_pair_totals(optical_flow(previous_gray[0], gray[0]),
                                        optical_flow(previous_gray[1], gray[1]))
                category = ('condition_transition' if frame_index == 1 else
                            'boundary' if (frame_index - 1) % block_rgb_frames == 0 else 'within_chunk')
                for scope in scopes:
                    scope['rgb_sse'] += error
                    scope['rgb_values'] += gt.size
                    scope['predicted_frames'] += 1
                    add_flow_totals(scope['total'], flows)
                    add_flow_totals(scope[category], flows)
                    if category == 'boundary':
                        scope['boundary_gt_rgb_sse'] += rgb_sse(previous[0], gt)
                        scope['boundary_pred_rgb_sse'] += rgb_sse(previous[1], pred)
            previous, previous_gray = (gt, pred), gray
            frames_read += 1
        if frames_read != count:
            raise ValueError(f"{job['id']}: decoded {frames_read} frames, expected {count}")
    finally:
        for reader in readers:
            reader.close()
    return {key: job[key] for key in ('id', 'split', 'history_mode', 'decoder_mode', 'title', 'caption', 'source', 'fingerprint')} | dict(
        frames=count, fps=fps, duration_seconds=count / fps,
        stats={part: summarize(totals[part]) for part in PARTS},
        pooled_totals=totals, frame_psnr_db=frame_psnr)


def load_jobs(selection_path):
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('preview_mode') != 'full_segment':
        raise ValueError('Metrics require a full_segment selection')
    root = selection_path.parent
    jobs, identities, pairs = [], set(), {}
    for window in selection['windows']:
        directory = (root / window['output_dir']).resolve()
        directory.relative_to(root)
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        history = window.get('history_mode', selection.get('history_mode', 'generated'))
        decoder = manifest.get('decoder_mode', 'predicted_prefix')
        if history not in ('gt', 'generated') or manifest.get('history_mode', manifest.get('history')) != history:
            raise ValueError(f"{window['id']}: history mode mismatch")
        if decoder != ('gt_prefix_montage' if history == 'gt' else 'predicted_prefix'):
            raise ValueError(f"{window['id']}: metrics require the actual current decoder mode")
        if window.get('decoder_mode', decoder) != decoder:
            raise ValueError(f"{window['id']}: selection decoder mode mismatch")
        identity = (window['id'], history)
        if identity in identities:
            raise ValueError(f'Duplicate metrics case: {identity}')
        identities.add(identity)
        if window['split'] not in ('train', 'test'):
            raise ValueError('Metrics split must be train or test')
        count = int(manifest['frames'])
        fps = float(manifest['fps'])
        indices = np.asarray(manifest['source_frame_indices'], dtype=np.int64)
        if count < 2 or indices.shape != (count,) or not np.all(np.diff(indices) == 1):
            raise ValueError(f"{window['id']}: metrics require every original continuous source frame")
        if (count != int(window['num_frames']) or manifest['sample_id'] != window['sample_id']
                or not math.isfinite(fps) or fps <= 0 or manifest.get('frame_stride') != 1):
            raise ValueError(f"{window['id']}: selection identity, true length or original frame timing differs")
        if int(manifest['latent_frames']) != 1 + (count - 1 + 3) // 4 or int(manifest['frames_per_chunk']) < 1:
            raise ValueError(f"{window['id']}: causal 4x VAE geometry differs")
        for key in ('checkpoint_step', 'seed', 'denoise_steps', 'guidance', 'context_sigma', 'fps'):
            if manifest[key] != selection[key]:
                raise ValueError(f"{window['id']}: {key} differs from selection")
        source = {}
        for name, path in (('manifest', manifest_path), ('gt', directory / 'gt.mp4'),
                           ('prediction', directory / 'generated.mp4')):
            source[name] = dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)
        fingerprint = hashlib.sha256(json.dumps(dict(window=window, source=source, method=METHOD),
                                               sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        jobs.append(dict(id=window['id'], split=window['split'], history_mode=history, decoder_mode=decoder,
                         title=window.get('title', window['id']), caption=window['caption'],
                         manifest=manifest, source=source, fingerprint=fingerprint))
        pair_key = (window['split'], manifest['sample_id'])
        signature = (count, fps, window['caption'], tuple(indices),
                     manifest['checkpoint'], manifest['seed'], manifest['denoise_steps'],
                     manifest['guidance'], manifest['context_sigma'])
        pair = pairs.setdefault(pair_key, {})
        if history in pair:
            raise ValueError(f'Duplicate segment/history: {pair_key}, {history}')
        pair[history] = signature
    if not jobs:
        raise ValueError('selection.windows must be nonempty')
    for key, pair in pairs.items():
        if set(pair) != {'gt', 'generated'} or pair['gt'] != pair['generated']:
            raise ValueError(f'{key}: modes must pair the same complete segment and settings')
    return selection, jobs


def aggregate_groups(samples):
    groups = []
    for split in ('train', 'test'):
        for history in ('gt', 'generated'):
            selected = [sample for sample in samples if sample['split'] == split and sample['history_mode'] == history]
            if not selected:
                continue
            totals = {part: empty_totals() for part in PARTS}
            for sample in selected:
                for part in PARTS:
                    add_totals(totals[part], sample['pooled_totals'][part])
            groups.append(dict(split=split, history_mode=history, samples_count=len(selected),
                               frames=sum(sample['frames'] for sample in selected),
                               duration_seconds=sum(sample['duration_seconds'] for sample in selected),
                               stats={part: summarize(totals[part]) for part in PARTS}, pooled_totals=totals))
    return groups


def build_report(selection_path, output, workers=4):
    selection_path, output = Path(selection_path).resolve(), Path(output).resolve()
    selection, jobs = load_jobs(selection_path)
    previous = json.loads(output.read_text(encoding='utf-8')) if output.is_file() else {}
    cache = {(sample['id'], sample['history_mode']): sample for sample in previous.get('samples', [])}
    measured, pending = {}, []
    for job in jobs:
        key = (job['id'], job['history_mode'])
        cached = cache.get(key)
        if cached and cached.get('fingerprint') == job['fingerprint']:
            measured[key] = cached
        else:
            pending.append(job)
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {pool.submit(measure_case, job): job for job in pending}
            for future in as_completed(futures):
                sample = future.result()
                measured[(sample['id'], sample['history_mode'])] = sample
                print(json.dumps(dict(event='case_measured', id=sample['id'], history_mode=sample['history_mode'],
                                      summary=sample['stats']['full']), ensure_ascii=False), flush=True)
    samples = [measured[(job['id'], job['history_mode'])] for job in jobs]
    report = dict(method=METHOD, version=selection['version'], checkpoint_step=selection['checkpoint_step'],
                  selection_path=str(selection_path), selection_sha256=sha256_file(selection_path),
                  settings={key: selection[key] for key in
                            ('checkpoint_step', 'seed', 'denoise_steps', 'guidance', 'context_sigma', 'fps')},
                  samples=samples, groups=aggregate_groups(samples))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    report = build_report(args.selection, args.output, args.workers)
    print(json.dumps(dict(event='metrics_complete', output=str(args.output.resolve()),
                          samples=len(report['samples']), groups=len(report['groups'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
