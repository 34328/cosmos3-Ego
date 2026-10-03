"""Exact full-segment packing budget audit; reads manifests/text, never RGB."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from cosmos3_ar_it2v.dataset import (
    EgoVerseIT2VDataset, FULL_SEGMENT_TOKEN_FORMULA, format_video_caption, full_segment_geometry, video_packing_tokens,
)

_tokenizer = None


def count_row(row):
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
    caption = format_video_caption(row['caption'], row['true_frames'], row['fps'])
    ids = tokenize_caption(caption, _tokenizer, is_video=True, use_system_prompt=False)
    if len(ids) > 1024:
        raise ValueError(f"Caption exceeds dataset token limit: {row['sample_id']}")
    padded, latent, padding = full_segment_geometry(row['true_frames'])
    return dict(row, text_tokens=len(ids), padded_frames=padded, latent_frames=latent,
                temporal_padding=padding, packing_tokens=video_packing_tokens(len(ids), padded))


def summarize(rows, budgets):
    duration = sum(r['duration_seconds'] for r in rows)
    result = {'segments': len(rows), 'hours': duration / 3600, 'excluded': 0}
    result['minimum_128_aligned_strict_budget'] = (max(r['packing_tokens'] for r in rows)//128 + 1)*128
    for key in ('true_frames', 'duration_seconds', 'text_tokens', 'packing_tokens'):
        result[key] = {str(q): float(np.quantile([r[key] for r in rows], q))
                       for q in (0, .25, .5, .75, .9, .95, .99, 1)}
    result['strict_budget_coverage'] = {}
    for budget in budgets:
        over = [r for r in rows if r['packing_tokens'] >= budget]
        seconds = sum(r['duration_seconds'] for r in over)
        result['strict_budget_coverage'][str(budget)] = dict(
            oversized=len(over), oversized_segment_fraction=len(over) / len(rows),
            oversized_hours=seconds / 3600, oversized_duration_fraction=seconds / duration,
            covered_segments=len(rows) - len(over))
    result['longest'] = sorted(rows, key=lambda r:r['packing_tokens'], reverse=True)[:5]
    ordered = sorted(rows, key=lambda r:r['true_frames'])
    result['smoke_candidates'] = [ordered[int(q * (len(ordered)-1))] for q in (.25, .5, .9)]
    return result


def main():
    global _tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-manifest', required=True)
    parser.add_argument('--segments-manifest', required=True)
    parser.add_argument('--tokenizer-path', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', required=True)
    parser.add_argument('--splits', nargs='+', choices=['train','test'], default=['train','test'])
    parser.add_argument('--budgets', nargs='+', type=int, default=[45056,49152,65536,98304,131072,179456,215424])
    args = parser.parse_args()
    start = time.monotonic()
    from cosmos_framework.model.generator.tokenizers.tokenization_qwen2 import Qwen2Tokenizer
    from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
    # Same native tokenizer class/files and added symbols as the training factory.
    _tokenizer, _ = add_special_tokens(Qwen2Tokenizer.from_pretrained(args.tokenizer_path))
    rows, manifest_summaries = [], {}
    for split in args.splits:
        ds = EgoVerseIT2VDataset(args.episodes_manifest, args.segments_manifest,
                                split=split, sample_mode='full_segment', cfg_dropout_rate=0)
        manifest_summaries[split] = {k:v for k,v in ds.manifest_summary.items() if k != 'clip_tier_counts'}
        for i, row in enumerate(ds.rows):
            fps = float(ds.episodes[row['episode_hash']]['fps'])
            frames = int(row['end_idx']) - int(row['start_idx'])
            rows.append(dict(split=split, row_index=i, sample_id=row['sample_id'],
                caption=row['text_normalized'], start_frame=int(row['start_idx']),
                end_frame_exclusive=int(row['end_idx']), true_frames=frames,
                fps=fps, duration_seconds=frames/fps))
    serial = [count_row(row) for row in rows[:16]]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        parallel = list(pool.map(count_row, rows[:16], chunksize=1))
        if serial != parallel:
            raise RuntimeError('Serial/parallel tokenizer results differ')
        counted = parallel + list(pool.map(count_row, rows[16:], chunksize=64))
    elapsed = time.monotonic() - start
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = output.with_suffix('.jsonl')
    with records.open('w') as f:
        for row in counted:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = dict(mode='full_segment', frame_stride=1, resolution=[368,640],
        token_formula=FULL_SEGMENT_TOKEN_FORMULA,
        budget_rule='packing_tokens < max_sequence_length', cfg='full captions; dropout only reduces cost',
        temporal_padding='repeat last RGB frame at most 3 times; real frame count retained',
        workers=args.workers, elapsed_seconds=elapsed, segments_per_second=len(counted)/elapsed,
        serial_parallel_equal=True, records=str(records.resolve()),
        records_sha256=hashlib.sha256(records.read_bytes()).hexdigest(), manifest_summaries=manifest_summaries,
        tokenizer_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(args.tokenizer_path).iterdir() if p.is_file()},
        splits={s:summarize([r for r in counted if r['split']==s], args.budgets) for s in args.splits})
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output':str(output), 'segments':len(counted), 'elapsed_seconds':elapsed,
                      'splits': {s:{k:v for k,v in summary['splits'][s].items()
                                    if k not in ('longest','smoke_candidates')} for s in args.splits}}))


if __name__ == '__main__':
    main()
