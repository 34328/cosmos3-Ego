"""Prepare a small, explicit full-segment GPU validation subset; never launch it.

The generated TOML and overrides enter the existing Cosmos official Trainer via
cosmos3_ar_it2v/launch.sh. No synthetic segments, crops or duplicated rows. An explicit retention subset
can exercise the approved oversize policy; source segment bounds stay intact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


REGISTRY = 'rbs_wam_ar_it2v_v0_2_ego100h_full_segments'
BANDS = {'short': (90, 105, 96), 'medium': (176, 200, 187), 'long': (1000, 1060, 1030)}


def segment_key(row):
    return (row['episode_hash'], row['span_index'], row['start_idx'], row['end_idx'])


def select_segments(rows, include_retention=False):
    selected, episodes = [], set()
    bands = dict(BANDS)
    if include_retention:
        bands['retention'] = (1600, 1700, 1670)
    for band, (low, high, target) in bands.items():
        candidates = [r for r in rows if r['split'] == 'train'
                      and low <= int(r['end_idx'])-int(r['start_idx']) <= high]
        # Fixed GT-only choice, independent of model output. Prefer distinct
        # episodes so the small validation does not consist of near-duplicates.
        candidates.sort(key=lambda r: (
            abs(int(r['end_idx'])-int(r['start_idx'])-target),
            hashlib.sha256(':'.join(segment_key(r)).encode()).hexdigest()))
        chosen = []
        for row in candidates:
            if row['episode_hash'] not in episodes:
                chosen.append((band, row))
                episodes.add(row['episode_hash'])
                if len(chosen) == 8:
                    break
        if len(chosen) != 8:
            raise ValueError(f'Need eight distinct-episode complete {band} segments')
        selected.extend(chosen)
    order = {segment_key(row): i for i, row in enumerate(rows)}
    selected.sort(key=lambda item: order[segment_key(item[1])])
    assert len({segment_key(row) for _, row in selected}) == 8 * len(bands)
    return selected


def prepare(episodes_manifest, segments_manifest, output, token_budget=65536, steps=6, rank_mixed=False, include_retention=False):
    if steps < 1:
        raise ValueError('Validation steps must be positive')
    episodes_manifest, segments_manifest, output = map(Path, (episodes_manifest, segments_manifest, output))
    with episodes_manifest.open(newline='') as f:
        episodes = {r['episode_hash']: r for r in csv.DictReader(f)}
    with segments_manifest.open(newline='') as f:
        reader = csv.DictReader(f)
        fieldnames, rows = reader.fieldnames, list(reader)
    selected = select_segments(rows, include_retention)
    if rank_mixed:
        import torch
        desired = [item for band in ('long', 'medium', 'short', 'retention') for item in selected if item[0] == band]
        permutation = torch.randperm(len(desired), generator=torch.Generator().manual_seed(42)).tolist()
        selected = [None] * len(desired)
        for index, item in zip(permutation, desired, strict=True):
            selected[index] = item
    samples = []
    for band, row in selected:
        ep = episodes[row['episode_hash']]
        start, end = int(row['start_idx']), int(row['end_idx'])
        assert ep['split'] == 'train' and 0 <= start < end <= int(ep['total_frames'])
        assert row['text_normalized'].strip() and float(ep['fps']) == 30.0
        frames = end-start
        latent = 1 + (frames-1+3)//4
        padded = 1+4*(latent-1)
        samples.append(dict(band=band, sample_id=':'.join(segment_key(row)),
            true_frames=frames, padded_frames=padded, temporal_padding=padded-frames,
            latent_frames=latent, partial_final_chunk=(latent-1)%4,
            vision_tokens=240*latent, start_idx=start, end_idx=end,
            text_normalized=row['text_normalized']))
    assert any(s['temporal_padding'] for s in samples)
    assert any(s['partial_final_chunk'] for s in samples)
    # Caption tokenizer has an existing 1024-token hard limit. This conservative
    # precheck makes all 24 rows safe; dataset additionally checks exact tokens.
    if any(s['vision_tokens']+1024+3 >= token_budget for s in samples if s['band'] != 'retention'):
        raise ValueError('Selected subset is unsafe under the requested token budget')
    output.mkdir(parents=True, exist_ok=False)
    subset = output/'segments.csv'
    with subset.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for _, row in selected)
    toml = output/'validation.toml'
    toml.write_text(f'''# Explicit {len(samples)}-segment GPU validation subset, not formal training.
[job]
task = "vfm"
experiment = "{REGISTRY}"
project = "rbs_wam_ar_it2v"
group = "ar_it2v_v0_2_validation"
name = "ar_it2v_v0_2_full_segment_packing_gate"
wandb_mode = "disabled"
[model]
precision = "bfloat16"
[model.parallelism]
data_parallel_shard_degree = 8
data_parallel_replicate_degree = 1
context_parallel_shard_degree = 1
[model.activation_checkpointing]
mode = "full"
[model.compile]
enabled = false
[optimizer]
lr = 2.0e-5
[trainer]
max_iter = {steps}
logging_iter = 1
[checkpoint]
save_iter = {steps}
''')
    prefix = 'dataloader_train.dataloader.datasets.video.dataset.'
    overrides = [prefix+'episodes_manifest='+str(episodes_manifest.resolve()),
        prefix+'segments_manifest='+str(subset.resolve()),
        prefix+'segment_statistics_path='+str((output/'summary.json').resolve()),
        'dataloader_train.max_sequence_length='+str(token_budget),
        'dataloader_train.max_samples_per_batch=null',
        'model.config.max_num_tokens_after_packing='+str(token_budget),
        prefix+'max_sequence_length='+str(token_budget),
        'dataloader_train.dataloader.num_workers=0',
        'dataloader_train.dataloader.persistent_workers=false',
        'dataloader_train.dataloader.prefetch_factor=null']
    if include_retention:
        overrides.append('++'+prefix+'long_segment_policy=uniform_retention')
    environment = dict(PATH='/home/lzh/miniconda3/envs/cosmos3/bin:'+os.environ.get('PATH',''),
        LD_LIBRARY_PATH='', TOML_FILE=str(toml.resolve()), EXTRA_TAIL_OVERRIDES=' '.join(overrides),
        WANDB_MODE='disabled', NNODES='1', NODE_RANK='0', NPROC_PER_NODE='8',
        MASTER_ADDR='127.0.0.1', MASTER_PORT='29907', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        OUTPUT_ROOT=str((output/'launcher').resolve()),
        IMAGINAIRE_OUTPUT_ROOT=str((output/'runs').resolve()))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dict(purpose='GPU validation only; no launch performed', world_size=8,
        samples_per_rank_per_epoch=len(samples)//8, token_budget_exclusive=token_budget, steps=steps,
        rank_mixed_first_epoch=rank_mixed, includes_retention_path=include_retention,
        selection='eight real train segments per length band; original bounds/captions unchanged',
        source_sha256={str(p): sha(p) for p in (episodes_manifest, segments_manifest)},
        subset_sha256=sha(subset), samples=samples, overrides=overrides,
        statistics_command=['python', '-m', 'cosmos3_ar_it2v.segment_statistics',
            '--episodes-manifest', str(episodes_manifest.resolve()),
            '--segments-manifest', str(subset.resolve()), '--tokenizer-path',
            '/mnt/checkpoints/Cosmos3-Nano/text_tokenizer', '--workers', '4', '--splits', 'train',
            '--budgets', str(token_budget), '--output', str((output/'summary.json').resolve())],
        launch_environment=environment, command=['bash', 'cosmos3_ar_it2v/launch.sh'])
    if include_retention:
        receipt['statistics_command'].extend(['--long-segment-policy', 'uniform_retention',
            '--policy-token-budget', str(token_budget)])
    (output/'selection.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-manifest', default='/mnt/lzh/cosmos-EgoWAM/training_manifests/mecka_100h_v1_episodes.csv')
    parser.add_argument('--segments-manifest', default='/mnt/lzh/cosmos-EgoWAM/training_manifests/mecka_100h_v1_segments.csv')
    parser.add_argument('--output', required=True)
    parser.add_argument('--token-budget', type=int, default=65536)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--rank-mixed', action='store_true')
    parser.add_argument('--include-retention', action='store_true', help='Eight original oversized segments; requires matching approved dataset/statistics policy')
    args = parser.parse_args()
    receipt = prepare(args.episodes_manifest, args.segments_manifest, args.output, args.token_budget, args.steps, args.rank_mixed, args.include_retention)
    print(json.dumps(dict(output=args.output, count=len(receipt['samples']),
        lengths=[s['true_frames'] for s in receipt['samples']],
        subset_sha256=receipt['subset_sha256'], launched=False)))


if __name__ == '__main__':
    main()
