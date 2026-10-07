"""Render fixed pure-video windows with the existing AR sampler; no training changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def decode_preview(model, batch, predicted_latents, *, history_mode, true_frames,
                   frames_per_chunk=4):
    """Select the real decoder contract without changing the AR sampler.

    Generated history never requests GT encoding. GT history uses one extra
    deterministic official full-segment encode because the sampler does not
    return its reference; no RGB output is encoded and no A/C variants run.
    """
    import torch
    if history_mode not in ('generated', 'gt'):
        raise ValueError('history_mode must be generated or gt')
    with torch.no_grad():
        if history_mode == 'generated':
            decoded = model.decode(predicted_latents.to(**model.tensor_kwargs))
            return decoded[:, :, :true_frames].detach().cpu(), 'predicted_prefix'
        from .decoder_diagnostic import decode_gt_prefix
        clean = model.get_data_and_condition(batch, vision_condition_indexes=None)
        if clean.batch_size != 1 or clean.x0_tokens_action is not None or len(clean.x0_tokens_vision) != 1:
            raise ValueError('expected one continuously encoded pure-video GT segment')
        reference = clean.x0_tokens_vision[0].to(**model.tensor_kwargs)
        decoded = decode_gt_prefix(model, predicted_latents, reference,
            true_frames=true_frames, frames_per_chunk=frames_per_chunk)
        return decoded, 'gt_prefix_montage'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--toml', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos3_ar_it2v.inference import generate_latents, load_model, training_layout_batch

    selection = json.loads(args.selection.read_text())
    root = args.selection.resolve().parent
    windows = {w['id']: w for w in selection['windows']}
    if len(set(args.ids)) != len(args.ids) or any(i not in windows for i in args.ids):
        raise ValueError('IDs must be unique existing selection windows')
    selected = [windows[i] for i in args.ids]
    history_modes = {w['id']: w.get('history_mode', selection.get('history_mode', 'generated'))
                     for w in selected}
    if any(mode not in ('generated', 'gt') for mode in history_modes.values()):
        raise ValueError('history_mode must be generated or gt')
    for window in selected:
        expected_decoder = 'gt_prefix_montage' if history_modes[window['id']] == 'gt' else 'predicted_prefix'
        if window.get('decoder_mode', selection.get('decoder_mode', expected_decoder)) != expected_decoder:
            raise ValueError('decoder_mode disagrees with the new preview history mode')
    full_segment = selection.get('preview_mode') == 'full_segment'
    worker_name = '_'.join(args.ids)
    worker_dir = root / 'workers' / worker_name
    worker_dir.mkdir(parents=True, exist_ok=False)
    claim = dict(pid=os.getpid(), hostname=socket.gethostname(),
                 gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), started_unix=time.time(), ids=args.ids)
    write_json(worker_dir / 'started.json', claim)
    error = None
    initialized = False
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('An allocated idle GPU is required')
        distributed.init()
        initialized = True
        model, config = load_model(args.toml, args.checkpoint)
        datasets = {}
        ds_cfg = next(iter(config.dataloader_train.dataloader.datasets.values())).dataset
        tiers = (17, 33, 49, 65, 81, 97) if full_segment else tuple(sorted({17, 33, 49, 65, 81, 97,
                              *(int(w['num_frames']) for w in selected)}, reverse=True))
        # Retain minimum tier17 so dataset row numbering matches the training dataset.
        for split in {w['split'] for w in selected}:
            # Full previews retain the entire caption segment; legacy windows
            # remain available only for reproducing earlier galleries.
            datasets[split] = instantiate(ds_cfg, split=split, iterable_shuffle=False,
                random_window=False, cfg_dropout_rate=0., clip_frame_tiers=tiers,
                sample_mode='full_segment' if full_segment else 'window', long_segment_policy='error',
                max_sequence_length=None, segment_statistics_path=None)
        for window in selected:
            history_mode = history_modes[window['id']]
            output = root / window['output_dir']
            output.mkdir(parents=False, exist_ok=False)
            write_json(output / 'started.json', {**claim, 'window': window})
            job_start = time.monotonic()
            print(json.dumps(dict(event='window_started', id=window['id']), ensure_ascii=False), flush=True)
            dataset = datasets[window['split']]
            if full_segment:
                matches = [i for i, row in enumerate(dataset.rows) if row['sample_id'] == window['sample_id']]
                if len(matches) != 1:
                    raise ValueError('Complete segment identity must match exactly once')
                row = dataset.rows[matches[0]]
                if (int(row['start_idx']) != int(window['segment_start_frame']) or
                    int(row['end_idx']) != int(window['segment_end_frame_exclusive']) or
                    row['text_normalized'].strip() != window['caption'].strip()):
                    raise ValueError('Complete segment bounds or caption changed')
                sample = dataset[matches[0]]
            else:
                sample = dataset.get_item_at_window(
                    int(window['row_index']), window_start=int(window['start_frame']))
            if sample['sample_id'] != window['sample_id']:
                raise ValueError('Fixed sample identity changed')
            indices = sample['source_frame_indices'].numpy()
            if len(indices) != int(window['num_frames']) or not np.all(np.diff(indices) == 1):
                raise ValueError('Preview must use the requested continuous original frames')
            torch.cuda.reset_peak_memory_stats()
            true_frames = int(sample.get('video_true_num_frames', sample['video'].shape[1]))
            batch = training_layout_batch(sample)
            with torch.no_grad():
                latent = generate_latents(model, batch,
                    num_steps=selection['denoise_steps'], guidance=selection['guidance'],
                    seed=selection['seed'], context_sigma=selection['context_sigma'],
                    history_mode=history_mode)
                decoded, decoder_mode = decode_preview(model, batch, latent,
                    history_mode=history_mode, true_frames=true_frames,
                    frames_per_chunk=model.config.frames_per_chunk)
            if not torch.isfinite(decoded).all():
                raise ValueError('Nonfinite decoded RGB')
            pred = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
            gt = sample['video'].permute(1, 2, 3, 0).cpu().numpy()[:true_frames]
            if pred.shape != gt.shape:
                raise ValueError(f'Generated/GT shape mismatch: {pred.shape}, {gt.shape}')
            pred, gt = pred[:true_frames, :360], gt[:true_frames, :360]
            fps = float(sample['conditioning_fps'])
            preview = np.concatenate((gt, pred), axis=2)
            videos = dict(generated=pred, gt=gt, preview=preview)
            short_first = short_count = None
            if not full_segment:
                short_first = round(window['short_start_seconds'] * fps)
                short_count = round(window['short_duration_seconds'] * fps)
                if not 0 <= short_first < short_first + short_count <= len(pred):
                    raise ValueError('Short preview crop is outside the long rollout')
                videos['short_preview'] = preview[short_first:short_first + short_count]
            for name, frames in videos.items():
                imageio.mimwrite(str(output / (name + '.mp4')), frames, fps=fps,
                    codec='libx264', macro_block_size=1, ffmpeg_params=['-threads', '1', '-movflags', '+faststart'])
            elapsed = time.monotonic() - job_start
            metadata = dict(window, version=selection['version'],
                checkpoint_step=selection['checkpoint_step'], checkpoint=str(Path(args.checkpoint).resolve()),
                seed=selection['seed'], denoise_steps=selection['denoise_steps'],
                guidance=selection['guidance'], context_sigma=selection['context_sigma'],
                caption=window['caption'], conditioning_caption=sample['ai_caption'],
                preview_mode=selection.get('preview_mode', 'window'),
                video_temporal_padding=int(sample.get('video_temporal_padding', 0)),
                frames=len(pred), fps=fps, latent_frames=latent.shape[2],
                frames_per_chunk=model.config.frames_per_chunk,
                local_attention_frames=model.config.local_attention_frames,
                history=history_mode, history_mode=history_mode, decoder_mode=decoder_mode,
                modalities=['text','video'], frame_stride=1,
                source_frame_indices=indices.tolist(), short_start_frame=short_first,
                short_num_frames=short_count, runtime_seconds=elapsed,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                toml_sha256=hashlib.sha256(Path(args.toml).read_bytes()).hexdigest(),
                videos={name:name+'.mp4' for name in videos})
            write_json(output / 'manifest.json', metadata)
            print(json.dumps(dict(event='window_completed', id=window['id'], seconds=elapsed,
                                  frames=len(pred)), ensure_ascii=False), flush=True)
            del latent, decoded, pred, gt, preview, videos, sample, batch
            torch.cuda.empty_cache()
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        write_json(worker_dir / 'exit.json', dict(claim, elapsed_seconds=time.monotonic()-started,
            completed=[i for i in args.ids if (root/windows[i]['output_dir']/'manifest.json').exists()],
            exit_code=0 if error is None else 1, error=error))
        if initialized:
            distributed.destroy_process_group()


if __name__ == '__main__':
    main()
