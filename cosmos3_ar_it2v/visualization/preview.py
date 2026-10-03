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
        tiers = tuple(sorted({17, 33, 49, 65, 81, 97,
                              *(int(w['num_frames']) for w in selected)}, reverse=True))
        # Retain minimum tier17 so dataset row numbering matches the training dataset.
        for split in {w['split'] for w in selected}:
            datasets[split] = instantiate(ds_cfg, split=split, iterable_shuffle=False,
                random_window=False, cfg_dropout_rate=0., clip_frame_tiers=tiers)
        for window in selected:
            output = root / window['output_dir']
            output.mkdir(parents=False, exist_ok=False)
            write_json(output / 'started.json', {**claim, 'window': window})
            job_start = time.monotonic()
            print(json.dumps(dict(event='window_started', id=window['id']), ensure_ascii=False), flush=True)
            sample = datasets[window['split']].get_item_at_window(
                int(window['row_index']), window_start=int(window['start_frame']))
            if sample['sample_id'] != window['sample_id']:
                raise ValueError('Fixed sample identity changed')
            indices = sample['source_frame_indices'].numpy()
            if len(indices) != int(window['num_frames']) or not np.all(np.diff(indices) == 1):
                raise ValueError('Preview must use the requested continuous original frames')
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                latent = generate_latents(model, training_layout_batch(sample),
                    num_steps=selection['denoise_steps'], guidance=selection['guidance'],
                    seed=selection['seed'], context_sigma=selection['context_sigma'])
                decoded = model.decode(latent.to(**model.tensor_kwargs))
            if not torch.isfinite(decoded).all():
                raise ValueError('Nonfinite decoded RGB')
            pred = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
            gt = sample['video'].permute(1, 2, 3, 0).cpu().numpy()
            if pred.shape != gt.shape:
                raise ValueError(f'Generated/GT shape mismatch: {pred.shape}, {gt.shape}')
            pred, gt = pred[:, :360], gt[:, :360]
            fps = float(sample['conditioning_fps'])
            short_first = round(window['short_start_seconds'] * fps)
            short_count = round(window['short_duration_seconds'] * fps)
            if not 0 <= short_first < short_first + short_count <= len(pred):
                raise ValueError('Short preview crop is outside the long rollout')
            preview = np.concatenate((gt, pred), axis=2)
            videos = dict(generated=pred, gt=gt, preview=preview,
                          short_preview=preview[short_first:short_first + short_count])
            for name, frames in videos.items():
                imageio.mimwrite(str(output / (name + '.mp4')), frames, fps=fps,
                    codec='libx264', macro_block_size=1, ffmpeg_params=['-threads', '1', '-movflags', '+faststart'])
            elapsed = time.monotonic() - job_start
            metadata = dict(window, version=selection['version'],
                checkpoint_step=selection['checkpoint_step'], checkpoint=str(Path(args.checkpoint).resolve()),
                seed=selection['seed'], denoise_steps=selection['denoise_steps'],
                guidance=selection['guidance'], context_sigma=selection['context_sigma'],
                caption=sample['ai_caption'], frames=len(pred), fps=fps, latent_frames=latent.shape[2],
                frames_per_chunk=model.config.frames_per_chunk,
                local_attention_frames=model.config.local_attention_frames,
                history='generated', modalities=['text','video'], frame_stride=1,
                source_frame_indices=indices.tolist(), short_start_frame=short_first,
                short_num_frames=short_count, runtime_seconds=elapsed,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                toml_sha256=hashlib.sha256(Path(args.toml).read_bytes()).hexdigest(),
                videos={name:name+'.mp4' for name in videos})
            write_json(output / 'manifest.json', metadata)
            print(json.dumps(dict(event='window_completed', id=window['id'], seconds=elapsed,
                                  frames=len(pred)), ensure_ascii=False), flush=True)
            del latent, decoded, pred, gt, preview, videos, sample
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
