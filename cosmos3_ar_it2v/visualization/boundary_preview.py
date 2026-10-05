"""One complete short segment: decoder-prefix diagnosis and shared-boundary trial.

No optimizer, weight updates, altered training config, or independent block VAE.
Artifacts are private except the existing gallery's explicit MP4 allowlist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from .preview import write_json

VARIANTS = {
    'predicted_prefix': ('原先解码', '各块使用真实历史预测，整段预测 latent 连续解码。'),
    'gt_prefix_montage': ('匹配真实历史解码', '与原先解码使用完全相同的预测；每块在真实前缀下解码后拼接。这是单块诊断拼图，不是连续自由生成。'),
    'gt_reconstruction': ('真实视频重建', '真实视频经同一个 VAE 连续编码、连续解码，不经过 AR 预测。'),
    'shared_boundary': ('共享边界实验', '沿用真实历史预测；从第二个预测块起，锁定上个预测块的末 latent，仍生成原来的四个新 latent，最后连续解码。'),
}


def build_diagnostic_page(root):
    from .build_page import build_page, inline_json
    page = build_page(root / 'selection.json')
    descriptions = [dict(title=title, description=description)
                    for title, description in VARIANTS.values()]
    script = """
<script>
document.title='接缝诊断 · Step 1500';
document.querySelector('.intro h1').textContent='同一段动作，检查画面接缝。';
document.querySelector('.intro p').textContent='同一个完整短片，原速 30fps。前两项使用完全相同的预测，只改变解码历史；第四项单独尝试共享边界。所有实验均不更新模型权重。';
document.querySelector('.toolbar').hidden=true;
document.querySelector('.section-heading h2').textContent='训练集 · 取香块并摆入盒中';
document.querySelector('.section-heading p').textContent='1 个完整动作段 · 4 种显示方式 · 每段 6.2 秒';
document.querySelector('#noise-note').textContent='预测块使用此前的真实视频作为历史，因此属于诊断模式。历史写入噪声为 0.02。共享边界取自上一块的模型预测，不提供当前块的真实图像。';
document.querySelector('.endnote').textContent='左栏始终为原始真实视频；右栏方式见每个视频标题。匹配真实历史解码是单块诊断拼图；真实视频重建只检查 VAE。两者都不代表自由生成能力。';
const variants=__VARIANTS__;
cards.forEach((card,i)=>{
  card.video.previousElementSibling.lastElementChild.textContent=variants[i].title;
  card.description.textContent=variants[i].description;
  card.video.setAttribute('aria-label',variants[i].title+'，真实视频与诊断结果');
  card.video.closest('.sample').querySelector('.source-links').lastElementChild.textContent='右栏视频';
});
document.querySelector('#params').lastElementChild.lastElementChild.textContent='左：真实视频　右：标题所示诊断';
</script>
"""
    page.write_text(page.read_text().replace('</body>',
        script.replace('__VARIANTS__', inline_json(descriptions)) + '</body>'), encoding='utf-8')
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--id', default='train_02')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--toml', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos3_ar_it2v.inference import generate_latents, load_model, training_layout_batch
    from .boundary_sampling import generate_boundary_latents
    from .decoder_diagnostic import decode_comparison, save_latent_archive

    selection = json.loads(args.selection.read_text())
    matches = [w for w in selection['windows'] if w['id'] == args.id]
    if len(matches) != 1 or selection.get('preview_mode') != 'full_segment':
        raise ValueError('Choose exactly one original complete segment')
    window = matches[0]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)  # Atomic claim; never overwrite a prior run.
    started = time.monotonic()
    claim = dict(pid=os.getpid(), hostname=socket.gethostname(),
                 gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), started_unix=time.time())
    write_json(root / 'started.json', claim)
    initialized = False
    error = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('An allocated idle GPU is required')
        distributed.init()
        initialized = True
        model, config = load_model(args.toml, args.checkpoint)
        ds_cfg = next(iter(config.dataloader_train.dataloader.datasets.values())).dataset
        dataset = instantiate(ds_cfg, split=window['split'], iterable_shuffle=False,
            random_window=False, cfg_dropout_rate=0., clip_frame_tiers=(17,33,49,65,81,97),
            sample_mode='full_segment', long_segment_policy='error',
            max_sequence_length=None, segment_statistics_path=None)
        rows = [i for i, r in enumerate(dataset.rows) if r['sample_id'] == window['sample_id']]
        if len(rows) != 1:
            raise ValueError('Complete segment must match exactly once')
        row = dataset.rows[rows[0]]
        if (int(row['start_idx']) != int(window['segment_start_frame']) or
            int(row['end_idx']) != int(window['segment_end_frame_exclusive']) or
            row['text_normalized'].strip() != window['caption'].strip()):
            raise ValueError('Original segment bounds/caption changed')
        sample = dataset[rows[0]]
        indices = sample['source_frame_indices'].numpy()
        true_frames = int(sample['video_true_num_frames'])
        fps = float(sample['conditioning_fps'])
        if (true_frames != int(window['num_frames']) or len(indices) != true_frames
            or not np.all(np.diff(indices) == 1) or fps != selection['fps']):
            raise ValueError('Original continuous frames or fps changed')
        if model.config.frames_per_chunk != 4:
            raise ValueError('This trial preserves the original C4 layout')
        torch.cuda.reset_peak_memory_stats()
        settings = dict(num_steps=selection['denoise_steps'], guidance=selection['guidance'],
                        seed=selection['seed'], context_sigma=selection['context_sigma'],
                        history_mode='gt')
        provenance = dict(window, checkpoint=str(Path(args.checkpoint).resolve()),
            toml_sha256=hashlib.sha256(Path(args.toml).read_bytes()).hexdigest(),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            seed=selection['seed'], denoise_steps=selection['denoise_steps'],
            guidance=selection['guidance'], context_sigma=selection['context_sigma'],
            source_frame_indices=indices.tolist(), conditioning_caption=sample['ai_caption'],
            frames=true_frames, fps=fps, history_mode='gt')
        with torch.no_grad():
            begin = time.monotonic()
            predicted = generate_latents(model, training_layout_batch(sample), **settings).detach().cpu()
            baseline_seconds = time.monotonic() - begin
            print(json.dumps(dict(event='baseline_latents_ready', seconds=baseline_seconds)), flush=True)
            begin = time.monotonic()
            boundary, reference = generate_boundary_latents(model, training_layout_batch(sample),
                                                            return_reference=True, **settings)
            boundary, reference = boundary.detach().cpu(), reference.detach().cpu()
            boundary_seconds = time.monotonic() - begin
            if not torch.equal(predicted[:, :, :1], reference[:, :, :1]):
                raise ValueError('The two runs did not encode the same first-frame condition')
            if not torch.equal(predicted[:, :, :5], boundary[:, :, :5]):
                raise ValueError('The unchanged first target C4 differs from baseline')
            save_latent_archive(root/'baseline_latents.pt', predicted, reference,
                                true_frames=true_frames, metadata=provenance)
            save_latent_archive(root/'boundary_latents.pt', boundary, reference,
                true_frames=true_frames, metadata=dict(provenance, boundary='previous predicted last latent; original time'))
            print(json.dumps(dict(event='boundary_latents_ready', seconds=boundary_seconds)), flush=True)
            decoded = decode_comparison(model, predicted, reference, true_frames=true_frames)
            boundary_rgb = model.decode(boundary.to(**model.tensor_kwargs)).detach().cpu()[:, :, :true_frames]
            if not torch.isfinite(boundary_rgb).all() or boundary_rgb.shape != decoded['predicted_prefix'].shape:
                raise ValueError('Shared-boundary decoded RGB is invalid')
            decoded['shared_boundary'] = boundary_rgb

        gt = sample['video'].permute(1,2,3,0).cpu().numpy()[:true_frames, :360]
        windows = []
        for name, (title, description) in VARIANTS.items():
            output = root / name
            output.mkdir()
            pred = ((decoded[name][0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).numpy()[:, :360]
            if pred.shape != gt.shape:
                raise ValueError('Diagnostic/GT RGB geometry mismatch')
            for filename, frames in dict(gt=gt, generated=pred, preview=np.concatenate((gt,pred), axis=2)).items():
                imageio.mimwrite(str(output/(filename+'.mp4')), frames, fps=fps, codec='libx264',
                    macro_block_size=1, ffmpeg_params=['-threads','1','-movflags','+faststart'])
            variant = dict(window, id=name, title=title, output_dir=name, history_mode='gt')
            windows.append(variant)
            write_json(output/'manifest.json', dict(provenance, variant=name, title=title,
                explanation=description, version=selection['version'],
                checkpoint_step=selection['checkpoint_step'], history='gt',
                preview_mode='full_segment', latent_frames=predicted.shape[2], frames_per_chunk=4,
                local_attention_frames=model.config.local_attention_frames,
                video_temporal_padding=int(sample['video_temporal_padding']),
                videos={n:n+'.mp4' for n in ('gt','generated','preview')}))
        display = {k:selection[k] for k in ('version','checkpoint_step','seed','denoise_steps',
                                           'guidance','context_sigma','fps')}
        display.update(preview_mode='full_segment', windows=windows)
        write_json(root/'selection.json', display)
        build_diagnostic_page(root)
        write_json(root/'completed.json', dict(provenance, variants=list(VARIANTS),
            baseline_generation_seconds=baseline_seconds, boundary_generation_seconds=boundary_seconds,
            total_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            identical_predictions_in_decoder_comparison=True, optimizer_updates=0))
        print(json.dumps(dict(event='completed', output=str(root))), flush=True)
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        write_json(root/'exit.json', dict(claim, elapsed_seconds=time.monotonic()-started,
                                          exit_code=0 if error is None else 1, error=error))
        if initialized:
            distributed.destroy_process_group()


if __name__ == '__main__':
    main()
