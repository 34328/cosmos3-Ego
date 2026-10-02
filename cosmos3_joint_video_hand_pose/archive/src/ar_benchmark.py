#!/usr/bin/env python3
r"""Time AR v0.1 inference per chunk: video denoising, action denoising, VAE encode/decode.

The sampler re-runs the truncated teacher-forcing forward at every solver step (no KV
cache), so a step costs more for later chunks. ``--steps-measured`` solver steps are timed
per phase and chunk and extrapolated to ``--steps`` (default 20); each step has the same
cost within a chunk. Resolutions are obtained by resizing the same clip (content does not
change the cost).

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/packages/cosmos3 \
        torchrun --nproc_per_node=1 -m cosmos3_joint_video_hand_pose.src.ar_benchmark \
        --ckpt <iter>/model --output <dir> --resolutions 320x192,640x368,1280x736
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .ar_inference import DEFAULT_TOML, ARSampler, _training_layout_batch, chunk_frame_ranges, flow_sigmas


def _sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def resize_sample(sample: dict, width: int, height: int) -> dict:
    """Resize the uint8 ``[3,T,H,W]`` video and its ``image_size`` record."""
    sample = dict(sample)
    video = sample["video"]
    h0, w0 = video.shape[-2:]
    frames = video.permute(1, 0, 2, 3).float()  # [T,3,H,W]
    frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
    sample["video"] = frames.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()
    size = sample["image_size"].clone()
    sample["image_size"] = torch.tensor(
        [height, width, round(float(size[2]) * height / h0), round(float(size[3]) * width / w0)], dtype=size.dtype
    )
    return sample


@torch.no_grad()
def benchmark(model, sample: dict, chunk_size: int, window: int, steps_measured: int, steps: int) -> dict:
    from cosmos_framework.utils import misc

    torch.cuda.reset_peak_memory_stats()
    t0 = _sync_time()
    batch = misc.to(_training_layout_batch(sample), device="cuda")
    sampler = ARSampler(model, batch, chunk_size, window)  # includes VAE encode of the clip
    encode_s = _sync_time() - t0
    video = sampler.gt_video.clone()
    action = sampler.gt_action.clone()
    video[:, :, 1:] = torch.randn_like(video[:, :, 1:])
    action[sampler.tokens_per_latent :] = torch.randn_like(action[sampler.tokens_per_latent :])
    sig = flow_sigmas(steps, 5.0)
    chunks = []
    for index, (start, end) in enumerate(chunk_frame_ranges(sampler.num_frames, chunk_size)):
        frame_sigmas = torch.zeros(sampler.num_frames)
        record = {"chunk": index + 1, "latent_frames": [start, end]}
        for phase in ("video", "action"):
            sampler.forward(video, action, start, end, frame_sigmas)  # warm-up (block-mask build)
            t0 = _sync_time()
            for i in range(steps_measured):
                frame_sigmas[start:end] = sig[i]
                pred_video, pred_action = sampler.forward(video, action, start, end, frame_sigmas)
                if phase == "video":
                    video[:, :, start:end] += (sig[i + 1] - sig[i]) * pred_video[:, :, start:end]
                else:
                    rows = sampler.rows(start, end)
                    action[rows] += (sig[i + 1] - sig[i]) * pred_action[rows]
            per_step = (_sync_time() - t0) / steps_measured
            record[f"{phase}_step_s"] = per_step
            record[f"{phase}_s"] = per_step * steps
        record["chunk_s"] = record["video_s"] + record["action_s"]
        chunks.append(record)
    t0 = _sync_time()
    model.decode(video.to(model.tensor_kwargs["dtype"]))
    decode_s = _sync_time() - t0
    height, width = (int(v) for v in sample["video"].shape[-2:])
    latent_hw = tuple(int(v) for v in sampler.gt_video.shape[-2:])
    return {
        "resolution": f"{width}x{height}",
        "latent_hw": latent_hw,
        "vision_tokens_per_latent": (latent_hw[0] + 1) // 2 * ((latent_hw[1] + 1) // 2),
        "num_latent_frames": sampler.num_frames,
        "steps": steps,
        "steps_measured": steps_measured,
        "vae_encode_s": encode_s,
        "vae_decode_s": decode_s,
        "chunks": chunks,
        "total_denoise_s": sum(c["chunk_s"] for c in chunks),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0, help="dataset index of a T=129 clip")
    parser.add_argument("--resolutions", default="320x192,480x288,640x368,960x544,1280x736")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--steps-measured", type=int, default=3)
    args = parser.parse_args()

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.generator.model_loader import _load_model
    from cosmos_framework.utils.lazy_config import instantiate

    from . import config as _config  # noqa: F401

    distributed.init()
    config = load_experiment_from_toml(args.toml, [
        "job.wandb_mode=disabled",
        "model.config.parallelism.data_parallel_shard_degree=1",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "model.config.parallelism.context_parallel_shard_degree=1",
        "model.config.parallelism.enable_inference_mode=true",
        "model.config.activation_checkpointing.mode=none",
    ])
    config.validate()
    config.freeze()
    model = instantiate(config.model).cuda()
    model.on_train_start()
    _load_model(model, checkpoint_path=args.ckpt, credential_path=None,
                keys_to_skip_loading=list(config.checkpoint.keys_to_skip_loading or []))
    model.eval()
    dataset = instantiate(config.dataloader_train.dataloader.datasets.egoverse.dataset,
                          iterable_shuffle=False, random_window=False, cfg_dropout_rate=0.0)
    base = dataset[args.index]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.resolutions.split(","):
        width, height = (int(v) for v in spec.lower().split("x"))
        try:
            record = benchmark(model, resize_sample(base, width, height), args.chunk_size, args.window,
                               args.steps_measured, args.steps)
        except torch.OutOfMemoryError as error:
            record = {"resolution": spec, "error": f"OOM: {str(error)[:160]}"}
        torch.cuda.empty_cache()
        results.append(record)
        print("AR_BENCH " + json.dumps(record), flush=True)
        (args.output / "benchmark.json").write_text(json.dumps(
            {"ckpt": args.ckpt, "gpu": torch.cuda.get_device_name(), "results": results}, indent=2) + "\n")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
