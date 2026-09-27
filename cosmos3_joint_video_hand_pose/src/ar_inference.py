#!/usr/bin/env python3
r"""Chunked autoregressive joint video-action sampling for AR v0.1 (lingbot-va order).

For every chunk the video latents are denoised first and the actions second,
each with an Euler flow-matching solver. Every solver step re-runs the replayed
teacher-forcing forward over the clip truncated to the end of the current
chunk: earlier chunks are clean conditions, the current chunk is noisy.
``--history`` selects what the model is conditioned on:

* ``oracle``: ground-truth history, and the action solver reads the ground-truth
  video of its own chunk (upper bound of the action branch);
* ``gt``: ground-truth history, actions read the generated video of their chunk;
* ``generated``: free rollout on the model's own earlier chunks.

Under the lingbot-va mask this is
exactly the receptive field of training -- noisy video sees text and clean
earlier chunks; noisy actions additionally see the clean video of their chunk
-- so no separate KV cache is needed (v0.1 trades speed for exactness).

``--consistency-check`` verifies that equivalence on a real clip: predictions for
one chunk from a full-clip forward (every chunk noisy, as in training) must match
a forward truncated to that chunk with clean ground-truth history.

Run on one GPU from the repository root::

    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD:$PWD/packages/cosmos3 \
        torchrun --nproc_per_node=1 -m cosmos3_joint_video_hand_pose.src.ar_inference \
        --ckpt <iter_dir>/model --output <dir> [--num-samples 4] [--history oracle,gt,generated]
        [--consistency-check]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOML = PROJECT_ROOT / "configs/ar_v0_1.toml"


def flow_sigmas(steps: int, shift: float) -> torch.Tensor:
    """Shifted flow-matching noise levels from 1 to 0 (``steps + 1`` values)."""
    u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return (shift * u / (1 + (shift - 1) * u)).float()


def chunk_frame_ranges(num_frames: int, chunk_size: int) -> list[tuple[int, int]]:
    """``[start, end)`` latent-frame ranges of chunks 1.. in the ``[1, C, C, ...]`` partition."""
    return [(start, min(start + chunk_size, num_frames)) for start in range(1, num_frames, chunk_size)]


def collapse_state_group(rows: torch.Tensor, tokens_per_latent: int) -> torch.Tensor:
    """Inverse of ``expand_state_group``: ``[L*K, D]`` -> ``[1 + (L-1)K, D]``."""
    return torch.cat([rows[:1], rows[tokens_per_latent:]], dim=0)


HISTORY_MODES = ("oracle", "gt", "generated")


class ARSampler:
    """Owns one clip's clean inputs and runs teacher-forcing forwards on truncated clips."""

    def __init__(self, model, batch: dict, chunk_size: int, window: int | None):
        self.model = model
        self.chunk_size = int(chunk_size)
        self.window = window
        (
            self.text,
            self.plans,
            self.gen,
            self.memory_info,
            _resolutions,
            _shapes,
        ) = model._prepare_training_data(batch, 0)
        while isinstance(self.plans, (list, tuple)) and len(self.plans) == 1 and isinstance(self.plans[0], (list, tuple)):
            self.plans = self.plans[0]
        if len(self.plans) != 1 or not dataclasses.is_dataclass(self.plans[0]):
            raise ValueError(f"ARSampler handles one clip at a time, got plans {self.plans!r}")
        self.gt_video = self.gen.x0_tokens_vision[0].float()  # [1,C,L,H,W]
        action = self.gen.x0_tokens_action[0]
        self.gt_action = action.reshape(-1, action.shape[-1]).float()  # [L*K,D]
        self.num_frames = int(self.gt_video.shape[2])
        self.tokens_per_latent = int(model.config.action_tokens_per_latent)
        if self.gt_action.shape[0] != self.num_frames * self.tokens_per_latent:
            raise ValueError(f"expected {self.num_frames} x K={self.tokens_per_latent} action rows")
        raw_dim = self.gen.raw_action_dim[0] if self.gen.raw_action_dim is not None else None
        self.raw_action_dim = int(torch.as_tensor(raw_dim).reshape(-1)[0]) if raw_dim is not None else None

    def rows(self, start: int, end: int) -> slice:
        return slice(start * self.tokens_per_latent, end * self.tokens_per_latent)

    @torch.no_grad()
    def forward(
        self,
        video: torch.Tensor,
        action: torch.Tensor,
        first_noisy: int,
        end: int,
        frame_sigmas: torch.Tensor,
        clean_video: torch.Tensor | None = None,
        clean_action: torch.Tensor | None = None,
    ):
        """Teacher-forcing forward on frames ``[0, end)``; frames ``< first_noisy`` are clean conditions.

        ``video``/``action`` feed the noisy pass. The clean pass reads ``clean_video``/
        ``clean_action`` when given (training layout: ground truth), else the same tensors.
        Under the lingbot mask a chunk's noisy queries only read clean earlier chunks and,
        for actions, the clean video of their own chunk, so sampling can pass one tensor.

        Returns velocity predictions ``(video [1,C,end,H,W], action [end*K, D])``.
        """
        model, K = self.model, self.tokens_per_latent
        clean_video = video if clean_video is None else clean_video
        clean_action = action if clean_action is None else clean_action
        plan = dataclasses.replace(self.plans[0], condition_frame_indexes_vision=list(range(first_noisy)))
        gen = dataclasses.replace(
            self.gen,
            x0_tokens_vision=[clean_video[:, :, :end].contiguous()],
            x0_tokens_action=[clean_action[: end * K].contiguous()],
        )
        max_timestep = float(model.rectified_flow_video.noise_scheduler.config.num_train_timesteps)
        timesteps = (frame_sigmas[:end].float() * max_timestep).reshape(1, end).cpu()  # [1,end]
        packed = model._pack_input_sequence(
            [plan],
            self.text,
            gen,
            timesteps,
            skip_text_tokens=self.memory_info["skip_text"],
            initial_mrope_temporal_offset=self.memory_info["initial_temporal_offset"],
        )
        info = dict(self.memory_info)
        with model.ar_context(self.chunk_size, self.window):
            info = model.pre_noise_memory_hook(packed, gen, info)
            if clean_video is not video or clean_action is not action:
                packed.vision.tokens = [video[:, :, :end].contiguous()]
                packed.action.tokens = [action[: end * K].contiguous()]
            packed.to_cuda()
            model._cast_generated_tokens_to_precision(packed)
            out = model.denoise(data_batch_packed=packed, memory=model.build_memory_state(packed, info))
        pred_video = out["preds_vision"][0].float().reshape(1, -1, end, *video.shape[-2:])
        pred_action = out["preds_action"][0].float().reshape(end * K, -1)
        return pred_video, pred_action

    def _zero_padding(self, action: torch.Tensor) -> None:
        if self.raw_action_dim is not None:
            action[:, self.raw_action_dim :] = 0

    @torch.no_grad()
    def sample(self, video_steps: int, action_steps: int, video_shift: float, action_shift: float, history: str, seed: int):
        """Generate every chunk and return the predictions.

        ``history='gt'`` conditions each chunk on ground-truth history (teacher forcing) and
        ``history='generated'`` on the model's own earlier chunks. Predictions are copied to a
        separate output buffer per chunk, so GT history never overwrites earlier predictions.
        ``history='oracle'`` is ``gt`` plus the ground-truth video of the current chunk as the
        action solver's condition; its returned video is still the generated one.
        """
        if history not in HISTORY_MODES:
            raise ValueError(f"unknown history mode {history!r}")
        generator = torch.Generator(device=self.gt_video.device).manual_seed(seed)
        video = self.gt_video.clone()
        action = self.gt_action.clone()
        video[:, :, 1:] = torch.randn(video[:, :, 1:].shape, generator=generator, device=video.device)
        action[self.tokens_per_latent :] = torch.randn(
            action[self.tokens_per_latent :].shape, generator=generator, device=action.device
        )
        self._zero_padding(action)
        out_video, out_action = self.gt_video.clone(), self.gt_action.clone()
        video_sigmas, action_sigmas = flow_sigmas(video_steps, video_shift), flow_sigmas(action_steps, action_shift)
        for start, end in chunk_frame_ranges(self.num_frames, self.chunk_size):
            if history in ("oracle", "gt"):
                video[:, :, :start] = self.gt_video[:, :, :start]
                action[self.rows(0, start)] = self.gt_action[self.rows(0, start)]
            frame_sigmas = torch.zeros(self.num_frames)
            for i in range(video_steps):
                frame_sigmas[start:end] = video_sigmas[i]
                pred_video, _ = self.forward(video, action, start, end, frame_sigmas)
                video[:, :, start:end] += (video_sigmas[i + 1] - video_sigmas[i]) * pred_video[:, :, start:end]
            out_video[:, :, start:end] = video[:, :, start:end]
            if history == "oracle":
                video[:, :, start:end] = self.gt_video[:, :, start:end]
            for i in range(action_steps):
                frame_sigmas[start:end] = action_sigmas[i]
                _, pred_action = self.forward(video, action, start, end, frame_sigmas)
                rows = self.rows(start, end)
                action[rows] += (action_sigmas[i + 1] - action_sigmas[i]) * pred_action[rows]
                self._zero_padding(action)
            out_action[self.rows(start, end)] = action[self.rows(start, end)]
        self._zero_padding(out_action)
        return out_video, out_action

    @torch.no_grad()
    def consistency_check(self, chunk_index: int, seed: int) -> dict:
        """Chunk predictions: full noisy clip (training layout) vs truncated clip with clean GT history."""
        generator = torch.Generator(device=self.gt_video.device).manual_seed(seed)
        ranges = chunk_frame_ranges(self.num_frames, self.chunk_size)
        start, end = ranges[chunk_index]
        chunk_sigmas = (torch.rand(len(ranges), generator=generator, device=generator.device) * 0.8 + 0.1).cpu()
        frame_sigmas = torch.zeros(self.num_frames)
        for (s, e), sigma in zip(ranges, chunk_sigmas):
            frame_sigmas[s:e] = sigma
        per_row = frame_sigmas.repeat_interleave(self.tokens_per_latent).to(self.gt_action.device)[:, None]
        noise_video = torch.randn(self.gt_video.shape, generator=generator, device=self.gt_video.device)
        noise_action = torch.randn(self.gt_action.shape, generator=generator, device=self.gt_action.device)
        sig_v = frame_sigmas.to(self.gt_video.device).view(1, 1, -1, 1, 1)
        noisy_video = self.gt_video * (1 - sig_v) + noise_video * sig_v
        noisy_action = self.gt_action * (1 - per_row) + noise_action * per_row
        self._zero_padding(noisy_action)
        noisy_video[:, :, :1] = self.gt_video[:, :, :1]
        noisy_action[self.rows(0, 1)] = self.gt_action[self.rows(0, 1)]

        # Training layout: clean pass on ground truth, noisy pass on the noised clip.
        full_video, full_action = self.forward(
            noisy_video, noisy_action, 1, self.num_frames, frame_sigmas,
            clean_video=self.gt_video, clean_action=self.gt_action,
        )
        # Inference layout: clip truncated at the chunk end, clean ground-truth history.
        history_video = noisy_video.clone()
        history_action = noisy_action.clone()
        history_video[:, :, :start] = self.gt_video[:, :, :start]
        history_action[self.rows(0, start)] = self.gt_action[self.rows(0, start)]
        trunc_video, trunc_action = self.forward(
            history_video, history_action, start, end, frame_sigmas,
            clean_video=self.gt_video, clean_action=self.gt_action,
        )

        rows = self.rows(start, end)

        def rel(a, b):
            return float((a - b).norm() / b.norm().clamp_min(1e-12))

        return {
            "chunk_index": chunk_index,
            "frames": [start, end],
            "sigma": float(chunk_sigmas[chunk_index]),
            "video_relative_l2": rel(trunc_video[:, :, start:end], full_video[:, :, start:end]),
            "action_relative_l2": rel(trunc_action[rows], full_action[rows]),
            "video_pred_norm": float(full_video[:, :, start:end].norm()),
            "action_pred_norm": float(full_action[rows].norm()),
        }


def _training_layout_batch(sample: dict) -> dict:
    """One packed batch exactly as ``PackingDataLoader`` builds it for a single training sample."""
    from collections import deque

    from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn

    packer = object.__new__(PackingDataLoader)
    packer.buffers = [deque()]
    packer.dataloaders = [iter([custom_collate_fn([sample])])]
    output: dict = {}
    PackingDataLoader._update_output_batch(packer, output, PackingDataLoader._get_next_sample(packer, 0))
    return output


def _to_uint8_frames(video: torch.Tensor, height: int = 360) -> np.ndarray:
    """``[1,3,T,H,W]`` in [-1, 1] (or uint8 ``[3,T,H,W]``) -> ``[T,height,W,3]`` uint8."""
    if video.dtype == torch.uint8:
        frames = video.reshape(3, *video.shape[-3:])
    else:
        frames = ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).reshape(3, *video.shape[-3:])
    return frames[:, :, :height].permute(1, 2, 3, 0).cpu().numpy()


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def per_chunk_metrics(pred57, ref57, builder, pred_frames, gt_frames, num_frames, chunk_size, K, tcf=4) -> list:
    """Error of every generated chunk: pose error at the chunk's last action and video PSNR.

    Latent ``j >= 1`` covers collapsed action rows ``1+(j-1)K .. jK`` and pixel frames ``tcf*j-tcf+1 .. tcf*j``.
    """
    pred, ref = builder.decode(pred57), builder.decode(ref57)
    streams = ("headcam_f0", "right_wrist_f0", "left_wrist_f0")
    records = []
    for index, (start, end) in enumerate(chunk_frame_ranges(num_frames, chunk_size)):
        first_row, last_row = 1 + (start - 1) * K, (end - 1) * K
        record = {"chunk": index + 1, "latent_frames": [start, end]}
        for name in streams:
            p, r = getattr(pred, name).numpy(), getattr(ref, name).numpy()
            err = np.linalg.norm(p[first_row : last_row + 1, :3, 3] - r[first_row : last_row + 1, :3, 3], axis=-1)
            record[f"{name[:-3]}_end_pos_err_mm"] = float(err[-1] * 1000)
            record[f"{name[:-3]}_mean_pos_err_mm"] = float(err.mean() * 1000)
        for side in ("right", "left"):
            p = getattr(pred, f"{side}_keypoints_f0").numpy()[first_row : last_row + 1]
            r = getattr(ref, f"{side}_keypoints_f0").numpy()[first_row : last_row + 1]
            record[f"{side}_hand_mpjpe_mm"] = float(np.linalg.norm(p - r, axis=-1).mean() * 1000)
        frames = slice(tcf * start - tcf + 1, tcf * (end - 1) + 1)
        record["video_psnr"] = _psnr(pred_frames[frames], gt_frames[frames])
        records.append(record)
    return records


def _write_side_by_side(path: Path, gt: np.ndarray, pred: np.ndarray, fps: float) -> None:
    import imageio.v2 as imageio

    frames = np.concatenate([gt, pred], axis=2)
    imageio.mimwrite(
        path, list(frames), format="FFMPEG", fps=fps, codec="libx264", pixelformat="yuv420p",
        macro_block_size=1, ffmpeg_log_level="error",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="DCP model directory (contains .metadata)")
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML, help="AR experiment TOML (default: %(default)s)")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--indices", default=None, help="comma-separated dataset indices (overrides --num-samples)")
    parser.add_argument("--num-samples", type=int, default=4, help="first N clips with --clip-frames frames")
    parser.add_argument("--clip-frames", type=int, default=129)
    parser.add_argument("--chunk-size", type=int, default=4, help="inference chunk C (lingbot demo: 4)")
    parser.add_argument("--window", type=int, default=30, help="block-id window (lingbot demo attn_window: 30)")
    parser.add_argument("--video-steps", type=int, default=20)
    parser.add_argument("--action-steps", type=int, default=20)
    parser.add_argument("--history", default="generated",
                        help=f"comma-separated subset of {HISTORY_MODES} (see module docstring)")
    parser.add_argument("--episodes-manifest", default=None, help="override the dataset episodes CSV (e.g. held-out)")
    parser.add_argument("--segments-manifest", default=None, help="override the dataset segments CSV")
    parser.add_argument("--split", default="train", help="manifest split to read")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--consistency-check", action="store_true", help="only run the TF consistency check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    histories = [h.strip() for h in args.history.split(",") if h.strip()]
    if not histories or any(h not in HISTORY_MODES for h in histories):
        raise SystemExit(f"--history must be a comma-separated subset of {HISTORY_MODES}, got {args.history!r}")
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils import distributed, misc
    from cosmos_framework.utils.generator.model_loader import _load_model
    from cosmos_framework.utils.lazy_config import instantiate

    from . import config as _config  # noqa: F401
    from .action import Action57Builder
    from .trajectory_metrics import evaluate_actions

    distributed.init()
    config = load_experiment_from_toml(
        args.toml,
        [
            "job.wandb_mode=disabled",
            "model.config.parallelism.data_parallel_shard_degree=1",
            "model.config.parallelism.data_parallel_replicate_degree=1",
            "model.config.parallelism.context_parallel_shard_degree=1",
            # As the official inference loader: bf16 weights, no fp32 master copy.
            "model.config.parallelism.enable_inference_mode=true",
            "model.config.activation_checkpointing.mode=none",
        ],
    )
    config.validate()
    config.freeze()
    model = instantiate(config.model).cuda()
    model.on_train_start()
    _load_model(model, checkpoint_path=args.ckpt, credential_path=None,
                keys_to_skip_loading=list(config.checkpoint.keys_to_skip_loading or []))
    model.eval()

    dataset_cfg = config.dataloader_train.dataloader.datasets.egoverse.dataset
    overrides = {"iterable_shuffle": False, "random_window": False, "cfg_dropout_rate": 0.0}
    if args.episodes_manifest:
        overrides["episodes_manifest"] = args.episodes_manifest
    if args.segments_manifest:
        overrides["segments_manifest"] = args.segments_manifest
    if args.split != "train":
        overrides["split"] = args.split
    dataset = instantiate(dataset_cfg, **overrides)
    raw_dataset = dataset.dataset
    if args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        indices = [i for i, row in enumerate(raw_dataset.rows) if row["_clip_frames"] == args.clip_frames][: args.num_samples]
    rf_cfg = config.model.config.rectified_flow_training_config
    video_shift = float(rf_cfg.shift["480"] if not isinstance(rf_cfg.shift, (int, float)) else rf_cfg.shift)
    action_shift = float(rf_cfg.shift_action if rf_cfg.shift_action is not None else video_shift)
    future_normalizer = dataset_cfg.future_normalizer
    builder = Action57Builder(state_normalizer=dataset_cfg.state_normalizer, future_normalizer=future_normalizer,
                              rigid_pose_frame_delta=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index in indices:
        raw = raw_dataset[index]
        # Match the training batch layout: RankPartitionedDataLoader collates one sample, then
        # PackingDataLoader wraps every field of each packed sample in a list.
        batch = misc.to(_training_layout_batch(dataset[index]), device="cuda")
        sampler = ARSampler(model, batch, args.chunk_size, args.window)
        if args.consistency_check:
            record = {"index": index, "sample_id": raw["sample_id"], "clip_frames": raw["clip_frames"]}
            started = time.time()
            chunks = len(chunk_frame_ranges(sampler.num_frames, args.chunk_size))
            record["consistency"] = [sampler.consistency_check(c, args.seed + c) for c in sorted({0, chunks // 2, chunks - 1})]
            record["seconds"] = time.time() - started
            results.append(record)
            print("AR_SAMPLE " + json.dumps(record), flush=True)
            continue
        gt_frames = _to_uint8_frames(raw["video"])
        ref57 = collapse_state_group(raw["action"].float(), sampler.tokens_per_latent)
        for history in histories:
            record = {"index": index, "sample_id": raw["sample_id"], "clip_frames": raw["clip_frames"],
                      "history": history}
            started = time.time()
            video, action = sampler.sample(args.video_steps, args.action_steps, video_shift, action_shift,
                                           history, args.seed)
            K = sampler.tokens_per_latent
            pred57 = collapse_state_group(action[:, :57].cpu(), K)
            record["action_metrics"] = evaluate_actions(pred57, ref57, builder)
            pred_frames = _to_uint8_frames(model.decode(video.to(model.tensor_kwargs["dtype"])))
            record["video_psnr_future"] = _psnr(pred_frames[1:], gt_frames[1:])
            record["per_chunk"] = per_chunk_metrics(pred57, ref57, builder, pred_frames, gt_frames,
                                                    sampler.num_frames, args.chunk_size, K)
            stem = args.output / f"{index:04d}_{history}"
            np.savez(stem.with_suffix(".npz"), pred_action57=pred57.numpy(), ref_action57=ref57.numpy(),
                     pred_video=pred_frames, source_frame_indices=raw["source_frame_indices"].numpy())
            fps_model = float(raw["conditioning_fps"])
            _write_side_by_side(stem.parent / f"{stem.name}_model_time.mp4", gt_frames, pred_frames, fps_model)
            _write_side_by_side(stem.parent / f"{stem.name}_real_time.mp4", gt_frames, pred_frames,
                                fps_model / raw_dataset.speed_factor)
            record["seconds"] = time.time() - started
            results.append(record)
            print("AR_SAMPLE " + json.dumps(record), flush=True)
        del sampler, batch
        torch.cuda.empty_cache()
    summary = {"ckpt": args.ckpt, "toml": str(args.toml), "args": {k: str(v) for k, v in vars(args).items()}, "samples": results}
    name = "consistency.json" if args.consistency_check else f"results_{'_'.join(histories)}.json"
    (args.output / name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
