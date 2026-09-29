#!/usr/bin/env python3
"""Audit real packed EgoVerse batches without constructing the 16B model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from .ar_dataset import ar_latent_frames, ar_token_count
from .temporal import cosmos_wam_token_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOML = PROJECT_ROOT / "configs/ar_v0_2_fixed_camera.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _single_media(value, name: str) -> torch.Tensor:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must resolve to a tensor, got {type(value).__name__}")
    return value


def _visibility(value) -> torch.Tensor:
    value = _single_media(value, "hand_visibility")
    while value.ndim > 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"hand_visibility must resolve to [T,2], got {tuple(value.shape)}")
    return value


def _squeeze_media_batch(value, name: str, expected_ndim: int) -> torch.Tensor:
    value = _single_media(value, name)
    while value.ndim > expected_ndim and value.shape[0] == 1:
        value = value.squeeze(0)
    return value


def audit_batch(batch: dict, cap: int, action_tokens_per_latent: int | None = None) -> dict:
    """Validate one packed batch.

    ``action_tokens_per_latent=None`` checks the legacy contract (one action row per
    video frame); an integer ``K`` checks the AR contract (``K`` rows per latent frame).
    """
    required = (
        "text_token_ids",
        "video",
        "action",
        "action_raw",
        "hand_visibility",
        "sequence_plan",
        "raw_action_dim",
    )
    num_samples = len(batch["video"])
    for key in required:
        if key not in batch or len(batch[key]) != num_samples:
            raise ValueError(f"packed field {key!r} is not aligned to {num_samples} samples")

    frame_counts = []
    sample_ids = []
    num_tokens = 0
    for index in range(num_samples):
        text_ids = _single_media(batch["text_token_ids"][index], "text_token_ids")
        video = _squeeze_media_batch(batch["video"][index], "video", 4)
        action = _squeeze_media_batch(batch["action"][index], "action", 2)
        action_raw = _squeeze_media_batch(batch["action_raw"][index], "action_raw", 2)
        visible = _visibility(batch["hand_visibility"][index])
        if video.ndim != 4 or tuple(video.shape[0:1] + video.shape[2:]) != (3, 368, 640):
            raise ValueError(f"video must be [3,T,368,640], got {tuple(video.shape)}")
        frames = int(video.shape[1])
        if (frames - 1) % 4:
            raise ValueError(f"video length must be 1+4n, got {frames}")
        is_v02 = "ar_layout_version" in batch
        if is_v02:
            version = batch["ar_layout_version"][index]
            while isinstance(version, list) and len(version) == 1:
                version = version[0]
            if version != "joint_chunk_cond_v1" or action_tokens_per_latent != 8:
                raise ValueError("unsupported explicit v0.2 layout version or K")
            rows = (ar_latent_frames(frames) - 1) * 8
            states = _squeeze_media_batch(batch["ar_boundary_states"][index], "ar_boundary_states", 2)
            if states.shape != (ar_latent_frames(frames) - 1, 64) or states[:, 57:].count_nonzero():
                raise ValueError("invalid v0.2 boundary state candidates")
        else:
            rows = frames if action_tokens_per_latent is None else ar_latent_frames(frames) * action_tokens_per_latent
        if tuple(action.shape) != (rows, 64):
            raise ValueError(f"action must be [{rows},64], got {tuple(action.shape)} for T={frames}")
        if tuple(action_raw.shape) != (rows, 57):
            raise ValueError(f"action_raw must be [{rows},57], got {tuple(action_raw.shape)} for T={frames}")
        if tuple(visible.shape) != (rows, 2):
            raise ValueError(f"visibility must be [{rows},2], got {tuple(visible.shape)} for T={frames}")
        if torch.count_nonzero(action[:, 57:]).item() != 0:
            raise ValueError("action padding channels [57:64] are not all zero")
        raw_dim = torch.as_tensor(batch["raw_action_dim"][index]).reshape(-1)
        if raw_dim.numel() != 1 or int(raw_dim.item()) != 57:
            raise ValueError(f"raw_action_dim must be 57, got {raw_dim.tolist()}")
        frame_counts.append(frames)
        if action_tokens_per_latent is None:
            num_tokens += cosmos_wam_token_count(int(text_ids.numel()), frames)
        else:
            count = ar_token_count(int(text_ids.numel()), frames, action_tokens_per_latent)
            # Conservative admission budget; the current model fixes C=4.
            if is_v02:
                from .ar_v02_dataloader import joint_training_token_budget

                num_tokens += joint_training_token_budget(int(text_ids.numel()), frames, 368, 640)
            else:
                num_tokens += count
        if "sample_id" in batch:
            sample_ids.append(str(batch["sample_id"][index]))

    if not 0 < num_tokens < cap:
        raise ValueError(f"packed token count must be in (0,{cap}), got {num_tokens}")

    return {
        "num_samples": num_samples,
        "num_tokens": num_tokens,
        "frame_counts": frame_counts,
        "sample_ids": sample_ids,
    }


def main() -> None:
    import torch.distributed as dist

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.context_managers import data_loader_init, distributed_init
    from cosmos_framework.utils.lazy_config import instantiate

    from . import config as _config  # noqa: F401

    args = parse_args()
    if args.batches < 1:
        raise ValueError("--batches must be positive")
    with distributed_init():
        distributed.init()
    config = load_experiment_from_toml(args.toml)
    config.validate()
    config.freeze()
    with data_loader_init():
        dataloader = instantiate(config.dataloader_train)

    iterator = iter(dataloader)
    local = [
        audit_batch(
            next(iterator), int(config.dataloader_train.max_sequence_length),
            action_tokens_per_latent=int(config.model.config.action_tokens_per_latent),
        )
        for _ in range(args.batches)
    ]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, {"rank": dist.get_rank(), "batches": local})
    if dist.get_rank() == 0:
        result = {"status": "success", "world_size": dist.get_world_size(), "ranks": gathered}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("EGOVERSE_DATALOADER_AUDIT=" + json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
