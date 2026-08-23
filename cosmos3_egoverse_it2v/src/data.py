from __future__ import annotations

import csv
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import zarr

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset
from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.augmentors.text_tokenizer import TextTokenizerTransform
from cosmos_framework.data.generator.sequence_packing import SequencePlan

H, W = 368, 640
MAX_TOKENS = 85_000
RETENTION = (1.0, 0.8, 0.7, 0.6, 0.5)


def _prompt_text(segment: str, task: str) -> str:
    segment, task = segment.strip(), task.strip()
    if segment and task:
        return f"Overall task: {task} Current segment: {segment}"
    if segment or task:
        return segment or task
    raise ValueError("empty segment and episode prompt")


def _unwrap(value) -> bytes:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return bytes(value)


def _decode(encoded) -> torch.Tensor:
    frames = [
        torch.from_numpy(np.asarray(Image.open(BytesIO(_unwrap(x))).convert("RGB"), dtype=np.uint8).copy())
        for x in encoded
    ]
    video = torch.stack(frames).permute(3, 0, 1, 2).contiguous()
    if tuple(video.shape[-2:]) != (360, 640):
        raise ValueError(f"expected 640x360 source, got {tuple(video.shape[-2:])}")
    return F.pad(video, (0, 0, 0, 8), mode="reflect")


def _aligned(frames: int) -> int:
    return frames - ((frames - 1) % 4)


def _indices(source: int, target: int) -> np.ndarray:
    aligned = _aligned(source)
    if target == aligned:
        return np.arange(aligned, dtype=np.int64)
    future = np.rint(np.linspace(1, aligned - 1, target - 1)).astype(np.int64)
    if np.unique(future).size != future.size:
        raise ValueError("duplicate temporal sample index")
    return np.concatenate(([0], future)).astype(np.int64)


def _prompt(caption: str, frames: int, fps: float) -> dict:
    formatted = ActionPromptJsonFormatter(float_seconds=True)({
        "ai_caption": caption,
        "viewpoint": "ego_view",
        "video": torch.empty((3, frames, H, W), device="meta"),
        "action": torch.empty((frames, 0), device="meta"),
        "conditioning_fps": float(fps),
        "image_size": torch.tensor([H, W, H, W]),
        "mode": "wam",
    })["ai_caption"]
    if not isinstance(formatted, dict):
        raise TypeError("Cosmos action JSON formatter returned non-dict prompt")
    return formatted


class EgoVerseVideoDataset(Dataset):
    def __init__(
        self,
        episodes_manifest: str,
        segments_manifest: str,
        *,
        tokenizer_config: dict,
        cfg_dropout_rate: float,
        max_tokens: int = MAX_TOKENS,
    ):
        with Path(episodes_manifest).open(newline="", encoding="utf-8") as f:
            episodes = {r["episode_hash"]: r for r in csv.DictReader(f) if r["split"] == "train"}
        with Path(segments_manifest).open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "train" and r["episode_hash"] in episodes]
        if not rows:
            raise ValueError("no train segments")
        self.episodes = episodes
        self.rows = []
        self._tokenizer = TextTokenizerTransform(
            input_keys=["ai_caption"], output_keys=["text_token_ids"],
            args={"tokenizer_config": tokenizer_config, "cfg_dropout_rate": cfg_dropout_rate},
        )
        counter = TextTokenizerTransform(
            input_keys=["ai_caption"], output_keys=["text_token_ids"],
            args={"tokenizer_config": tokenizer_config, "cfg_dropout_rate": 0.0},
        )
        for row in rows:
            source = int(row["end_idx"]) - int(row["start_idx"])
            aligned = _aligned(source)
            if aligned <= 1:
                continue
            for ratio in RETENTION:
                frames = aligned if ratio == 1.0 else 1 + 4 * int(ratio * (aligned - 1) // 4)
                fps = float(episodes[row["episode_hash"]]["fps"]) * (frames - 1) / (aligned - 1)
                caption = _prompt_text(row["text_normalized"], episodes[row["episode_hash"]]["task_description"])
                prompt = _prompt(caption, frames, fps)
                text_tokens = len(counter({"ai_caption": prompt})["text_token_ids"])
                tokens = text_tokens + 1 + 240 * (1 + (frames - 1) // 4) + 2 + frames
                if tokens < max_tokens:
                    row = dict(row)
                    row.update(_frames=frames, _fps=fps, _prompt=prompt)
                    self.rows.append(row)
                    break
        if not self.rows:
            raise ValueError("all segments exceed the video token cap")

    def __len__(self):
        return len(self.rows)

    def get_shuffle_blocks(self):
        return [(i, 1) for i in range(len(self))]

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        episode = self.episodes[row["episode_hash"]]
        start = int(row["start_idx"])
        source = int(row["end_idx"]) - start
        indexes = start + _indices(source, int(row["_frames"]))
        group = zarr.open_group(episode["abs_zarr_path"], mode="r")
        video = _decode(group["images.front_1"][indexes])
        sample = {
            "ai_caption": row["_prompt"],
            "video": video,
            "conditioning_fps": float(row["_fps"]),
            "image_size": torch.tensor([H, W, H, W], dtype=torch.float32),
            "sequence_plan": SequencePlan(has_text=True, has_vision=True, has_action=False, condition_frame_indexes_vision=[0]),
        }
        return self._tokenizer(sample)


def get_egoverse_it2v_dataset(
    *, episodes_manifest: str, segments_manifest: str, tokenizer_config: dict,
    cfg_dropout_rate: float = 0.1, iterable_shuffle: bool = True, seed: int = 42,
    max_sequence_length: int = MAX_TOKENS, prompt_mode: str = "episode_context_and_segment",
):
    if prompt_mode != "episode_context_and_segment":
        raise ValueError("IT2V uses the episode_context_and_segment prompt contract")
    dataset = EgoVerseVideoDataset(
        episodes_manifest,
        segments_manifest,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_tokens=max_sequence_length,
    )
    return ActionIterableShuffleDataset(dataset, seed=seed) if iterable_shuffle else dataset
