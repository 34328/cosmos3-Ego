from __future__ import annotations

import csv
from io import BytesIO
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import zarr

from .action import Action57Builder
from .temporal import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    MAX_SEQUENCE_LENGTH,
    cosmos_wam_token_count,
    retained_fps,
    retention_candidates,
    uniform_frame_indices,
)


PROMPT_MODE_SEGMENT_ONLY = "segment_only"
PROMPT_MODE_EPISODE_CONTEXT_AND_SEGMENT = "episode_context_and_segment"
PROMPT_MODES = (
    PROMPT_MODE_SEGMENT_ONLY,
    PROMPT_MODE_EPISODE_CONTEXT_AND_SEGMENT,
)


def build_prompt_text(
    segment_text: str,
    task_description: str,
    *,
    mode: str = PROMPT_MODE_SEGMENT_ONLY,
) -> str:
    """Build the natural-language description inserted into Cosmos action JSON."""
    segment_text = segment_text.strip()
    task_description = task_description.strip()
    if mode == PROMPT_MODE_SEGMENT_ONLY:
        prompt = segment_text or task_description
    elif mode == PROMPT_MODE_EPISODE_CONTEXT_AND_SEGMENT:
        if segment_text and task_description:
            prompt = f"Overall task: {task_description} Current segment: {segment_text}"
        else:
            prompt = segment_text or task_description
    else:
        raise ValueError(f"unsupported prompt mode {mode!r}; expected one of {PROMPT_MODES}")
    if not prompt:
        raise ValueError("both segment text and episode task description are empty")
    return prompt


def _unwrap_jpeg(value) -> bytes:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected nested scalar JPEG bytes, got {type(value).__name__}")
    return bytes(value)


def decode_rgb_video(encoded_frames) -> torch.Tensor:
    frames = []
    for encoded in encoded_frames:
        with Image.open(BytesIO(_unwrap_jpeg(encoded))) as image:
            frames.append(torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()))
    video = torch.stack(frames).permute(3, 0, 1, 2).contiguous()
    if tuple(video.shape[-2:]) != (360, 640):
        raise ValueError(f"expected native 640x360 RGB, got {tuple(video.shape[-2:][::-1])}")
    return F.pad(video, (0, 0, 0, 8), mode="reflect")


class CosmosTextTokenCounter:
    def __init__(self, tokenizer_path: str = "/mnt/checkpoints/Cosmos3-Nano/text_tokenizer"):
        from transformers import Qwen2Tokenizer
        from cosmos_framework.data.generator.sequence_packing.modality import add_special_tokens

        self._tokenizer, _ = add_special_tokens(Qwen2Tokenizer.from_pretrained(tokenizer_path))

    def __call__(self, caption: str) -> int:
        from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption

        # Match Cosmos' TextTokenizerTransform exactly. With no system prompt,
        # the is_video flag is intentionally irrelevant, but keeping this
        # explicit prevents the estimator from drifting if that default changes.
        return len(tokenize_caption(caption, self._tokenizer, is_video=False, use_system_prompt=False))


class CosmosActionPromptFormatter:
    """Build the exact official action JSON string consumed by the tokenizer."""

    def __init__(self) -> None:
        from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter

        self._formatter = ActionPromptJsonFormatter(float_seconds=True)

    def __call__(self, caption: str, frames: int, fps: float) -> str:
        sample = {
            "ai_caption": caption,
            "viewpoint": "ego_view",
            "video": torch.empty((3, frames, CANVAS_HEIGHT, CANVAS_WIDTH), device="meta"),
            "action": torch.empty((frames, 0), device="meta"),
            "conditioning_fps": float(fps),
            # Cosmos crops encoded latents to image_size[2:] using integer
            # division by the VAE factor. Keep the complete padded canvas here;
            # 360 would floor to 352 and discard eight source-image rows.
            "image_size": torch.tensor([CANVAS_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_WIDTH]),
            "mode": "wam",
        }
        structured = self._formatter(sample)["ai_caption"]
        if not isinstance(structured, dict):
            raise TypeError("official action prompt formatter did not return a JSON-compatible dictionary")
        return json.dumps(structured)


class EgoVerseSegmentDataset(Dataset):
    """Map-style manifest dataset producing unpadded 57D Cosmos WAM samples."""

    def __init__(
        self,
        episodes_manifest: str | Path,
        segments_manifest: str | Path,
        *,
        split: str = "train",
        token_counter: Callable[[str], int] | None = None,
        prompt_formatter: Callable[[str, int, float], str] | None = None,
        action_builder: Action57Builder | None = None,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        prompt_mode: str = PROMPT_MODE_SEGMENT_ONLY,
    ):
        with Path(episodes_manifest).open(newline="", encoding="utf-8") as handle:
            episodes = {row["episode_hash"]: row for row in csv.DictReader(handle) if row["split"] == split}
        with Path(segments_manifest).open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split and row["episode_hash"] in episodes]
        if not rows:
            raise ValueError(f"no {split!r} segments found")
        self.episodes = episodes
        self.token_counter = token_counter or CosmosTextTokenCounter()
        self.prompt_formatter = prompt_formatter or CosmosActionPromptFormatter()
        self.action_builder = action_builder or Action57Builder()
        self.max_sequence_length = int(max_sequence_length)
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"unsupported prompt mode {prompt_mode!r}; expected one of {PROMPT_MODES}")
        self.prompt_mode = prompt_mode
        self.retention_counts = {"100%": 0, "80%": 0, "70%": 0, "60%": 0, "50%": 0, "dropped": 0}
        self.dropped_segments = []
        self.rows = []
        for row in rows:
            plan = self._build_temporal_plan(row, episodes[row["episode_hash"]])
            if plan is None:
                self.retention_counts["dropped"] += 1
                self.dropped_segments.append(
                    {
                        "sample_id": (
                            f"{row['episode_hash']}:{row['span_index']}:{row['start_idx']}:{row['end_idx']}"
                        ),
                        "source_frames": int(row["end_idx"]) - int(row["start_idx"]),
                        "reason": "no_legal_retention_tier_under_cap",
                    }
                )
                continue
            row.update(plan)
            self.rows.append(row)
            self.retention_counts[row["_retention_label"]] += 1
        if not self.rows:
            raise ValueError(f"all {split!r} segments exceed the 50% retention token budget")

    def _build_temporal_plan(self, row: dict, episode: dict) -> dict | None:
        caption = build_prompt_text(
            row["text_normalized"],
            episode["task_description"],
            mode=self.prompt_mode,
        )
        source_frames = int(row["end_idx"]) - int(row["start_idx"])
        source_fps = float(episode["fps"])
        candidates = retention_candidates(source_frames)
        if not candidates:
            return None
        aligned_frames = candidates[0][1]
        for ratio, frames in candidates:
            effective_fps = retained_fps(source_fps, aligned_frames, frames)
            prompt = self.prompt_formatter(caption, frames, effective_fps)
            text_tokens = self.token_counter(prompt)
            if cosmos_wam_token_count(text_tokens, frames) < self.max_sequence_length:
                return {
                    "_selected_frames": frames,
                    "_conditioning_fps": effective_fps,
                    "_structured_prompt": prompt,
                    "_retention_label": f"{int(round(ratio * 100))}%",
                }
        return None

    def __len__(self) -> int:
        return len(self.rows)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return [(index, 1) for index in range(len(self))]

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        episode = self.episodes[row["episode_hash"]]
        start, end = int(row["start_idx"]), int(row["end_idx"])
        source_frames = end - start
        fps = float(row["_conditioning_fps"])
        relative = uniform_frame_indices(source_frames, int(row["_selected_frames"]))
        frame_indexes = start + relative

        group = zarr.open_group(episode["abs_zarr_path"], mode="r")
        total_frames = int(group.attrs["total_frames"])
        if frame_indexes[-1] >= min(end, total_frames):
            raise IndexError("sampled index exceeds the manifest segment or episode")

        def read(name: str, dtype=np.float64):
            return np.asarray(group[name][frame_indexes], dtype=dtype)

        right_wrist = read("right.obs_wrist_pose")
        left_wrist = read("left.obs_wrist_pose")
        right_keypoints = read("right.obs_keypoints")
        left_keypoints = read("left.obs_keypoints")
        action = self.action_builder.build(
            head_pose=read("obs_head_pose"),
            right_wrist_pose=right_wrist,
            left_wrist_pose=left_wrist,
            right_keypoints=right_keypoints,
            left_keypoints=left_keypoints,
        )
        visibility = torch.from_numpy(
            np.stack(
                (
                    read("right.obs_palm_in_fov_front_1", np.uint8),
                    read("left.obs_palm_in_fov_front_1", np.uint8),
                ),
                axis=-1,
            ).astype(np.bool_)
        )
        video = decode_rgb_video(group["images.front_1"][frame_indexes])
        length = len(frame_indexes)
        structured_prompt = json.loads(row["_structured_prompt"])
        assert video.shape == (3, length, 368, 640)
        assert action.shape == (length, 57)
        assert visibility.shape == (length, 2)
        return {
            "ai_caption": structured_prompt,
            "video": video,
            "action": action,
            "hand_visibility": visibility,
            "conditioning_fps": fps,
            "mode": "wam",
            "domain_id": torch.tensor(3, dtype=torch.long),
            "viewpoint": "ego_view",
            "sample_id": f"{row['episode_hash']}:{row['span_index']}:{start}:{end}",
            "temporal_retention": row["_retention_label"],
            "source_frame_indices": torch.from_numpy(frame_indexes.copy()),
        }


class EgoVerseCosmosDataset(Dataset):
    """Apply Cosmos' native tokenizer/SequencePlan/action padding without resize."""

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def get_shuffle_blocks(self):
        return self.dataset.get_shuffle_blocks()

    def __getitem__(self, index: int) -> dict:
        sample = self.transform(self.dataset[index], resolution=None)
        # Preserve all 23 latent rows from the 368 px padded canvas. Output
        # visualization is responsible for cropping pixels back to 360 px.
        sample["image_size"] = torch.tensor([368, 640, 368, 640], dtype=torch.float32)
        if sample["action"].shape[-1] != 64 or int(sample["raw_action_dim"]) != 57:
            raise AssertionError("Cosmos action padding contract is not 57D -> 64D")
        return sample


def get_egoverse_cosmos_dataset(
    *,
    episodes_manifest: str,
    segments_manifest: str,
    tokenizer_config: dict,
    cfg_dropout_rate: float = 0.1,
    iterable_shuffle: bool = True,
    seed: int = 42,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    prompt_mode: str = PROMPT_MODE_SEGMENT_ONLY,
    state_normalizer: str | None = None,
    future_normalizer: str | None = None,
):
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

    action_builder_kwargs = {}
    if state_normalizer is not None:
        action_builder_kwargs["state_normalizer"] = state_normalizer
    if future_normalizer is not None:
        action_builder_kwargs["future_normalizer"] = future_normalizer
    raw = EgoVerseSegmentDataset(
        episodes_manifest,
        segments_manifest,
        action_builder=Action57Builder(**action_builder_kwargs),
        max_sequence_length=max_sequence_length,
        prompt_mode=prompt_mode,
    )
    transform = ActionTransformPipeline(
        pad_keys=[],
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=64,
        action_channel_masking=True,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        # The adapter has already applied Cosmos' official JSON formatter with
        # the real 640x368 canvas metadata. The generic resize stage would
        # otherwise describe its nearest 832x480 bucket in the prompt.
        format_prompt_as_json=False,
    )
    dataset = EgoVerseCosmosDataset(raw, transform)
    return ActionIterableShuffleDataset(dataset, seed=seed) if iterable_shuffle else dataset
