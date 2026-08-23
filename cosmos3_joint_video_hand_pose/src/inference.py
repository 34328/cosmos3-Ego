"""EgoVerse WAM inference entrypoint backed by Cosmos' native inference pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
    pad_action_to_max_dim,
)
from cosmos_framework.data.generator.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.inference.args import ModelMode
from cosmos_framework.inference.vision import read_media_frames

from . import config as _config  # noqa: F401
from .dataset import CosmosActionPromptFormatter
from .temporal import CANVAS_HEIGHT, CANVAS_WIDTH


def build_egoverse_wam_batch(
    *,
    video: torch.Tensor,
    initial_action: torch.Tensor,
    prompt: str,
    frames: int,
    fps: float,
    max_action_dim: int = 64,
    input_video_key: str = "video",
    batch_size: int = 1,
    device: Any = "cuda",
) -> dict[str, Any]:
    """Build the same T-action initial-state WAM layout used during training."""
    if tuple(video.shape) != (3, 1, 360, 640):
        raise ValueError(f"conditioning image must have shape [3,1,360,640], got {tuple(video.shape)}")
    if frames < 1 or (frames - 1) % 4:
        raise ValueError(f"frames must satisfy T=1+4n, got {frames}")
    initial_action = torch.as_tensor(initial_action, dtype=torch.float32).reshape(-1)
    if initial_action.shape != (57,):
        raise ValueError(f"initial_action must have shape [57], got {tuple(initial_action.shape)}")

    video = F.pad(video, (0, 0, 0, 8), mode="reflect")
    video = video.repeat(1, frames, 1, 1)
    action_raw = torch.zeros((frames, 57), dtype=torch.float32)
    action_raw[0] = initial_action
    action = pad_action_to_max_dim(action_raw, max_action_dim)
    sequence_plan = build_sequence_plan_from_mode(
        mode="wam",
        video_length=frames,
        action_length=frames,
        has_text=True,
    )
    if sequence_plan.condition_frame_indexes_action != [0]:
        raise AssertionError("EgoVerse WAM inference did not preserve clean action state a0")

    caption = CosmosActionPromptFormatter()(prompt.strip(), frames, float(fps))
    record = ActionProcessingRecord(raw_action_dim=57, action_normalizer=None)
    # The model removes latent padding according to image_size[2:]. Reporting
    # 360 here floors the decoded height to 352 because the VAE factor is 16.
    # Preserve the full padded canvas and crop to 360 only after VAE decode.
    image_size = torch.tensor(
        [CANVAS_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_WIDTH], dtype=torch.float32
    )
    return {
        input_video_key: [[video]] * batch_size,
        "action": [[action]] * batch_size,
        **make_batched_action_processing_fields(record, batch_size),
        "mode": ["wam"] * batch_size,
        "ai_caption": [caption] * batch_size,
        "prompt": [prompt] * batch_size,
        "conditioning_fps": [torch.tensor(float(fps))] * batch_size,
        "image_size": image_size.unsqueeze(0).to(device=device),
        "domain_id": [torch.tensor(3, dtype=torch.long)] * batch_size,
        "sequence_plan": [sequence_plan] * batch_size,
    }


def get_egoverse_action_sample_data(
    model_config: Any,
    *,
    batch_size: int,
    prompt: str,
    vision_path: Path,
    model_mode: ModelMode,
    action_path: Path | None,
    domain_name: str,
    view_point: str,
    resolution: str,
    action_chunk_size: int,
    max_action_dim: int,
    fps: int,
    device: Any,
) -> dict[str, Any]:
    del resolution
    if model_mode is not ModelMode.WAM:
        raise ValueError("EgoVerse inference entrypoint only supports WAM policy mode")
    if domain_name.strip().lower() != "hand_pose":
        raise ValueError(f"EgoVerse inference requires domain_name='hand_pose', got {domain_name!r}")
    if view_point != "ego_view":
        raise ValueError(f"EgoVerse inference requires view_point='ego_view', got {view_point!r}")
    if action_path is None:
        raise ValueError("EgoVerse WAM inference requires --action-path containing initial_action57")
    raw_action = torch.tensor(json.loads(Path(action_path).read_text(encoding="utf-8")), dtype=torch.float32)
    if raw_action.shape != (57,):
        raise ValueError(f"initial action file must contain exactly [57], got {tuple(raw_action.shape)}")
    video, _ = read_media_frames(Path(vision_path), max_frames=1)
    return build_egoverse_wam_batch(
        video=video,
        initial_action=raw_action,
        prompt=prompt,
        frames=action_chunk_size,
        fps=float(fps),
        max_action_dim=max_action_dim,
        input_video_key=model_config.input_video_key,
        batch_size=batch_size,
        device=device,
    )


def _default_to_regular_weights(argv: list[str]) -> None:
    """Make project inference use the directly optimized ``net.*`` weights by default."""
    if "--use-ema-weights" not in argv and "--no-use-ema-weights" not in argv:
        argv.append("--no-use-ema-weights")


def main() -> None:
    import cosmos_framework.inference.action as action_inference
    from cosmos_framework.scripts.inference import main as cosmos_inference_main

    _default_to_regular_weights(sys.argv)
    action_inference.get_action_sample_data = get_egoverse_action_sample_data
    cosmos_inference_main()


if __name__ == "__main__":
    main()
