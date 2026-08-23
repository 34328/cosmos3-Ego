"""EgoVerse IT2V inference with the exact 640x368 training canvas."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.args import ModelMode
from cosmos_framework.inference.vision import read_media_frames

from . import config as _config  # noqa: F401


def get_egoverse_it2v_sample_data(sample_args: Any, model: Any, *, device: Any = "cuda") -> dict[str, Any]:
    """Build the same first-frame-conditioned video contract used in training."""
    if sample_args.model_mode is not ModelMode.IMAGE2VIDEO:
        raise ValueError("EgoVerse IT2V inference only supports image2video")
    if sample_args.vision_path is None:
        raise ValueError("image2video requires vision_path")

    first_frame, _ = read_media_frames(Path(sample_args.vision_path), max_frames=1)
    if tuple(first_frame.shape) != (3, 1, 360, 640):
        raise ValueError(f"expected first frame [3,1,360,640], got {tuple(first_frame.shape)}")
    if (sample_args.num_frames - 1) % 4:
        raise ValueError(f"num_frames must satisfy T=1+4n, got {sample_args.num_frames}")

    video = F.pad(first_frame, (0, 0, 0, 8), mode="reflect")
    video = (video.to(dtype=torch.bfloat16) / 127.5 - 1.0).repeat(1, sample_args.num_frames, 1, 1)
    video = video.unsqueeze(0).to(device=device)
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=False,
        condition_frame_indexes_vision=[0],
    )
    return {
        model.input_video_key: [video],
        "image_size": [torch.tensor([[368, 640, 368, 640]], dtype=torch.float32, device=device)],
        "fps": torch.tensor([float(sample_args.fps)], device=device),
        "conditioning_fps": torch.tensor([float(sample_args.fps)], device=device),
        "num_frames": torch.tensor([sample_args.num_frames], device=device),
        "is_preprocessed": True,
        "sequence_plan": [plan],
        # The prompt is already the exact structured JSON produced by the
        # training formatter; do not let generic inference rewrite duration.
        model.input_caption_key: [sample_args.prompt],
    }


def main() -> None:
    import cosmos_framework.inference.inference as inference_module
    from cosmos_framework.scripts.inference import main as cosmos_inference_main

    inference_module.get_sample_data = get_egoverse_it2v_sample_data
    if "--use-ema-weights" not in sys.argv and "--no-use-ema-weights" not in sys.argv:
        sys.argv.append("--no-use-ema-weights")
    cosmos_inference_main()


if __name__ == "__main__":
    main()
