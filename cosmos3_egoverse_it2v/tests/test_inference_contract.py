from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from cosmos_framework.inference.args import ModelMode
from cosmos3_egoverse_it2v.src.inference import get_egoverse_it2v_sample_data


def test_exact_canvas_condition_and_prompt(tmp_path: Path):
    image = np.arange(360 * 640 * 3, dtype=np.uint8).reshape(360, 640, 3)
    path = tmp_path / "first.png"
    Image.fromarray(image).save(path)
    args = SimpleNamespace(
        model_mode=ModelMode.IMAGE2VIDEO,
        vision_path=path,
        num_frames=9,
        fps=30,
        prompt='{"actions":[{"description":"test"}],"duration":"0.30s"}',
    )
    model = SimpleNamespace(input_video_key="video", input_caption_key="ai_caption")
    batch = get_egoverse_it2v_sample_data(args, model, device="cpu")
    assert batch["video"][0].shape == (1, 3, 9, 368, 640)
    assert batch["image_size"][0].tolist() == [[368, 640, 368, 640]]
    assert batch["sequence_plan"][0].condition_frame_indexes_vision == [0]
    assert batch["ai_caption"] == [args.prompt]
    assert torch.equal(batch["video"][0][:, :, 0], batch["video"][0][:, :, -1])
