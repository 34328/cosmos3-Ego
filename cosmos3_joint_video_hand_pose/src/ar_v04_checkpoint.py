"""Explicit visual-format guard for official DCP lifecycle and offline loading."""
from __future__ import annotations

import json
from pathlib import Path

from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils import distributed

from .ar_v04_codec import VIDEO_LATENT_FORMAT

FORMAT_FILE = "ar_v04_video_format.json"
OFFICIAL_NANO = Path("/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp")


def checkpoint_directory(path):
    path = Path(path).resolve()
    return path.parent if path.name == "model" else path


def validate_video_checkpoint(path, *, allow_official_nano=False):
    """Reject old joint weights, even though their tensor shapes are compatible."""
    root = checkpoint_directory(path)
    if allow_official_nano and root == OFFICIAL_NANO.resolve():
        return {"initialization": "official_nano"}
    marker = root / FORMAT_FILE
    if not marker.is_file():
        raise ValueError(f"checkpoint lacks V0.4 continuous-video format receipt: {marker}")
    receipt = json.loads(marker.read_text())
    if (receipt.get("video_latent_format") != VIDEO_LATENT_FORMAT
            or receipt.get("model_version") != "ar_v0.4"):
        raise ValueError("checkpoint video format is not V0.4; old joint DCP is incompatible")
    return receipt


class ARV04VideoFormatCallback(Callback):
    def on_load_checkpoint_end(self, model, iteration=0, checkpoint_path=None):
        # Official loader resolves same-job resume / warm start and supplies the
        # actual source; validate before any optimizer step, not a guessed path.
        if checkpoint_path is None:
            raise ValueError("V0.4 must initialize from official Nano or a verified V0.4 checkpoint")
        validate_video_checkpoint(checkpoint_path, allow_official_nano=True)

    def on_save_checkpoint_success(self, iteration=0, elapsed_time=0):
        # Only a successfully completed official DCP gets a format receipt.
        if distributed.is_rank0():
            root = Path(self.config.job.path_local) / "checkpoints" / f"iter_{iteration:09d}"
            if not (root / "model" / ".metadata").is_file():
                raise ValueError("completed V0.4 checkpoint is missing its model metadata")
            receipt = dict(model_version="ar_v0.4", video_latent_format=VIDEO_LATENT_FORMAT,
                           step=int(iteration), encoding="continuous_full_clip",
                           boundary_condition="gather_previous_latent")
            target = root / FORMAT_FILE
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(target)
