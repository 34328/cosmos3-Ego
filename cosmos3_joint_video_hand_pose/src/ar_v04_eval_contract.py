"""Require a V0.4 checkpoint and its actual training recipe before video decode."""

from pathlib import Path
import re

from .ar_v031_eval_contract import (
    PREFIX_LOSS_DENOMINATOR, PREFIX_LOSS_MASK_SCOPE,
    sha256, validate_inference_config,
)
from .ar_v04_codec import require_video_format

MODEL_TARGET = "cosmos3_joint_video_hand_pose.src.ar_v04_model.EgoVerseARV04Model"
MODEL_VERSION = "ar_v0.4"


def validate_checkpoint_snapshot(snapshot, checkpoint):
    import yaml
    from .ar_v04_checkpoint import validate_video_checkpoint
    from cosmos3_joint_video_hand_pose.scripts.ar_v03_eval_followup import checkpoint_ready

    snapshot, checkpoint = Path(snapshot).resolve(), Path(checkpoint).resolve()
    match = re.fullmatch(r"iter_(\d{9})", checkpoint.parent.name)
    if (snapshot.name != "config.yaml" or not snapshot.is_file()
            or checkpoint.name != "model" or not match
            or checkpoint.parent.parent != snapshot.parent / "checkpoints"):
        raise ValueError("V0.4 requires this actual run config.yaml and model DCP")
    config = yaml.safe_load(snapshot.read_text())
    if config["model"]["_target_"] != MODEL_TARGET or config["model"]["config"].get("mask_prefix_loss") is not True:
        raise ValueError("V0.4 inference requires its continuous-VAE model with V0.3.1 loss recipe")
    require_video_format(config["model"]["config"].get("video_latent_format"))
    validate_video_checkpoint(checkpoint)
    ready = checkpoint_ready(snapshot.parent, int(match.group(1)))
    if not ready.get("ready"):
        raise ValueError("official V0.4 save is incomplete: " + repr(ready))
    return config, dict(training_snapshot_sha256=sha256(snapshot),
                        checkpoint_metadata_sha256=sha256(checkpoint / ".metadata"),
                        checkpoint_readiness=ready)
