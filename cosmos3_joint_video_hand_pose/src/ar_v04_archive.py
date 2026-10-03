"""Continuous-video identity checks around the unchanged action/RGB container."""

from . import ar_v02_eval as shared
from .ar_v04_codec import VIDEO_LATENT_FORMAT, require_video_format


def validate_metadata(metadata):
    require_video_format(metadata.get("video_latent_format"))
    if metadata.get("model_version") != "ar_v0.4" or metadata.get("video_decode_mode") != "continuous_full_sequence":
        raise ValueError("V0.4 archive requires its model identity and continuous full-sequence decode")


def save_rollout(path, **kwargs):
    validate_metadata(kwargs["metadata"])
    # The shared container stores decoded RGB, never reset/continuous latent weights.
    return shared.save_rollout(path, **kwargs)


def load_rollout(path):
    result = shared.load_rollout(path)
    validate_metadata(result[1])
    return result


def evaluate_archive(path, args):
    load_rollout(path)
    return shared.evaluate_archive(path, args)


def overlay_archive(path, output, args):
    load_rollout(path)
    return shared.overlay_archive(path, output, args)
