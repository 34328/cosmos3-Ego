"""V0.4 continuous video VAE semantics, retaining the V0.3.1 joint objective."""
from __future__ import annotations

import attrs
import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

from .ar_v02_layout import JointChunkLayout
from .ar_v031_model import EgoVerseARV031Model, EgoVerseARV031ModelConfig
from .ar_v04_codec import VIDEO_LATENT_FORMAT, gather_joint_latents


@attrs.define(slots=False)
class EgoVerseARV04ModelConfig(EgoVerseARV031ModelConfig):
    video_latent_format: str = VIDEO_LATENT_FORMAT


class EgoVerseARV04Model(EgoVerseARV031Model):
    def __init__(self, config, **kwargs):
        if config.video_latent_format != VIDEO_LATENT_FORMAT:
            raise ValueError("V0.4 requires continuous video VAE latent semantics")
        super().__init__(config, **kwargs)

    @torch.no_grad()
    def _encode_vision_x0_tokens(
        self, raw_state_vision, num_vision_items_per_sample,
        vision_condition_indexes, num_views_per_vision_item=None,
        balance_vae_encode=False,
    ):
        if self._ar_step is None:
            raise ValueError("choose C before VAE encoding")
        if num_vision_items_per_sample is not None or num_views_per_vision_item is not None:
            raise ValueError("joint chunks expect one monocular RGB stream per sample")
        for raw in raw_state_vision:
            if raw.ndim != 5 or raw.shape[2] < 5 or (raw.shape[2] - 1) % 4:
                raise ValueError("RGB clip must contain 1+4N sampled frames with future video")
        # Official whole-item normalization/encoding and optional load balancing.
        # No prefix optimization: later U rows refer to the same continuous VAE
        # sequence, rather than independently re-encoding RGB boundary images.
        continuous = OmniMoTModel._encode_vision_x0_tokens(
            self, raw_state_vision, None, None,
            balance_vae_encode=balance_vae_encode,
        )
        self._joint_original_frames = []
        result = []
        for raw, latent in zip(raw_state_vision, continuous, strict=True):
            frames = 1 + (raw.shape[2] - 1) // 4
            if latent.shape[2] != frames:
                raise ValueError("continuous causal VAE returned unexpected clip length")
            self._joint_original_frames.append(frames)
            layout = JointChunkLayout(frames, 1, self._ar_step.chunk_size)
            result.append(gather_joint_latents(layout, latent))
        return result
