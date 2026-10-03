"""V0.4 joint sampler: preserve video latent history across chunk boundaries."""

import torch

from .ar_v03_inference import DiffusionForcingJointARSampler
from .ar_v04_codec import VIDEO_LATENT_FORMAT, continuous_video_latents, require_video_format


class ContinuousJointARSampler(DiffusionForcingJointARSampler):
    """Retain action, schedules and joint KV refresh; remove boundary VAE cycling."""

    def __init__(self, model, *args, **kwargs):
        require_video_format(getattr(model.config, "video_latent_format", None))
        super().__init__(model, *args, **kwargs)

    @torch.no_grad()
    def _next_condition_video(self, block):
        if block.ndim != 5 or block.shape[2] < 2 or not torch.isfinite(block).all():
            raise ValueError("expected a finite completed [U,V] video block")
        return block[:, :, -1:].clone()

    @torch.no_grad()
    def sample(self, **kwargs):
        video, action = super().sample(**kwargs)
        continuous_video_latents(self.layout, video, video_latent_format=VIDEO_LATENT_FORMAT,
                                 history=kwargs.get("history", "gt"))
        for report in self.chunk_reports:
            report["video_latent_format"] = VIDEO_LATENT_FORMAT
            report["video_boundary_condition"] = "continuous_latent_alias"
            report["video_condition_origin"] = (
                "previous_generated_latent" if report["condition_source"] == "prediction" else "gt_continuous_latent")
        return video, action
