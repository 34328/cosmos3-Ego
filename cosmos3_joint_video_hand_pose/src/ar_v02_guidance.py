"""Video-only text CFG using the existing joint sampler and two isolated KV caches."""

import copy
import math

import torch

from .ar_v02_cache import JointKVCache
from .ar_v02_inference import JointARSampler


def video_guidance_output(conditional, unconditional, scale):
    """Guide only video velocities; action uses the conditional forward unchanged."""
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("video guidance must be finite and >= 1")
    if scale == 1:
        return conditional
    result = dict(conditional)
    result["preds_vision"] = [
        u.float() + scale * (c.float() - u.float())
        for c, u in zip(conditional["preds_vision"], unconditional["preds_vision"], strict=True)
    ]
    return result


class VideoGuidedJointARSampler(JointARSampler):
    def __init__(self, *args, video_guidance, negative_text_ids, **kwargs):
        if not math.isfinite(video_guidance) or video_guidance <= 1:
            raise ValueError("guided sampler requires video guidance > 1")
        super().__init__(*args, **kwargs)
        self.video_guidance = float(video_guidance)
        self.negative_text_ids = list(negative_text_ids)
        self._negative = None

    @torch.no_grad()
    def _cache_forward(self, video, action, indexes, *, chunk, phase, **kwargs):
        conditional = super()._cache_forward(video, action, indexes, chunk=chunk, phase=phase, **kwargs)
        if phase == "text":
            # Share read-only model weights and inputs, never cache buffers or text packs.
            negative = copy.copy(self)
            negative.text = [self.negative_text_ids]
            net = self.model.net
            negative.cache = JointKVCache(
                self.layout, num_layers=net.num_hidden_layers, num_kv_heads=net.num_kv_heads,
                head_dim=net.head_dim, device=video.device, dtype=self.model.tensor_kwargs["dtype"],
            )
            negative._cache_phase = None
            with self.model.ar_context(self.chunk_size, 15):
                negative._cache_template = self.model._pack_input_sequence(
                    self.plans, negative.text, self.gen,
                    torch.zeros(1, self.layout.num_video_frames),
                    initial_mrope_temporal_offset=self.memory_info["initial_temporal_offset"],
                )
            self._negative = negative
        unconditional = JointARSampler._cache_forward(
            self._negative, video, action, indexes, chunk=chunk, phase=phase, **kwargs
        )
        if phase == "noisy":
            return video_guidance_output(conditional, unconditional, self.video_guidance)
        return conditional

    @torch.no_grad()
    def sample(self, **kwargs):
        if not kwargs.get("use_cache", True) or kwargs.get("verify_cache", False):
            raise ValueError("video CFG currently requires cached inference without reference verification")
        self._negative = None
        result = super().sample(**kwargs)
        for report in self.chunk_reports:
            report["conditional_forward_calls"] = report["forward_calls"]
            report["unconditional_forward_calls"] = 32
            report["forward_calls"] += 32
            report["video_guidance"] = self.video_guidance
            report["action_guidance"] = 1.0
        # Release the second cache before the next history mode/window.
        self._negative = None
        return result
