"""Parallel teacher forcing over per-sample [history, prediction] streams."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import ChunkCausalAttention, causal_video_visibility, chunk_ids


def parallel_tf_visibility(q_frames, k_frames, q_prediction, k_prediction, *,
                           frames_per_chunk=4, local_attention_frames=32):
    """H→H same/past; P→H strictly past; P→P same chunk; H never sees P."""
    q_chunk = chunk_ids(q_frames, frames_per_chunk)
    k_chunk = chunk_ids(k_frames, frames_per_chunk)
    local = causal_video_visibility(q_frames, k_frames,
        frames_per_chunk=frames_per_chunk, local_attention_frames=local_attention_frames)
    history = ~k_prediction & (~q_prediction | (k_chunk < q_chunk))
    prediction = q_prediction & k_prediction & (q_chunk == k_chunk)
    return local & (history | prediction)


class ParallelTeacherForcingAttention(ChunkCausalAttention):
    def __init__(self, vision_shapes, text_lengths, *, device,
                 frames_per_chunk=4, local_attention_frames=32):
        # Keep the inherited run-builder's frame IDs unique between H and P.
        # Physical frame IDs below reset for P; absolute RoPE also repeats H's grid.
        super().__init__([(2*t, h, w) for t, h, w in vision_shapes], text_lengths,
            device=device, frames_per_chunk=frames_per_chunk,
            local_attention_frames=local_attention_frames)
        frames, streams = [], []
        for t, h, w in vision_shapes:
            frames.append(torch.arange(t, device=device).repeat(2).repeat_interleave(h*w))
            streams.append(torch.arange(2, device=device).repeat_interleave(t*h*w).bool())
        padding = self.gen_pad-self.flat_gen_tokens
        self.physical_frames = F.pad(torch.cat(frames), (0, padding), value=-1)
        self.prediction = F.pad(torch.cat(streams), (0, padding), value=False)

    def mask_mod(self, text_pad):
        text_samples = F.pad(self.text_samples, (0, text_pad-len(self.text_samples)), value=-2)
        samples, frames, prediction = self.samples, self.physical_frames, self.prediction
        C, W = self.frames_per_chunk, self.local_attention_frames

        def mask(b, h, q, k):
            qi = q.clamp(max=self.gen_pad-1)
            ki = (k-text_pad).clamp(0, self.gen_pad-1)
            text_i = k.clamp(0, text_pad-1)
            text_ok = (k < text_pad) & (samples[qi] == text_samples[text_i])
            video_ok = (k >= text_pad) & (samples[qi] == samples[ki]) & (samples[ki] >= 0)
            video_ok = video_ok & parallel_tf_visibility(
                frames[qi], frames[ki], prediction[qi], prediction[ki],
                frames_per_chunk=C, local_attention_frames=W)
            return (q < self.flat_gen_tokens) & (text_ok | video_ok)

        return mask
