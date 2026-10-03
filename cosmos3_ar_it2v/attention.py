"""Pure-video [1,C,C,...] causal attention; no action or replay stream."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def chunk_ids(frames: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return torch.where(frames == 0, 0, 1 + (frames - 1).div(chunk_size, rounding_mode="floor"))


def causal_video_visibility(q_frames, k_frames, *, frames_per_chunk=4, local_attention_frames=16):
    """A block sees itself and preceding frames up to the fixed total window.

    No frame-zero sink: frame zero eventually rolls out exactly as other history.
    Partial final chunks retain the nominal block end so training/cache agree.
    """
    if local_attention_frames < frames_per_chunk:
        raise ValueError("local_attention_frames must include at least one complete chunk")
    end = chunk_ids(q_frames, frames_per_chunk) * frames_per_chunk + 1
    begin = (end - local_attention_frames).clamp_min(0)
    return (k_frames >= begin) & (k_frames < end)


class ChunkCausalAttention:
    def __init__(self, vision_shapes, text_lengths, *, device, frames_per_chunk=4, local_attention_frames=16):
        self.shapes = [tuple(map(int, shape)) for shape in vision_shapes]
        self.text_lengths = list(map(int, text_lengths))
        if len(self.shapes) != len(self.text_lengths) or not self.shapes:
            raise ValueError("one text length is required per video sample")
        self.frames_per_chunk = frames_per_chunk
        self.local_attention_frames = local_attention_frames
        self.flat_gen_tokens = sum(t*h*w for t,h,w in self.shapes)
        self.gen_pad = ((self.flat_gen_tokens + 127)//128)*128
        self.device = torch.device(device)
        frames = [torch.arange(t, device=device).repeat_interleave(h*w) for t,h,w in self.shapes]
        samples = [torch.full((t*h*w,), i, device=device, dtype=torch.long) for i,(t,h,w) in enumerate(self.shapes)]
        self.frames = F.pad(torch.cat(frames), (0, self.gen_pad-self.flat_gen_tokens), value=-1)
        self.samples = F.pad(torch.cat(samples), (0, self.gen_pad-self.flat_gen_tokens), value=-1)
        self.text_samples = torch.cat([torch.full((n,), i, device=device, dtype=torch.long) for i,n in enumerate(self.text_lengths)])
        self._masks = {}

    def mask_mod(self, text_pad):
        text_samples = F.pad(self.text_samples, (0, text_pad-len(self.text_samples)), value=-2)
        samples, frames = self.samples, self.frames
        C, W = self.frames_per_chunk, self.local_attention_frames
        def mask(b, h, q, k):
            qi = q.clamp(max=self.gen_pad-1)
            ki = (k-text_pad).clamp(0, self.gen_pad-1)
            text_i = k.clamp(0, text_pad-1)
            text_ok = (k < text_pad) & (samples[qi] == text_samples[text_i])
            video_ok = (k >= text_pad) & (samples[qi] == samples[ki]) & (samples[ki] >= 0)
            video_ok = video_ok & causal_video_visibility(frames[qi], frames[ki], frames_per_chunk=C, local_attention_frames=W)
            return (q < self.flat_gen_tokens) & (text_ok | video_ok)
        return mask

    def block_mask(self, text_pad):
        if text_pad not in self._masks:
            from torch.nn.attention.flex_attention import BlockMask
            from cosmos_framework.model.generator.mot.flex_attention_utils import (
                metadata_run_groups, build_block_mask_from_metadata_runs,
            )
            # The predicate is constant within each sample/frame metadata run.
            # Reuse the native run-level builder: create_block_mask materializes
            # token-pair masks (and a large int64 reduction temporary at 75k).
            text_samples = F.pad(self.text_samples, (0, text_pad-len(self.text_samples)), value=-2)
            q_groups, q_representatives = metadata_run_groups(
                (self.samples, self.frames), device=self.device)
            kv_groups, kv_representatives = metadata_run_groups((
                torch.cat((torch.zeros(text_pad, device=self.device, dtype=torch.long),
                           torch.ones(self.gen_pad, device=self.device, dtype=torch.long))),
                torch.cat((text_samples, self.samples)),
                torch.cat((torch.full((text_pad,), -2, device=self.device, dtype=torch.long), self.frames)),
            ), device=self.device)
            mask_mod = self.mask_mod(text_pad)
            mask = build_block_mask_from_metadata_runs(
                q_group_id=q_groups, kv_group_id=kv_groups,
                q_representatives=q_representatives, kv_representatives=kv_representatives,
                pair_allowed=mask_mod, mask_mod=mask_mod,
                q_len=self.gen_pad, kv_len=text_pad+self.gen_pad,
                device=self.device, block_size=(128, 128))
            # Ordered masked tiles keep traversal identical between layouts and caches.
            dense = mask.to_dense()
            counts = dense.sum(-1).to(torch.int32)
            indices = torch.argsort(dense.to(torch.int32), dim=-1, descending=True, stable=True).to(torch.int32)
            self._masks[text_pad] = BlockMask.from_kv_blocks(counts, indices, BLOCK_SIZE=mask.BLOCK_SIZE,
                                                         mask_mod=mask.mask_mod, seq_lengths=mask.seq_lengths)
        return self._masks[text_pad]

    def __call__(self, q, k, v, text_k, text_v, memory_value):
        from cosmos_framework.model.generator.mot.flex_attention import _COMPILED_FLEX_ATTENTION
        shape = q.shape
        if shape[0] != 1 or shape[1]*shape[2] != self.flat_gen_tokens:
            raise ValueError("GEN query length does not match pure-video layout")
        text_pad = ((max(text_k.shape[1], 1)+127)//128)*128
        def pad_rows(x, size):
            return F.pad(x, (0,0,0,0,0,size-x.shape[1]))
        q = pad_rows(q.reshape(1,self.flat_gen_tokens,shape[-2],shape[-1]),self.gen_pad)
        k = torch.cat([pad_rows(text_k,text_pad),pad_rows(k.flatten(1,2),self.gen_pad)],1).to(q.dtype)
        v = torch.cat([pad_rows(text_v,text_pad),pad_rows(v.flatten(1,2),self.gen_pad)],1).to(q.dtype)
        out = _COMPILED_FLEX_ATTENTION(q.transpose(1,2).contiguous(), k.transpose(1,2).contiguous(),
                                     v.transpose(1,2).contiguous(), block_mask=self.block_mask(text_pad),
                                     enable_gqa=q.shape[2] != k.shape[2])
        return out.transpose(1,2)[:,:self.flat_gen_tokens].reshape(shape)
