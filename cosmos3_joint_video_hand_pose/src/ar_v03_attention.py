"""One-pass diffusion-forcing attention over the existing joint U/S/V/A layout."""

import torch

from .ar_attention import FLEX_BLOCK_SIZE, _pad_rows, _round_up
from .ar_v02_attention import create_joint_block_mask
from .ar_v02_layout import JointChunkLayout, joint_mask_mod


class JointDiffusionForcingAttention:
    """Use the V0.2 clean-pass mask on one *noisy* GEN stream.

    A chunk's V/A keys are the very same differentiable tensors used for its
    own prediction and for later chunks' history. There is no clean replay,
    detached history, or second GEN stream. U/S cleanliness is owned by the
    model's noise routing; their queries retain the existing condition-only
    visibility rule.
    """

    def __init__(self, layout, device, *, text_lengths=None, compile_block_mask=False):
        if compile_block_mask:
            raise ValueError("joint attention requires eager block-mask construction for verified gradients")
        self.layout = layout
        self.layouts = [layout] if isinstance(layout, JointChunkLayout) else list(layout)
        if not self.layouts:
            raise ValueError("at least one joint layout is required")
        if text_lengths is None and len(self.layouts) > 1:
            raise ValueError("packed samples require explicit text_lengths for sample isolation")
        if text_lengths is not None:
            text_lengths = [int(n) for n in text_lengths]
            if len(text_lengths) != len(self.layouts) or any(n < 0 for n in text_lengths):
                raise ValueError("one nonnegative text length per joint sample is required")
        self.device = torch.device(device)
        self.gen_len = self.flat_gen_tokens = sum(x.num_tokens for x in self.layouts)
        self.gen_pad_len = _round_up(self.gen_len, FLEX_BLOCK_SIZE)
        self.roles = torch.full((self.gen_pad_len,), -1, device=self.device, dtype=torch.long)
        self.chunks = self.roles.clone()
        self.sample_ids = self.roles.clone()
        metadata = [x.metadata(device=self.device) for x in self.layouts]
        self.roles[:self.gen_len] = torch.cat([x[0] for x in metadata])
        self.chunks[:self.gen_len] = torch.cat([x[1] for x in metadata])
        self.sample_ids[:self.gen_len] = torch.cat([
            torch.full((x.num_tokens,), i, device=self.device, dtype=torch.long)
            for i, x in enumerate(self.layouts)
        ])
        self.text_sample_ids = None
        if text_lengths is not None:
            self.text_sample_ids = torch.cat([
                torch.full((n,), i, device=self.device, dtype=torch.long)
                for i, n in enumerate(text_lengths)
            ])
            if self.text_sample_ids.numel() == 0:
                self.text_sample_ids = torch.full((1,), -1, device=self.device, dtype=torch.long)
        self._block_masks = {}
        self._text_len = None

    def block_mask(self, *, text_pad_len, text_len):
        key = (text_pad_len, text_len)
        if key not in self._block_masks:
            predicate = joint_mask_mod(
                self.roles, self.chunks, text_pad_len=text_pad_len, text_len=text_len,
                noisy=False, history_chunks=15, sample_ids=self.sample_ids,
                text_sample_ids=self.text_sample_ids,
            )
            self._block_masks[key] = create_joint_block_mask(
                predicate, B=None, H=None, Q_LEN=self.gen_pad_len,
                KV_LEN=text_pad_len + self.gen_pad_len, device=self.device,
                BLOCK_SIZE=FLEX_BLOCK_SIZE,
            )
        return self._block_masks[key]

    def _real_text_len(self, memory_value):
        if self._text_len is None:
            self._text_len = int(memory_value.und_kv_offsets[-1]) if bool(memory_value.has_caption) else 0
        return self._text_len

    def __call__(self, q_2d, k_2d, v_2d, text_k, text_v, memory_value):
        from cosmos_framework.model.generator.mot.flex_attention import _COMPILED_FLEX_ATTENTION
        from cosmos_framework.model.generator.utils.kv_cache import KVTrainMemoryValue

        if not isinstance(memory_value, KVTrainMemoryValue):
            raise TypeError(f"joint diffusion forcing expects KVTrainMemoryValue, got {type(memory_value).__name__}")
        batch, num_frames, tokens_per_frame, num_heads, head_dim = q_2d.shape
        if (batch, num_frames, tokens_per_frame) != (1, 1, self.flat_gen_tokens):
            raise ValueError("packed GEN query does not match the joint diffusion-forcing layout")
        num_kv_heads = k_2d.shape[3]
        text_len = self._real_text_len(memory_value)
        text_pad = _round_up(max(text_k.shape[1], 1), FLEX_BLOCK_SIZE)
        if text_len > text_k.shape[1]:
            raise ValueError(f"text length {text_len} exceeds the {text_k.shape[1]} provided text keys")

        query = _pad_rows(q_2d.reshape(1, self.gen_len, num_heads, head_dim), self.gen_pad_len)
        key = torch.cat([
            _pad_rows(text_k, text_pad),
            _pad_rows(k_2d.reshape(1, self.gen_len, num_kv_heads, head_dim), self.gen_pad_len),
        ], dim=1).to(query.dtype)
        value = torch.cat([
            _pad_rows(text_v, text_pad),
            _pad_rows(v_2d.reshape(1, self.gen_len, num_kv_heads, head_dim), self.gen_pad_len),
        ], dim=1).to(query.dtype)
        out = _COMPILED_FLEX_ATTENTION(
            query.transpose(1, 2).contiguous(), key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            block_mask=self.block_mask(text_pad_len=text_pad, text_len=text_len),
            enable_gqa=num_heads != num_kv_heads,
        )
        return out.transpose(1, 2)[:, :self.gen_len].reshape_as(q_2d)
