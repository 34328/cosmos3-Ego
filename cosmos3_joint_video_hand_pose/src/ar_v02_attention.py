"""Joint noisy V/A attention with explicit states and a 15-chunk history."""

import torch

from .ar_attention import FLEX_BLOCK_SIZE, LingbotTeacherForcingAttention, _create_block_mask, _round_up
from .ar_v02_layout import JointChunkLayout, joint_mask_mod


def create_joint_block_mask(*args, **kwargs):
    """Keep ascending KV traversal independent of full/partial tile classification.

    FlexAttention normally visits fully-visible tiles separately from masked ones.
    Dropping zero-loss queries or adding cache padding can change that classification
    without changing a real query's visible keys. Online softmax rounds its block
    probabilities in bf16, so reordering these tiles changes actual Nano outputs.
    Treat all retained tiles as masked: identical visibility, one ordered traversal.
    Build once per layout, not once per layer or denoising step.
    """
    from torch.nn.attention.flex_attention import BlockMask

    mask = _create_block_mask(False)(*args, **kwargs)
    dense = mask.to_dense()
    counts = dense.sum(-1).to(torch.int32)
    indexes = torch.argsort(dense.to(torch.int32), dim=-1, descending=True, stable=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        counts, indexes, BLOCK_SIZE=mask.BLOCK_SIZE, mask_mod=mask.mask_mod, seq_lengths=mask.seq_lengths
    )


class JointTeacherForcingAttention(LingbotTeacherForcingAttention):
    def __init__(self, layout, device, *, text_lengths=None, compile_block_mask=False):
        # On the installed torch 2.10 runtime, reusing compiled create_block_mask
        # across clean/noisy layouts corrupted text-K gradients despite matching
        # forward values. Eager mask construction avoids that reproduced failure;
        # the actual FlexAttention forward/backward kernels remain compiled.
        if compile_block_mask:
            raise ValueError("v0.2 requires eager block-mask construction until its gradient equivalence is verified")
        self.layout = layout
        self.device = torch.device(device)
        self.compile_block_mask = compile_block_mask
        layouts = [layout] if isinstance(layout, JointChunkLayout) else list(layout)
        self.layouts = layouts
        self.gen_len = self.flat_gen_tokens = sum(x.num_tokens for x in layouts)
        self.gen_pad_len = _round_up(self.gen_len, FLEX_BLOCK_SIZE)
        roles = torch.cat([x.metadata(device=self.device)[0] for x in layouts])
        chunks = torch.cat([x.metadata(device=self.device)[1] for x in layouts])
        self.sample_ids = torch.full((self.gen_pad_len,), -1, device=self.device, dtype=torch.long)
        self.sample_ids[: self.gen_len] = torch.cat(
            [torch.full((x.num_tokens,), i, device=self.device, dtype=torch.long) for i, x in enumerate(layouts)]
        )
        self.text_sample_ids = (
            torch.cat([torch.full((n,), i, device=self.device, dtype=torch.long) for i, n in enumerate(text_lengths)])
            if text_lengths is not None
            else None
        )
        self.roles = torch.full((self.gen_pad_len,), -1, device=self.device, dtype=torch.long)
        self.chunks = self.roles.clone()
        self.roles[: self.gen_len], self.chunks[: self.gen_len] = roles, chunks
        self._block_masks = {}
        self._text_len = None

    def block_mask(self, *, noisy, text_pad_len, text_len):
        key = (noisy, text_pad_len, text_len)
        if key not in self._block_masks:
            predicate = joint_mask_mod(
                self.roles,
                self.chunks,
                text_pad_len=text_pad_len,
                text_len=text_len,
                noisy=noisy,
                history_chunks=15,
                sample_ids=self.sample_ids,
                text_sample_ids=self.text_sample_ids,
            )
            self._block_masks[key] = create_joint_block_mask(
                predicate,
                B=None,
                H=None,
                Q_LEN=self.gen_pad_len,
                KV_LEN=text_pad_len + self.gen_pad_len * (2 if noisy else 1),
                device=self.device,
                BLOCK_SIZE=FLEX_BLOCK_SIZE,
            )
        return self._block_masks[key]
