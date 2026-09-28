"""Lingbot-VA teacher-forcing attention for joint video-action AR training on Cosmos.

Installed on Cosmos' replayed teacher forcing through
``KVTrainMemoryValue.gen_attention_override``: Pass 1 (clean) and Pass 2 (noisy)
of ``OmniMoTCausalModel`` then follow the lingbot-va visibility rules below
instead of Cosmos' built-in chunkwise rules.

Token layout (Cosmos temporal-causal supertokens): latent frame ``t`` packs
``[action_t (K tokens), vision_t (HW tokens)]``. Frame 0 is its own chunk; frames
``1..T-1`` form chunks of ``C`` frames (the last chunk may be shorter).
Block ids (lingbot ``frame_ids``): chunk ``c`` gives vision ``2c`` and action ``2c+1``.

Visibility (real text keys are visible to every GEN query):

* clean query -> clean key: ``id_k <= id_q``
* noisy query -> noisy key: ``id_k == id_q``
* noisy query -> clean key: ``id_k < id_q``

and every GEN pair also needs ``|id_q - id_k| <= window`` (``None`` = unbounded).
Noisy video of chunk ``c`` therefore sees clean video and action of earlier chunks;
noisy action of chunk ``c`` additionally sees the clean video of chunk ``c``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

FLEX_BLOCK_SIZE = 128
_UNBOUNDED_WINDOW = 1 << 30


@dataclass(frozen=True)
class ARChunkLayout:
    """Supertoken geometry and chunk/window parameters of one packed sample."""

    num_frames: int
    action_tokens: int
    vision_tokens: int
    chunk_size: int
    window: int | None = None

    def __post_init__(self) -> None:
        for name in ("num_frames", "action_tokens", "vision_tokens", "chunk_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.window is not None and int(self.window) < 0:
            raise ValueError(f"window must be >= 0 or None, got {self.window}")

    @property
    def tokens_per_frame(self) -> int:
        return self.action_tokens + self.vision_tokens

    @property
    def num_tokens(self) -> int:
        return self.num_frames * self.tokens_per_frame

    @property
    def num_chunks(self) -> int:
        return 1 + math.ceil((self.num_frames - 1) / self.chunk_size)


def frame_chunk_ids(num_frames: int, chunk_size: int) -> torch.Tensor:
    """Chunk index per latent frame for the ``[1, C, C, ...]`` partition (last chunk may be short)."""
    if num_frames < 1 or chunk_size < 1:
        raise ValueError(f"num_frames and chunk_size must be >= 1, got {num_frames}, {chunk_size}")
    frames = torch.arange(num_frames)
    return torch.where(frames == 0, torch.zeros_like(frames), (frames - 1) // chunk_size + 1)  # [T]


def token_block_ids(layout: ARChunkLayout) -> torch.Tensor:
    """Lingbot block id of every GEN token in supertoken order ``[action_t, vision_t]``."""
    chunk = frame_chunk_ids(layout.num_frames, layout.chunk_size)  # [T]
    action = (2 * chunk + 1)[:, None].expand(-1, layout.action_tokens)  # [T,K]
    vision = (2 * chunk)[:, None].expand(-1, layout.vision_tokens)  # [T,HW]
    return torch.cat([action, vision], dim=1).reshape(-1)  # [T*S]


def make_mask_mod(
    block_ids: torch.Tensor,
    *,
    gen_pad_len: int,
    text_pad_len: int,
    text_len: int,
    noisy: bool,
    window: int | None,
):
    """FlexAttention predicate over keys ``[text | GEN (| clean GEN)]`` for GEN queries.

    ``block_ids`` has ``gen_pad_len`` entries; padding positions carry ``-1``. The key
    stream is ``text_pad_len`` text slots (the first ``text_len`` are real), then the
    GEN stream of the current pass, then -- in the noisy pass -- the clean GEN stream.
    Padded queries attend only to padded keys of their own stream, so no row is empty.

    Lengths and the window are captured as tensors so a compiled block-mask build is
    reused across steps whose window or text length differ.
    """
    device = block_ids.device
    gen_pad = torch.tensor(gen_pad_len, device=device)
    text_pad = torch.tensor(text_pad_len, device=device)
    text_real = torch.tensor(text_len, device=device)
    limit = torch.tensor(_UNBOUNDED_WINDOW if window is None else int(window), device=device)
    zero = torch.zeros((), dtype=torch.int64, device=device)

    def mask_mod(b, h, q_idx, kv_idx):
        del b, h
        q_id = block_ids[q_idx]
        q_real = q_id >= 0
        is_text = kv_idx < text_pad
        gen_kv = kv_idx - text_pad
        is_clean_stream = gen_kv >= gen_pad
        gen_idx = torch.where(is_clean_stream, gen_kv - gen_pad, gen_kv)
        gen_idx = torch.minimum(torch.maximum(gen_idx, zero), gen_pad - 1)
        k_id = block_ids[gen_idx]
        k_real = k_id >= 0
        if noisy:
            relation = torch.where(is_clean_stream, k_id < q_id, k_id == q_id)
        else:
            relation = k_id <= q_id
        text_ok = is_text & (kv_idx < text_real) & q_real
        gen_ok = (~is_text) & q_real & k_real & relation & ((q_id - k_id).abs() <= limit)
        pad_ok = (~is_text) & (~q_real) & (~k_real) & (~is_clean_stream)
        return text_ok | gen_ok | pad_ok

    return mask_mod


_COMPILED_CREATE_BLOCK_MASK = None


def _create_block_mask(compiled: bool):
    """``create_block_mask``, compiled once per process when requested (avoids the dense eager build)."""
    global _COMPILED_CREATE_BLOCK_MASK
    from torch.nn.attention.flex_attention import create_block_mask

    if not compiled:
        return create_block_mask
    if _COMPILED_CREATE_BLOCK_MASK is None:
        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask)
    return _COMPILED_CREATE_BLOCK_MASK


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_rows(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Zero-pad dim 1 of ``[1, N, H, D]`` to ``length`` rows."""
    missing = length - tensor.shape[1]
    if missing < 0:
        raise ValueError(f"cannot pad {tensor.shape[1]} rows down to {length}")
    return F.pad(tensor, (0, 0, 0, 0, 0, missing)) if missing else tensor


class LingbotTeacherForcingAttention:
    """``gen_attention_override`` implementing the lingbot-va rules for one packed sample.

    One FlexAttention call per layer covers GEN self-attention and the video->text
    cross-attention in a single softmax, so gradients come from FlexAttention's own
    autograd. Block masks are built once per pass and reused by every layer (and by
    activation-checkpoint recomputation).
    """

    def __init__(self, layout: ARChunkLayout, device: torch.device | str, *, compile_block_mask: bool = True):
        self.layout = layout
        self.device = torch.device(device)
        self.compile_block_mask = compile_block_mask
        self.gen_len = layout.num_tokens
        self.gen_pad_len = _round_up(self.gen_len, FLEX_BLOCK_SIZE)
        ids = torch.full((self.gen_pad_len,), -1, dtype=torch.int64)
        ids[: self.gen_len] = token_block_ids(layout)
        self.block_ids = ids.to(self.device)
        self._block_masks: dict[tuple[bool, int, int], object] = {}
        self._text_len: int | None = None

    def block_mask(self, *, noisy: bool, text_pad_len: int, text_len: int):
        key = (noisy, text_pad_len, text_len)
        if key not in self._block_masks:
            mask_mod = make_mask_mod(
                self.block_ids,
                gen_pad_len=self.gen_pad_len,
                text_pad_len=text_pad_len,
                text_len=text_len,
                noisy=noisy,
                window=self.layout.window,
            )
            kv_len = text_pad_len + self.gen_pad_len * (2 if noisy else 1)
            self._block_masks[key] = _create_block_mask(self.compile_block_mask)(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=self.gen_pad_len,
                KV_LEN=kv_len,
                device=self.device,
                BLOCK_SIZE=FLEX_BLOCK_SIZE,
            )
        return self._block_masks[key]

    def _real_text_len(self, memory_value) -> int:
        # Constant within one packed sample; read once to avoid a host sync per layer.
        if self._text_len is None:
            has_caption = bool(memory_value.has_caption)
            self._text_len = int(memory_value.und_kv_offsets[-1]) if has_caption else 0
        return self._text_len

    def __call__(self, q_2d, k_2d, v_2d, text_k, text_v, memory_value):
        from cosmos_framework.model.generator.mot.flex_attention import _COMPILED_FLEX_ATTENTION
        from cosmos_framework.model.generator.utils.kv_cache import TFNoisyMemoryValue, TFReplayCleanMemoryValue

        noisy = isinstance(memory_value, TFNoisyMemoryValue)
        if not noisy and not isinstance(memory_value, TFReplayCleanMemoryValue):
            raise TypeError(f"lingbot teacher forcing expects replay TF memory, got {type(memory_value).__name__}")
        _, num_frames, tokens_per_frame, num_heads, head_dim = q_2d.shape
        expected_shape = (
            (1, self.flat_gen_tokens)
            if hasattr(self, "flat_gen_tokens")
            else (self.layout.num_frames, self.layout.tokens_per_frame)
        )
        if (num_frames, tokens_per_frame) != expected_shape:
            raise ValueError(
                f"packed GEN layout {(num_frames, tokens_per_frame)} does not match the attention layout "
                f"{expected_shape}"
            )
        num_kv_heads = k_2d.shape[3]
        gen_len, gen_pad = self.gen_len, self.gen_pad_len
        text_len = self._real_text_len(memory_value)
        text_pad = _round_up(max(text_k.shape[1], 1), FLEX_BLOCK_SIZE)
        if text_len > text_k.shape[1]:
            raise ValueError(f"text length {text_len} exceeds the {text_k.shape[1]} provided text keys")

        query = _pad_rows(q_2d.reshape(1, gen_len, num_heads, head_dim), gen_pad)
        keys = [_pad_rows(text_k, text_pad), _pad_rows(k_2d.reshape(1, gen_len, num_kv_heads, head_dim), gen_pad)]
        values = [_pad_rows(text_v, text_pad), _pad_rows(v_2d.reshape(1, gen_len, num_kv_heads, head_dim), gen_pad)]
        if noisy:
            keys.append(_pad_rows(memory_value.cached_clean_gen_k[:, :gen_len], gen_pad))
            values.append(_pad_rows(memory_value.cached_clean_gen_v[:, :gen_len], gen_pad))
        key = torch.cat(keys, dim=1).to(query.dtype)  # [1,KV,H_kv,D]
        value = torch.cat(values, dim=1).to(query.dtype)  # [1,KV,H_kv,D]
        block_mask = self.block_mask(noisy=noisy, text_pad_len=text_pad, text_len=text_len)
        out = _COMPILED_FLEX_ATTENTION(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            block_mask=block_mask,
            enable_gqa=num_heads != num_kv_heads,
        )  # [1,H,Lp,D]
        return out.transpose(1, 2)[:, :gen_len].reshape(1, num_frames, tokens_per_frame, num_heads, head_dim)
