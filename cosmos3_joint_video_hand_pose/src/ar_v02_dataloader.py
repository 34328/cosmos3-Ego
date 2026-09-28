"""Use Cosmos' greedy packer with the actual two-pass joint layout budget."""

import math
import operator
import torch
from .dataloader_state import RecoverablePackingDataLoader
from .ar_v02_layout import LAYOUT_VERSION
from .ar_dataset import AR_V02_TOKEN_BUDGET_VERSION

JOINT_ATTENTION_ALIGNMENT = 128
# CP1, CUDA-graph padding disabled: runtime adds a non-empty trailing pad
# segment to each stream before attention. Two streams * two passes * 128.
JOINT_PACK_PADDING_RESERVE = 4 * JOINT_ATTENTION_ALIGNMENT


def _integer(value, name, minimum=1):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def joint_training_token_budget(text_tokens, frames, height, width, patch_pixels=32):
    """C=1 raw forward-token bound, excluding per-pack padding (unchanged)."""
    text_tokens = _integer(text_tokens, "text_tokens", 0)
    frames = _integer(frames, "frames")
    height = _integer(height, "height")
    width = _integer(width, "width")
    patch_pixels = _integer(patch_pixels, "patch_pixels")
    if frames < 5 or (frames - 1) % 4:
        raise ValueError("expected 1+4N RGB frames")
    n = (frames - 1) // 4
    patches = math.ceil(height / patch_pixels) * math.ceil(width / patch_pixels)
    # Worst C=1: each group contributes one U, one S, one V and eight A.
    # Current replay retains zero-loss condition query rows in pass 2 as well.
    # Count them honestly; no assumption that masking removes their compute.
    return 2 * (int(text_tokens) + 2 + n * (2 * patches + 9))


def joint_pack_padding_audit(text_tokens_per_pass, gen_tokens_per_pass):
    """Audit CP1 three-way replay storage, not FLOPs or total Q/K/V bytes.

    Text includes each sample's EOS/generation markers. runtime.py adds one
    pad row per stream; the custom Flex path rounds the provided text KV and
    real GEN lengths to 128, strips runtime GEN padding, then restores it.
    The envelope below counts each stream once per pass. Noisy KV also reads
    clean GEN memory, so its separate capacity must not be called forward rows.
    """
    text = _integer(text_tokens_per_pass, "text_tokens_per_pass")
    gen = _integer(gen_tokens_per_pass, "gen_tokens_per_pass")
    alignment = JOINT_ATTENTION_ALIGNMENT
    text_pad = ((text + 1 + alignment - 1) // alignment) * alignment
    gen_pad = ((gen + alignment - 1) // alignment) * alignment
    raw = 2 * (text + gen)
    capacity = 2 * (text_pad + max(gen + 1, gen_pad))
    return dict(
        raw_forward_tokens=raw,
        runtime_forward_capacity=raw + 4,
        attention_padded_capacity=capacity,
        padding_overhead=capacity - raw,
        flex_gen_query_capacity_per_pass=gen_pad,
        flex_clean_kv_capacity=text_pad + gen_pad,
        flex_noisy_kv_capacity=text_pad + 2 * gen_pad,
        reserved_budget_tokens=raw + JOINT_PACK_PADDING_RESERVE,
    )


def _unwrap(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


class JointChunkPackingDataLoader(RecoverablePackingDataLoader):
    def __init__(self, *args, joint_max_samples=4, **kwargs):
        self.joint_max_samples = _integer(joint_max_samples, "joint_max_samples")
        # Resume must restore the inner stream before workers/prefetch start.
        kwargs.setdefault("lazy_initialize_child_iterators", True)
        super().__init__(*args, **kwargs)
        if self.tokenizer_temporal_compression_factor != 4:
            raise ValueError("joint_chunk_cond_v1 requires temporal compression 4")
        if self.max_sequence_length is not None:
            _integer(self.max_sequence_length, "max_sequence_length")
            if self.max_sequence_length <= JOINT_PACK_PADDING_RESERVE:
                raise ValueError("token cap must exceed the per-pack padding reserve")
        if any(limit < 1 for limit in self.lookahead_limits):
            raise ValueError("lookahead_limit must be positive; zero forces one-sample packs")
        # Parent initialization requires mutually exclusive token/sample config.
        # Its iteration loop already implements BOTH checks independently. Add
        # our second ceiling only after initialization, so the loop stops before
        # reading another candidate at the limit, without a stateful fit counter.
        configured_samples = self.max_samples_per_batch
        self.max_samples_per_batch = min(
            self.joint_max_samples,
            (
                _integer(configured_samples, "max_samples_per_batch")
                if configured_samples is not None
                else self.joint_max_samples
            ),
        )

    def _sample_fits(self, **kwargs):
        # Reserve once for the entire candidate pack, not for each sample.
        # Keep max_sequence_length=70000 and all raw per-sample counts intact.
        kwargs["packed_tokens"] += JOINT_PACK_PADDING_RESERVE
        return super()._sample_fits(**kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            text = sum(_unwrap(tokens).numel() + 2 for tokens in batch["text_token_ids"])
            raw = int(batch["_num_tokens"])
            audit = joint_pack_padding_audit(text, raw // 2 - text)
            if raw != audit["raw_forward_tokens"]:
                raise ValueError("raw pack count differs from its stream totals")
            if audit["padding_overhead"] > JOINT_PACK_PADDING_RESERVE:
                raise ValueError("attention padding exceeds the reserved pack budget")
            if self.max_sequence_length is not None and audit["reserved_budget_tokens"] >= self.max_sequence_length:
                raise ValueError("pack exceeds the padding-inclusive token cap")
            batch.update({"_ar_c1_" + key: value for key, value in audit.items()})
            batch["_ar_pack_padding_reserve"] = JOINT_PACK_PADDING_RESERVE
            yield batch

    def _compute_token_split_per_sample(self, data_batch):
        if _unwrap(data_batch.get("ar_layout_version")) != LAYOUT_VERSION:
            raise ValueError("joint packer requires versioned chunk-conditioned samples")
        rgb = _unwrap(data_batch["video"])
        if not isinstance(rgb, torch.Tensor) or rgb.ndim not in (4, 5) or (rgb.ndim == 5 and rgb.shape[0] != 1):
            raise ValueError("joint packer expects one RGB clip per sample")
        if rgb.shape[-4] != 3:
            raise ValueError("joint packer expects three RGB channels")
        t, h, w = rgb.shape[-3:]
        text = _unwrap(data_batch["text_token_ids"])
        if not isinstance(text, torch.Tensor) or text.ndim not in (1, 2) or (text.ndim == 2 and text.shape[0] != 1):
            raise ValueError("joint packer expects one tokenized caption per sample")
        nt = text.numel()
        count = joint_training_token_budget(nt, t, h, w, self.tokenizer_spatial_compression_factor * self.patch_spatial)
        action = _unwrap(data_batch["action"])
        if (
            not isinstance(action, torch.Tensor)
            or action.ndim not in (2, 3)
            or (action.ndim == 3 and action.shape[0] != 1)
            or action.shape[-2:] != (2 * (t - 1), 64)
        ):
            raise ValueError("joint budget requires future-only K=8 zero-padded 64D action rows")
        if "ar_num_tokens" in data_batch or "ar_token_budget_version" in data_batch:
            if _unwrap(data_batch.get("ar_token_budget_version")) != AR_V02_TOKEN_BUDGET_VERSION:
                raise ValueError("dataset token budget version mismatch")
            declared = _unwrap(data_batch.get("ar_num_tokens"))
            if isinstance(declared, torch.Tensor) and declared.numel() == 1:
                declared = declared.item()
            if _integer(declared, "ar_num_tokens") != count:
                raise ValueError("dataset token budget differs from actual RGB/action/text geometry")
        return 2 * (nt + 2), count - 2 * (nt + 2)
