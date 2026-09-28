"""Target-only noisy teacher forcing; clean condition/text KV keeps gradients."""

import dataclasses
import torch

from cosmos_framework.model.generator.utils.memory import MemoryState
from .ar_attention import FLEX_BLOCK_SIZE, _pad_rows, _round_up
from .ar_v02_attention import create_joint_block_mask
from .ar_v02_layout import ACTION, VIDEO, STATE, CONDITION_VIDEO, LAYOUT_VERSION


def compact_joint_targets(packed):
    """Remove text/U/S queries by explicit roles, preserving all target timestamps."""
    if getattr(packed, "joint_layout_version", None) != LAYOUT_VERSION:
        raise ValueError("compact teacher forcing requires an explicit joint layout")
    layouts = packed.joint_layouts
    selections, sample_lens = [], []
    offset = 0
    for layout, nt, n in zip(layouts, packed.joint_text_lengths, packed.sample_lens, strict=True):
        roles, _, _ = layout.metadata()
        selected = torch.where((roles == VIDEO) | (roles == ACTION))[0]
        if n != nt + layout.num_tokens:
            raise ValueError("joint sample length differs from declared roles")
        selections.append(selected + offset + nt)
        sample_lens.append(selected.numel())
        offset += n
    selected = torch.cat(selections)
    remap = torch.full((packed.sequence_length,), -1, dtype=torch.long)
    remap[selected] = torch.arange(selected.numel())
    row_maps = {}

    def select_modality(name):
        mod = getattr(packed, name)
        video = name == "vision"
        rows = [
            (
                torch.where(lay.video_metadata()[0] == VIDEO)[0]
                if video
                else torch.where(lay.action_metadata()[0] == ACTION)[0]
            )
            for lay in layouts
        ]
        row_maps[name] = rows
        tokens, shapes, conditions, noisy = [], [], [], []
        for i, idx in enumerate(rows):
            tokens.append(mod.tokens[i].index_select(2 if video else 0, idx.to(mod.tokens[i].device)))
            shapes.append((len(idx), *mod.token_shapes[i][1:]))
            cond = mod.condition_mask[i].index_select(0, idx.to(mod.condition_mask[i].device))
            conditions.append(cond)
            noisy.append(torch.where(cond.reshape(len(idx), -1)[:, 0] == 0)[0])
        original = mod.sequence_indexes.cpu()
        keep = remap[original] >= 0
        loss = mod.mse_loss_indexes.cpu()
        if (remap[loss] < 0).any():
            raise ValueError("condition tokens unexpectedly carry flow loss")
        domains = mod.domain_id
        if domains:
            domains = [
                d if d.numel() == 1 else d.reshape(-1).index_select(0, idx.to(d.device))
                for d, idx in zip(domains, rows, strict=True)
            ]
        return dataclasses.replace(
            mod,
            tokens=tokens,
            token_shapes=shapes,
            sequence_indexes=remap[original[keep]],
            mse_loss_indexes=remap[loss],
            condition_mask=conditions,
            noisy_frame_indexes=noisy,
            domain_id=domains,
        )

    vision, action = select_modality("vision"), select_modality("action")
    compact = dataclasses.replace(
        packed,
        sample_lens=sample_lens,
        split_lens=sample_lens.copy(),
        attn_modes=["full"] * len(layouts),
        sequence_length=selected.numel(),
        text_ids=torch.empty(0, dtype=torch.long),
        text_indexes=torch.empty(0, dtype=torch.long),
        position_ids=packed.position_ids[:, selected.to(packed.position_ids.device)],
        vision=vision,
        action=action,
        text_caption_lens=[],
        text_caption_view_ids=[],
        action_state_mask=torch.zeros(sum(len(x) for x in row_maps["action"]), dtype=torch.bool),
        vision_condition_type_mask=torch.zeros(vision.sequence_indexes.numel(), dtype=torch.bool),
        vision_item_split_lens=[[n] for n in sample_lens],
    )
    return compact, row_maps


def restore_joint_predictions(output, original, row_maps):
    """Scatter velocities back to the loss layout; excluded conditions have zero loss."""
    result = dict(output)
    for name in ("vision", "action"):
        key = "preds_" + name
        dim = 2 if name == "vision" else 0
        result[key] = [
            p.new_zeros(tok.shape).index_copy(dim, idx.to(p.device), p)
            for p, tok, idx in zip(output[key], getattr(original, name).tokens, row_maps[name], strict=True)
        ]
    return result


class CompactJointAttention:
    cached_text_only = True

    def __init__(self, layouts, text_lengths, device):
        meta = [lay.metadata(device=device) for lay in layouts]
        roles = torch.cat([x[0] for x in meta])
        chunks = torch.cat([x[1] for x in meta])
        samples = torch.cat([torch.full_like(x[0], i) for i, x in enumerate(meta)])
        target = (roles == VIDEO) | (roles == ACTION)
        self.target_positions = torch.where(target)[0]
        self.flat_gen_tokens = int(target.sum())
        self.clean_len = len(roles)
        self.qpad = _round_up(self.flat_gen_tokens, FLEX_BLOCK_SIZE)
        self.cpad = _round_up(self.clean_len, FLEX_BLOCK_SIZE)
        pad = lambda x, n: torch.nn.functional.pad(x, (0, n - len(x)), value=-1)
        self.qmeta = tuple(pad(x[target], self.qpad) for x in (roles, chunks, samples))
        self.cmeta = tuple(pad(x, self.cpad) for x in (roles, chunks, samples))
        self.text_samples = torch.cat(
            [torch.full((n,), i, device=device, dtype=torch.long) for i, n in enumerate(text_lengths)]
        )
        self.text_len = sum(text_lengths)
        self.masks = {}

    def predicate(self, text_pad):
        qr, qc, qs = self.qmeta
        kr, kc, ks = (torch.cat((c, c)) for c in self.cmeta)
        cpad, text_len, text_samples = self.cpad, self.text_len, self.text_samples

        def mask(b, h, q, kv):
            del b, h
            is_text = kv < text_pad
            idx = (kv - text_pad).clamp(0, len(kr) - 1)
            clean = idx >= cpad
            same = qc[q] == kc[idx]
            condition = (kr[idx] == STATE) | (kr[idx] == CONDITION_VIDEO)
            past = (kc[idx] < qc[q]) & (kc[idx] >= qc[q] - 15)
            relation = torch.where(clean, past | (same & condition), same & (~condition))
            real = (qr[q] >= 0) & (kr[idx] >= 0) & (qs[q] == ks[idx])
            text = is_text & (kv < text_len) & (qr[q] >= 0) & (qs[q] == text_samples[kv.clamp(0, text_len - 1)])
            padding = (~is_text) & (~clean) & (qr[q] < 0) & (kr[idx] < 0)
            return text | ((~is_text) & real & relation) | padding

        return mask

    def __call__(self, q, k, v, text_k, text_v, memory):
        from cosmos_framework.model.generator.mot.flex_attention import _COMPILED_FLEX_ATTENTION

        heads, dim, kvheads = q.shape[-2], q.shape[-1], k.shape[-2]
        n = self.flat_gen_tokens
        text_pad = _round_up(max(text_k.shape[1], 1), FLEX_BLOCK_SIZE)
        if text_pad not in self.masks:
            self.masks[text_pad] = create_joint_block_mask(
                self.predicate(text_pad),
                B=None,
                H=None,
                Q_LEN=self.qpad,
                KV_LEN=text_pad + 2 * self.cpad,
                device=q.device,
                BLOCK_SIZE=FLEX_BLOCK_SIZE,
            )
        query = _pad_rows(q.reshape(1, n, heads, dim), self.qpad)

        # Compact queries, but keep original absolute key tile positions. Removing
        # U/S key holes otherwise changes bf16 block reductions despite the same mask.
        def expand_keys(x):
            return x.new_zeros((1, self.cpad, kvheads, dim)).index_copy(
                1, self.target_positions, x.reshape(1, n, kvheads, dim)
            )

        key = torch.cat(
            (
                _pad_rows(text_k, text_pad),
                expand_keys(k),
                _pad_rows(memory.cached_clean_gen_k[:, : self.clean_len], self.cpad),
            ),
            1,
        ).to(q.dtype)
        value = torch.cat(
            (
                _pad_rows(text_v, text_pad),
                expand_keys(v),
                _pad_rows(memory.cached_clean_gen_v[:, : self.clean_len], self.cpad),
            ),
            1,
        ).to(q.dtype)
        out = _COMPILED_FLEX_ATTENTION(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            block_mask=self.masks[text_pad],
            enable_gqa=heads != kvheads,
        )
        return out.transpose(1, 2)[:, :n].reshape_as(q)


class CompactJointMemory(MemoryState):
    """Read-only view of the clean pass, independent of its mutable init metadata."""

    def __init__(self, clean_memory, layouts, text_lengths, device):
        if clean_memory.pass_number != 2:
            raise ValueError("compact memory requires a completed clean pass")
        self.base = clean_memory
        self.attention = CompactJointAttention(layouts, text_lengths, device)

    def init(self, hidden_states, device):
        if hidden_states["_num_causal_tokens"] or hidden_states["_num_full_tokens"] != self.attention.flat_gen_tokens:
            raise ValueError("compact forward must contain exactly the declared V/A targets")
        self.offsets = torch.tensor([0, self.attention.flat_gen_tokens], device=device, dtype=torch.int32)
        self.false = torch.tensor(False, device=device)

    def read_for_layer(self, layer_idx):
        value = self.base.read_for_layer(layer_idx)
        # UndKVCache is detached by design for inference. Training must instead use
        # the differentiable text capture made by the clean forward.
        tk, tv = self.base._joint_clean_text_kv[layer_idx]
        return dataclasses.replace(
            value,
            cached_und_k=tk,
            cached_und_v=tv,
            has_new_caption=self.false,
            gen_q_offsets=self.offsets,
            gen_attention_override=self.attention,
        )

    def write_for_layer(self, layer_idx, kv_to_store):
        pass

    def is_gen_only(self):
        return True

    def requires_natten_metadata(self):
        return False
