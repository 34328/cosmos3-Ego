"""Bounded persistent clean KV for explicit joint chunks (inference only)."""

import torch

from cosmos_framework.model.generator.utils.kv_cache import KVTrainMemoryValue
from cosmos_framework.model.generator.utils.memory import MemoryState

from .ar_attention import FLEX_BLOCK_SIZE, _pad_rows, _round_up
from .ar_v02_layout import ACTION, CONDITION_VIDEO, STATE, VIDEO
from .ar_v02_attention import create_joint_block_mask


class StreamingJointAttention:
    def __init__(self, query_meta, history_meta, *, cached_text_only, text_len, capturing=False):
        self.flat_gen_tokens = query_meta[0].numel()
        self.qpad = _round_up(max(self.flat_gen_tokens, 1), FLEX_BLOCK_SIZE)
        self.cached_text_only, self.text_len = cached_text_only, text_len
        self.qmeta = tuple(torch.nn.functional.pad(x, (0, self.qpad - x.numel()), value=-1) for x in query_meta)
        self.history_slots = torch.where(history_meta[2] >= 0)[0]
        hids, qids = history_meta[2][self.history_slots], query_meta[2]

        def span(ids):
            base = int(ids.min()) // FLEX_BLOCK_SIZE * FLEX_BLOCK_SIZE if ids.numel() else 0
            size = _round_up(int(ids.max()) - base + 1, FLEX_BLOCK_SIZE) if ids.numel() else FLEX_BLOCK_SIZE
            return base, size

        # Match the complete-prefix key stream and absolute tile boundaries, while
        # omitting whole expired tiles. Storage and work remain bounded by H=15.
        if capturing:
            base, self.key_len = span(torch.cat((hids, qids)))
            self.history_positions, self.query_positions = hids - base, qids - base
        else:
            qb, qlen = span(qids)
            hb, hlen = span(hids)
            self.key_len = qlen + hlen
            # Training replay reads [text, noisy GEN, clean GEN].
            self.query_positions, self.history_positions = qids - qb, qlen + hids - hb
        self.key_meta = tuple(query_meta[0].new_full((self.key_len,), -1) for _ in range(3))
        for out, h, q in zip(self.key_meta, history_meta, query_meta):
            out[self.history_positions] = h[self.history_slots]
            out[self.query_positions] = q
        self.key_current = torch.zeros(self.key_len, dtype=torch.bool, device=qids.device)
        self.key_current[self.query_positions] = True
        self.masks = {}

    def predicate(self, text_pad):
        qr, qc, qi = self.qmeta
        text_len = self.text_len
        roles, chunks, ids = self.key_meta
        current = self.key_current

        def mask(b, h, q, kv):
            del b, h
            is_text = kv < text_pad
            index = (kv - text_pad).clamp(0, len(roles) - 1)
            kr, kc, ki = roles[index], chunks[index], ids[index]
            anchor = (qr[q] == STATE) | (qr[q] == CONDITION_VIDEO)
            key_condition = (kr == STATE) | (kr == CONDITION_VIDEO)
            anchor_pair = (qc[q] == kc) & key_condition
            past = (kc < qc[q]) & (kc >= qc[q] - 15)
            same = kc == qc[q]
            in_current = current[index]
            pair = torch.where(anchor, anchor_pair, past | (same & (key_condition | in_current)))
            real = (qr[q] >= 0) & (kr >= 0)
            text = is_text & (kv < text_len) & (qr[q] >= 0)
            padding = (~is_text) & (qr[q] < 0) & (kr < 0)
            return text | ((~is_text) & real & pair) | padding

        return mask

    def assemble_gen(self, history, live):
        """Place live/history keys at their reference-aligned tile positions."""
        heads, dim = history.shape[-2:]
        keys = live.new_zeros((1, self.key_len, heads, dim))
        keys[:, self.history_positions] = history[:, self.history_slots].to(keys.dtype)
        keys[:, self.query_positions] = live.reshape(1, self.flat_gen_tokens, heads, dim)
        return keys

    def __call__(self, q, k, v, text_k, text_v, memory):
        from cosmos_framework.model.generator.mot.flex_attention import _COMPILED_FLEX_ATTENTION

        heads, dim, kvheads = q.shape[-2], q.shape[-1], k.shape[-2]
        text_pad = _round_up(max(text_k.shape[1], 1), FLEX_BLOCK_SIZE)
        if text_pad not in self.masks:
            self.masks[text_pad] = create_joint_block_mask(
                self.predicate(text_pad),
                B=None,
                H=None,
                Q_LEN=self.qpad,
                KV_LEN=text_pad + self.key_len,
                device=q.device,
                BLOCK_SIZE=FLEX_BLOCK_SIZE,
            )
        query = _pad_rows(q.reshape(1, self.flat_gen_tokens, heads, dim), self.qpad)

        def assemble(text, history, live):
            keys = self.assemble_gen(history, live)
            return torch.cat((_pad_rows(text, text_pad), keys), 1).to(query.dtype)

        key = assemble(text_k, memory.cached_gen_k, k)
        value = assemble(text_v, memory.cached_gen_v, v)
        out = _COMPILED_FLEX_ATTENTION(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            block_mask=self.masks[text_pad],
            enable_gqa=heads != kvheads,
        )
        return out.transpose(1, 2)[:, : self.flat_gen_tokens].reshape_as(q)


class JointKVCache(MemoryState):
    """One fixed-capacity cache per rollout; noisy calls never write to it.

    Text-only prefill precedes each chunk's U/S prefill, 30 read-only
    noisy V/A passes and one clean V/A refresh. Expiration uses chunk ownership,
    not action row counts. Text remains cached independently.
    """

    def __init__(self, layout, *, num_layers, num_kv_heads, head_dim, device, dtype):
        self.layout = layout
        self.layers, self.heads, self.dim = num_layers, num_kv_heads, head_dim
        self.device, self.dtype = torch.device(device), dtype
        # Fifteen complete histories plus one current U/S/V/A slot.
        self.capacity = _round_up(
            16 * (layout.vision_tokens + 1 + layout.chunk_size * (layout.vision_tokens + layout.action_tokens)),
            FLEX_BLOCK_SIZE,
        )
        self.roles = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.chunks, self.ids = self.roles.clone(), self.roles.clone()
        shape = (1, self.capacity, num_kv_heads, head_dim)
        self.kv = [
            (torch.zeros(shape, device=self.device, dtype=dtype), torch.zeros(shape, device=self.device, dtype=dtype))
            for _ in range(num_layers)
        ]
        self.text_kv = [None] * num_layers
        self.text_len = 0
        self.current_chunk = 0
        self.capture = False
        self.include_text = True
        self.attention = None
        self.forward_calls = 0
        self.all_roles, self.all_chunks, _ = layout.metadata(device=self.device)
        self.written_layers = set()
        self._pending = False
        self.text_ready = False

    def ensure_complete(self):
        if self._pending and self.written_layers != set(range(self.layers)):
            raise RuntimeError("incomplete cache capture; discard cache and restart the episode")

    def begin(self, gen_indexes, *, chunk, capture, include_text=False):
        if torch.is_grad_enabled():
            raise RuntimeError("persistent rollout cache is inference-only; use torch.no_grad()")
        self.ensure_complete()
        indexes = torch.as_tensor(gen_indexes, device=self.device, dtype=torch.long)
        if indexes.ndim != 1 or not torch.equal(indexes, indexes.unique(sorted=True)):
            raise ValueError("GEN indexes must be sorted and unique")
        if indexes.numel() and (indexes[0] < 0 or indexes[-1] >= self.layout.num_tokens):
            raise ValueError("GEN indexes outside layout")
        if include_text:
            if self.text_ready or self.text_len or indexes.numel() or chunk != 0 or not capture:
                raise ValueError("text-only prefill must occur once, with empty GEN indexes and chunk=0")
        else:
            if not self.text_ready:
                raise ValueError("cache must first complete text-only prefill")
            if chunk < max(1, self.current_chunk) or chunk > self.current_chunk + 1:
                raise ValueError("cache chunks must advance sequentially")
            if chunk > self.current_chunk and self.current_chunk:
                previous = self.all_chunks == self.current_chunk
                if not torch.isin(torch.where(previous)[0], self.ids[self.ids >= 0]).all():
                    raise ValueError("previous chunk needs clean refresh before advancing")
            conditions = self.layout.condition_prefill_indexes(chunk).to(self.device)
            targets = torch.where(
                (self.all_chunks == chunk) & ((self.all_roles == VIDEO) | (self.all_roles == ACTION))
            )[0]
            is_condition = torch.equal(indexes, conditions) and indexes.numel() > 0
            is_target = torch.equal(indexes, targets) and indexes.numel() > 0
            if not (is_condition or is_target) or (is_condition and not capture):
                raise ValueError("select complete U/S prefill or complete V/A target tokens")
            live_ids = self.ids[self.ids >= 0]
            if is_target and not torch.isin(conditions, live_ids).all():
                raise ValueError("current U/S must be prefilled before V/A")
            if torch.isin(indexes, live_ids).any():
                raise ValueError("clean tokens may be written only once and cannot be denoised after refresh")
        self.current_chunk, self.capture, self.include_text = chunk, capture, include_text
        expired = (self.roles >= 0) & (self.chunks < chunk - 15)
        self.roles[expired] = self.chunks[expired] = self.ids[expired] = -1
        self.query_meta = self.all_roles[indexes], self.all_chunks[indexes], indexes
        self.slots = torch.where(self.roles < 0)[0][: indexes.numel()]
        if capture and self.slots.numel() != indexes.numel():
            raise RuntimeError("joint KV capacity exhausted")
        self.attention = None
        self._pending = False

    def init(self, hidden_states, device):
        self.ensure_complete()
        if self.query_meta[0].numel() != hidden_states["_num_full_tokens"]:
            raise ValueError("forward GEN tokens differ from the explicit cache selection")
        if self.include_text:
            self.text_len = int(hidden_states["_num_causal_tokens"])
            if self.text_len < 1:
                raise ValueError("initial prefill requires text")
        elif hidden_states["_num_causal_tokens"] != 0:
            raise ValueError("continuation must forward only current GEN tokens, with cached text")
        self.text_pad = hidden_states["causal_seq"].shape[0]
        if self.attention is None:
            self.attention = StreamingJointAttention(
                self.query_meta,
                (self.roles.clone(), self.chunks.clone(), self.ids.clone()),
                cached_text_only=not self.include_text,
                text_len=self.text_len,
                capturing=self.capture,
            )
        self.written_layers = set()
        self._pending = self.capture
        self.forward_calls += 1

    def read_for_layer(self, layer_idx):
        text = self.text_kv[layer_idx]
        if text is None:
            z = torch.zeros(1, self.text_pad, self.heads, self.dim, device=self.device, dtype=self.dtype)
            text = z, z
        scalar = lambda x: torch.tensor(x, device=self.device)
        offsets = lambda n: torch.tensor([0, n], device=self.device, dtype=torch.int32)
        return KVTrainMemoryValue(
            vision_token_shapes=[],
            num_action_tokens_per_supertoken=0,
            has_new_caption=scalar(self.include_text),
            has_caption=scalar(True),
            has_cached_gen=scalar(True),
            und_kv_offsets=offsets(self.text_len),
            gen_q_offsets=offsets(self.query_meta[0].numel()),
            gen_ca_cached_kv_offsets=offsets(self.capacity),
            cached_und_k=text[0],
            cached_und_v=text[1],
            cached_gen_k=self.kv[layer_idx][0],
            cached_gen_v=self.kv[layer_idx][1],
            max_gen_cache_tokens=self.capacity,
            clamp_empty_varlen_kv=True,
            uses_rolling_gen_cache=False,
            gen_attention_override=self.attention,
        )

    def write_for_layer(self, layer_idx, kv_to_store):
        if not self.capture:
            return
        if layer_idx in self.written_layers:
            raise RuntimeError("duplicate cache write within one forward")
        k, v, tk, tv = kv_to_store
        n = self.query_meta[0].numel()
        self.kv[layer_idx][0][:, self.slots] = k[:, :n].to(self.dtype)
        self.kv[layer_idx][1][:, self.slots] = v[:, :n].to(self.dtype)
        if self.include_text:
            self.text_kv[layer_idx] = tk[:, : self.text_len].detach().clone(), tv[:, : self.text_len].detach().clone()
        self.written_layers.add(layer_idx)
        if len(self.written_layers) == self.layers:
            if self.include_text:
                self.text_ready = True
            self.roles[self.slots], self.chunks[self.slots], self.ids[self.slots] = self.query_meta

    def is_gen_only(self):
        return not self.include_text

    def requires_natten_metadata(self):
        return False


class BoundedJointKVCache(JointKVCache):
    """Unbounded episode length, fixed 16-slot storage, no episode layout.

    The legacy JointKVCache remains the fixed-clip/reference interface.
    Absolute token IDs and chunk IDs are counters, never allocation sizes.
    A slot is overwritten only after its owner leaves the H=15 window.
    """

    def __init__(self, *, vision_tokens, chunk_size, **kwargs):
        from .ar_v02_layout import JointChunkLayout

        if chunk_size not in (1, 2, 3, 4):
            raise ValueError("chunk_size must be C=1..4")
        super().__init__(JointChunkLayout(1 + chunk_size, vision_tokens, chunk_size), **kwargs)
        del self.layout, self.all_roles, self.all_chunks
        self.chunk_size = chunk_size
        self.slot_width = vision_tokens + 1 + chunk_size * (vision_tokens + 8)
        self.condition_count = vision_tokens + 1
        self._phase = None
        self._complete_phase = None
        self._next_token = 0
        self._chunk_token = 0
        self._frames = 0
        self._role_templates = {
            0: torch.tensor([CONDITION_VIDEO] * vision_tokens + [STATE], device=self.device),
            **{
                c: torch.tensor(([ACTION] * 8 + [VIDEO] * vision_tokens) * c, device=self.device)
                for c in range(1, chunk_size + 1)
            },
        }

    def begin(self, *args, **kwargs):
        raise TypeError("streaming cache uses begin_phase(), not full-clip GEN indexes")

    def begin_phase(self, phase, *, chunk=0, frames=0):
        if torch.is_grad_enabled():
            raise RuntimeError("persistent rollout cache is inference-only; use torch.no_grad()")
        self.ensure_complete()
        if phase == "text":
            if self.text_ready or self._phase is not None or chunk != 0:
                raise ValueError("text-only prefill must occur once per episode")
            roles = self.roles[:0]
            ids = self.ids[:0]
            slots = self.ids[:0]
        elif phase == "condition":
            if not self.text_ready or chunk != self.current_chunk + 1:
                raise ValueError("condition chunks must advance sequentially after text prefill")
            if self.current_chunk and self._complete_phase != "refresh":
                raise ValueError("previous chunk needs clean refresh before advancing")
            if not isinstance(frames, int) or not 1 <= frames <= self.chunk_size:
                raise ValueError("frames must be in 1..chunk_size")
            expired = (self.roles >= 0) & (self.chunks < chunk - 15)
            self.roles[expired] = self.chunks[expired] = self.ids[expired] = -1
            self._frames, self._chunk_token = frames, self._next_token
            roles = self._role_templates[0]
            ids = torch.arange(self._chunk_token, self._chunk_token + len(roles), device=self.device)
            start = ((chunk - 1) % 16) * self.slot_width
            slots = torch.arange(start, start + len(roles), device=self.device)
            self._next_token += self.condition_count + len(self._role_templates[frames])
        elif phase in ("noisy", "refresh"):
            if chunk != self.current_chunk or frames != self._frames:
                raise ValueError("target geometry must match the current condition")
            if self._complete_phase != "condition" or self._phase not in ("condition", "noisy"):
                raise ValueError("targets require condition prefill and cannot follow refresh")
            roles = self._role_templates[frames]
            start_id = self._chunk_token + self.condition_count
            ids = torch.arange(start_id, start_id + len(roles), device=self.device)
            start = ((chunk - 1) % 16) * self.slot_width + self.condition_count
            slots = torch.arange(start, start + len(roles), device=self.device)
        else:
            raise ValueError("unknown cache phase")
        self.current_chunk = chunk
        self.capture, self.include_text = phase != "noisy", phase == "text"
        self.query_meta = roles, torch.full_like(roles, chunk), ids
        self.slots = slots
        self.attention = None
        self._pending = False
        self._phase = phase

    def write_for_layer(self, layer_idx, kv_to_store):
        super().write_for_layer(layer_idx, kv_to_store)
        if self.capture and len(self.written_layers) == self.layers:
            self._complete_phase = self._phase

    def reset(self):
        """Discard an episode, including partial failed writes, without reallocating KV."""
        for tensor in (self.roles, self.chunks, self.ids):
            tensor.fill_(-1)
        for k, v in self.kv:
            k.zero_()
            v.zero_()
        self.text_kv = [None] * self.layers
        self.text_len = self.current_chunk = self.forward_calls = 0
        self.text_ready = self._pending = self.capture = False
        self.include_text = True
        self.attention = self._phase = self._complete_phase = None
        self._next_token = self._chunk_token = self._frames = 0
        self.written_layers = set()
