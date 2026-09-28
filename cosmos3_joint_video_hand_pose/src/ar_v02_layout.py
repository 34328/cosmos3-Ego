"""Explicit per-chunk U/S/V/A ownership; never infer roles from row counts."""

from dataclasses import dataclass
import torch
from .ar_chunk_state import chunk_boundaries

VIDEO, ACTION, STATE, CONDITION_VIDEO = 0, 1, 2, 3
LAYOUT_VERSION = "joint_chunk_cond_v1"


@dataclass(frozen=True)
class JointChunkLayout:
    # Original clip latent count (1 + future groups), NOT the packed video length.
    num_frames: int
    vision_tokens: int
    chunk_size: int
    action_tokens: int = 8
    chunk_state_conditioning: bool = True
    history_chunks: int = 15

    def __post_init__(self):
        chunk_boundaries(self.num_frames, self.chunk_size, tokens_per_latent=self.action_tokens)
        if self.num_frames < 2 or self.vision_tokens < 1 or self.action_tokens != 8:
            raise ValueError("joint layout requires future video, spatial patches and K=8")
        if self.history_chunks != 15:
            raise ValueError("history is exactly 15 chunks, excluding current")

    @property
    def boundaries(self):
        return chunk_boundaries(self.num_frames, self.chunk_size, tokens_per_latent=self.action_tokens)

    @property
    def state_chunks(self):
        return tuple(b.chunk_id for b in self.boundaries) if self.chunk_state_conditioning else ()

    @property
    def num_video_frames(self):
        return self.num_frames - 1 + len(self.boundaries)

    @property
    def num_action_rows(self):
        return (self.num_frames - 1) * self.action_tokens + len(self.state_chunks)

    @property
    def num_tokens(self):
        return self.num_video_frames * self.vision_tokens + self.num_action_rows

    def spans(self):
        """(role, chunk, modality payload start, count, source frame)."""
        ar = vf = 0
        for b in self.boundaries:
            yield CONDITION_VIDEO, b.chunk_id, vf * self.vision_tokens, self.vision_tokens, b.source_start
            vf += 1
            if b.chunk_id in self.state_chunks:
                yield STATE, b.chunk_id, ar, 1, b.source_start
                ar += 1
            for f in range(b.latent_start, b.latent_stop):
                yield ACTION, b.chunk_id, ar, self.action_tokens, (f - 1) * self.action_tokens + 1
                ar += self.action_tokens
                yield VIDEO, b.chunk_id, vf * self.vision_tokens, self.vision_tokens, f * self.action_tokens
                vf += 1

    def metadata(self, *, device="cpu"):
        roles, chunks, sources = [], [], []
        for role, chunk, _, count, source in self.spans():
            roles += [role] * count
            chunks += [chunk] * count
            sources += list(range(source, source + count)) if role == ACTION else [source] * count
        return tuple(torch.tensor(x, dtype=torch.long, device=device) for x in (roles, chunks, sources))

    def video_metadata(self, *, device="cpu"):
        spans = [s for s in self.spans() if s[0] in (VIDEO, CONDITION_VIDEO)]
        return tuple(torch.tensor([s[i] for s in spans], dtype=torch.long, device=device) for i in (0, 1, 4))

    def video_indexes(self, chunk, condition=None):
        r, c, _ = self.video_metadata()
        keep = c == chunk
        if condition is not None:
            keep &= r == (CONDITION_VIDEO if condition else VIDEO)
        return torch.where(keep)[0]

    def action_metadata(self, *, device="cpu"):
        r, c, s = self.metadata(device=device)
        keep = (r == STATE) | (r == ACTION)
        return r[keep], c[keep], s[keep]

    def assemble_action(self, future, boundary_states, visibility):
        if future.shape != ((self.num_frames - 1) * 8, 64):
            raise ValueError("future actions must have exact full-rate rows; no initial state")
        if boundary_states.shape != (self.num_frames - 1, 64) or visibility.shape != (future.shape[0], 2):
            raise ValueError("boundary candidates/visibility do not match original clip")
        if not torch.isfinite(future).all() or not torch.isfinite(boundary_states).all():
            raise ValueError("non-finite action/state")
        if torch.count_nonzero(future[:, 57:]) or torch.count_nonzero(boundary_states[:, 57:]):
            raise ValueError("57D padding must be zero")
        if torch.count_nonzero(boundary_states[:, :9]):
            raise ValueError("chunk-camera state camera slot must be zero in input space")
        rows, masks = [], []
        for role, _, _, count, source in self.spans():
            if role == STATE:
                rows.append(boundary_states[source // 8 : source // 8 + 1].to(future))
                masks.append(torch.zeros(1, 2, dtype=torch.bool, device=visibility.device))
            elif role == ACTION:
                rows.append(future[source - 1 : source - 1 + count])
                masks.append(visibility[source - 1 : source - 1 + count])
        return torch.cat(rows), torch.cat(masks)

    def unpack_action(self, payload):
        if payload.ndim != 2 or payload.shape[0] != self.num_action_rows:
            raise ValueError("action payload does not match explicit layout")
        r, _, s = self.action_metadata(device=payload.device)
        return payload[r == STATE], payload[r == ACTION], s[r == ACTION]

    def condition_prefill_indexes(self, chunk):
        r, c, _ = self.metadata()
        return torch.where((c == chunk) & ((r == STATE) | (r == CONDITION_VIDEO)))[0]

    def initial_prefill_indexes(self):
        return self.condition_prefill_indexes(1)


def joint_mask_mod(
    roles, chunks, *, text_pad_len, text_len, noisy, history_chunks=15, sample_ids=None, text_sample_ids=None
):
    """Keys [text|current pass|clean pass]; isolate packed samples on both paths."""
    n = roles.numel()
    if sample_ids is None:
        sample_ids = torch.where(roles >= 0, 0, -1)
    if text_sample_ids is None:
        text_sample_ids = torch.zeros(max(text_pad_len, 1), dtype=torch.long, device=roles.device)

    def mask(b, h, q, kv):
        del b, h
        is_text = kv < text_pad_len
        off = kv - text_pad_len
        cached = off >= n
        ki = torch.clamp(torch.where(cached, off - n, off), 0, n - 1)
        qr, kr, qc, kc = roles[q], roles[ki], chunks[q], chunks[ki]
        same_sample = (sample_ids[q] == sample_ids[ki]) & (sample_ids[q] >= 0)
        real = (qr >= 0) & (kr >= 0) & same_sample
        qcond = (qr == STATE) | (qr == CONDITION_VIDEO)
        kcond = (kr == STATE) | (kr == CONDITION_VIDEO)
        past = (kc < qc) & (kc >= qc - history_chunks)
        same = kc == qc
        if noisy:
            target = torch.where(cached, past | (same & kcond), same & (~kcond))
            relation = torch.where(qcond, cached & same & kcond, target)
        else:
            relation = torch.where(qcond, same & kcond, past | same)
        text_id = text_sample_ids[torch.clamp(kv, 0, text_sample_ids.numel() - 1)]
        text_ok = is_text & (kv < text_len) & (qr >= 0) & (sample_ids[q] == text_id)
        pad_ok = (~is_text) & (qr < 0) & (kr < 0) & (~cached)
        return text_ok | ((~is_text) & real & relation) | pad_ok

    return mask
