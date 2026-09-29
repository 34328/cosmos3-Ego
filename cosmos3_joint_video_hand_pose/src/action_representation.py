"""Explicit representation dispatch; dimensions alone never select semantics."""

import torch
from collections.abc import Mapping
from .ar_chunk_state import (
    decode_chunk_camera_action, decode_chunk_camera_state, encode_chunk_camera_state,
)

FIXED_CAMERA = "fixed_camera_wrist_local_delta_latent_v1"
LEGACY = "legacy_local_delta_absolute_hand_v1"


class ActionRepresentationAdapter:
    def __init__(self, state_normalizer, future_normalizer, hand_codecs=None):
        self.state_normalizer = state_normalizer
        self.future_normalizer = future_normalizer
        hand_codecs = (hand_codecs['right'], hand_codecs['left']) if isinstance(hand_codecs, Mapping) else hand_codecs
        self.hand_codecs = hand_codecs
        self.representation = getattr(state_normalizer, "representation", LEGACY)
        future_representation = getattr(future_normalizer, "representation", LEGACY)
        if self.representation != future_representation:
            raise ValueError("state and future action representations differ")
        if self.representation not in (LEGACY, FIXED_CAMERA):
            raise ValueError("unsupported action representation")
        if self.representation == FIXED_CAMERA:
            if hand_codecs is None or len(hand_codecs) != 2:
                raise ValueError("fixed-camera decoding requires two explicitly bound hand codecs")
            if getattr(state_normalizer, "kind", None) != "state" or getattr(future_normalizer, "kind", None) != "future":
                raise ValueError("state/future normalizer roles are not interchangeable")
            for codec in hand_codecs:
                if getattr(codec, "representation", None) != FIXED_CAMERA:
                    raise ValueError("hand codec representation does not match wrist-local delta actions")
            hashes = tuple(getattr(codec, "checkpoint_sha256", None) for codec in hand_codecs)
            if None in hashes or hashes != tuple(state_normalizer.codec_sha256) or hashes != tuple(future_normalizer.codec_sha256):
                raise ValueError("state, future statistics and hand codecs must bind identical hashes")
            if getattr(state_normalizer, "manifest_sha256", None) != getattr(future_normalizer, "manifest_sha256", None):
                raise ValueError("state and future statistics use different audited windows")

    def validate_state(self, state):
        from .ar_chunk_state import ChunkCameraState
        from .action_fixed_camera import FixedCameraState
        expected = FixedCameraState if self.representation == FIXED_CAMERA else ChunkCameraState
        if not isinstance(state, expected):
            raise ValueError("state type does not match the action representation")

    def validate_model(self, model):
        if getattr(model, "action_representation", LEGACY) != self.representation:
            raise ValueError("model and action representation differ")

    def encode_state(self, state):
        self.validate_state(state)
        if self.representation == LEGACY:
            return encode_chunk_camera_state(state, self.state_normalizer)
        from .action_fixed_camera import encode_state_physical
        return self.state_normalizer.normalize(encode_state_physical(state))

    def decode_state(self, encoded, *, source_index):
        if self.representation == LEGACY:
            return decode_chunk_camera_state(encoded, self.state_normalizer, source_index=source_index)
        from .action_fixed_camera import decode_state_physical
        return decode_state_physical(self.state_normalizer.denormalize(self._payload(encoded)), source_index=source_index)

    def decode_action(self, state, action):
        self.validate_state(state)
        if self.representation == LEGACY:
            return decode_chunk_camera_action(state, action, self.future_normalizer)
        from .action_fixed_camera import decode_future_physical
        for codec in self.hand_codecs:
            if isinstance(codec, torch.nn.Module):
                codec.to(device=action.device, dtype=torch.float32)
        return decode_future_physical(state, self.future_normalizer.denormalize(self._payload(action)), self.hand_codecs)

    @staticmethod
    def _payload(values):
        if values.shape[-1] not in (57, 64) or not torch.isfinite(values).all():
            raise ValueError("expected finite 57D actions or zero-padded 64D payload")
        if values.shape[-1] == 64 and torch.count_nonzero(values[..., 57:]):
            raise ValueError("action padding must be zero")
        return values[..., :57].float()
