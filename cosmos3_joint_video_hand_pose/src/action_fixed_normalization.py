"""Versioned, train-only normalization for fixed-camera 57D action/state.

No codec or model is held here. Codec hashes are ordered (right, left).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

REPRESENTATION = "fixed_camera_wrist_local_delta_latent_v1"
NORMALIZER_SCHEMA = "fixed_camera57_normalizer_v1"
VALID_WINDOWS_SCHEMA = "fixed_camera57_valid_windows_v1"


def load_fixed_codecs(paths):
    """Load explicitly validated artifacts; no unversioned wrist-local fallback."""
    from .codec_fixed_camera import FrozenFixedCameraHandCodec
    if len(paths) != 2 or any(p is None for p in paths):
        raise ValueError("explicit right/left fixed-camera codec paths required")
    pair = tuple(FrozenFixedCameraHandCodec(p) for p in paths)
    for codec, path, side in zip(pair, paths, ("right", "left")):
        if codec.metadata.get("side") != side:
            raise ValueError(f"fixed-camera codec side mismatch: expected {side}, got {codec.metadata.get('side')!r}")
        codec.checkpoint_path = str(Path(path).resolve())
    return pair


def encode_fixed_window(head, right, left, right_points, left_points, codecs):
    """Physical C=4 K=8 targets plus all candidate states for the packer.

    Candidate states every eight frames preserve the existing layout index.
    Only every fourth candidate is used for training/statistics. A final short
    chunk retains all source actions; no state is inserted in the action rows.
    """
    from .action_fixed_camera import encode_chunk_physical, encode_state_physical
    h = torch.as_tensor(head, dtype=torch.float32)
    w = torch.as_tensor(np.stack((right, left), axis=1), dtype=torch.float32)
    # Storage may expose [T,63] or [T,21,3]; tracking audit accepts both.
    points = [np.asarray(v) for v in (right_points, left_points)]
    if any(len(v) != len(h) or v.reshape(len(v), -1).shape[1] != 63 for v in points):
        raise ValueError("fixed window requires 21 XYZ keypoints per hand/frame")
    p = torch.as_tensor(np.stack([v.reshape(len(h),21,3) for v in points], axis=1), dtype=torch.float32)
    n = len(h)
    if n < 9 or (n - 1) % 8 or len(w) != n or len(p) != n:
        raise ValueError("fixed window requires 1+8N source frames (K=8)")
    states, actions = [], []
    with torch.no_grad():
        for b in range(0, n - 1, 8):
            end = min(b + 32, n - 1) if b % 32 == 0 else b + 1
            item = encode_chunk_physical(h[b:end+1], w[b:end+1], p[b:end+1],
                                         torch.arange(b, end+1), codecs)
            states.append(encode_state_physical(item.state))
            if b % 32 == 0:
                actions.append(item.actions)
    future = torch.cat(actions)
    if future.shape != (n - 1, 57):
        raise AssertionError("fixed chunk encoding lost or duplicated future actions")
    return torch.stack(states), future


def profile_sha256(profile):
    data = {k: v for k, v in profile.items() if k != "profile_sha256"}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class FixedCameraNormalizer:
    representation = REPRESENTATION
    schema = NORMALIZER_SCHEMA

    def __init__(self, profile, *, kind=None, codec_sha256=None):
        self.path = Path(profile).resolve() if isinstance(profile, (str, Path)) else None
        self.file_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path else None
        self.profile = json.loads(Path(profile).read_text()) if isinstance(profile, (str, Path)) else dict(profile)
        p = self.profile
        if p.get("schema") != self.schema or p.get("representation") != self.representation:
            raise ValueError("fixed-camera normalizer requires explicit new representation/schema; legacy statistics rejected")
        self.kind = p.get("kind")
        if self.kind not in ("state", "future") or (kind is not None and self.kind != kind):
            raise ValueError("normalizer state/future kind mismatch")
        if p.get("chunk_size") != 4 or p.get("tokens_per_latent") != 8:
            raise ValueError("fixed-camera normalizer requires C=4 K=8")
        if p.get("split") != "train" or p.get("frozen") is not True:
            raise ValueError("normalizer must contain frozen train-only statistics")
        hashes = p.get("codec_sha256")
        if not isinstance(hashes, (tuple, list)) or len(hashes) != 2 or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
            raise ValueError("normalizer requires right/left codec SHA256 identities")
        self.codec_sha256 = tuple(hashes)
        if codec_sha256 is not None and tuple(codec_sha256) != self.codec_sha256:
            raise ValueError("normalizer codec identity mismatch")
        if not isinstance(p.get("manifest_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", p["manifest_sha256"]):
            raise ValueError("normalizer requires audited manifest SHA256")
        self.manifest_sha256 = p["manifest_sha256"]
        if p.get("method") != "piecewise_asinh_per_channel_v1" or p.get("profile_sha256") != profile_sha256(p):
            raise ValueError("normalizer method or profile integrity mismatch")
        self.beta = float(p.get("beta", 0))
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("piecewise-asinh beta must be positive")
        self.center = torch.as_tensor(p["stats"]["center"], dtype=torch.float64)
        self.scale = torch.as_tensor(p["stats"]["scale"], dtype=torch.float64)
        if self.center.shape != (57,) or self.scale.shape != (57,) or not torch.isfinite(self.center).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("normalizer requires finite 57D centers and positive scales")
        if self.kind == "state" and (self.center[:9].count_nonzero() or not torch.equal(self.scale[:9], torch.ones(9, dtype=torch.float64))):
            raise ValueError("state camera slots require center=0 scale=1")
        floors = torch.as_tensor(p.get("scale_floor", []), dtype=torch.float64)
        if floors.shape != (57,) or not torch.isfinite(floors).all() or (floors <= 0).any() or (self.scale < floors).any():
            raise ValueError("normalizer requires explicit positive per-unit floors")

    def _apply(self, value, inverse):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.shape[-1:] != (57,):
            raise ValueError("normalizer input must be a floating tensor [...,57]")
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite action/state")
        if self.kind == "state" and value[..., :9].count_nonzero():
            raise ValueError("state camera slots must be exactly zero")
        center, scale = self.center.to(value), self.scale.to(value)
        beta = value.new_tensor(self.beta)
        if inverse:
            tail = 1 + torch.sinh(beta * (value.abs()-1).clamp_min(0)) / beta
            z = torch.where(value.abs() <= 1, value, value.sign()*tail)
            result = z * scale + center
        else:
            z = (value - center) / scale
            tail = 1 + torch.asinh(beta * (z.abs()-1).clamp_min(0)) / beta
            result = torch.where(z.abs() <= 1, z, z.sign()*tail)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite piecewise-asinh output")
        if self.kind == "state":
            result = torch.cat((torch.zeros_like(result[..., :9]), result[..., 9:]), -1)
        return result

    def normalize(self, value):
        return self._apply(value, False)

    def denormalize(self, value):
        return self._apply(value, True)


def fit_fixed_normalizer(values, *, kind, codec_sha256, manifest_sha256,
                         translation_floor=0.01, rotation_floor=0.05, latent_floor=0.01, beta=1.0):
    """Fit physical state or future rows separately, after C=4 chunk encoding.

    Quantile center/half-range includes latent deltas. No variance equalization
    or extra calibration is applied. Floors retain the old physical-pose units;
    latent floor 0.01 is in codec-standardized z units (limits linear gain to 100).
    Caller supplies train rows only; metadata is deliberately explicit.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 57 or not len(x) or not np.isfinite(x).all():
        raise ValueError("fit requires finite nonempty [N,57] rows")
    if kind not in ("state", "future"):
        raise ValueError("normalizer kind must be state/future")
    if any(not np.isfinite(f) or f <= 0 for f in (translation_floor, rotation_floor, latent_floor, beta)):
        raise ValueError("unit floors and beta must be positive")
    if kind == "state" and np.count_nonzero(x[:, :9]):
        raise ValueError("physical state camera slots must be zero")
    q01, q99 = np.quantile(x, [0.01, 0.99], axis=0)
    center, raw_scale = (q01+q99)/2, (q99-q01)/2
    floors = np.full(57, latent_floor)
    for start in (0, 9, 33):
        floors[start:start+3] = translation_floor
        floors[start+3:start+9] = rotation_floor
    scale = np.maximum(raw_scale, floors)
    if kind == "state":
        center[:9], scale[:9], floors[:9] = 0, 1, 1
    active = np.arange(9 if kind == "state" else 0, 57)
    profile = dict(schema=NORMALIZER_SCHEMA, representation=REPRESENTATION, kind=kind,
                   chunk_size=4, tokens_per_latent=8, split="train", frozen=True,
                   codec_sha256=list(codec_sha256), manifest_sha256=manifest_sha256,
                   method="piecewise_asinh_per_channel_v1", beta=float(beta),
                   scale_floor=floors.tolist(), count=len(x),
                   floor_units=dict(translation="metres", rotation="dimensionless rotation-column entries",
                                    latent="codec-standardized z" if kind == "state" else "delta of codec-standardized z"),
                   floor_reason="pose floors preserve 1cm/0.05 protection; latent 0.01 limits near-constant-channel gain to 100; no extra calibration",
                   floor_channels=active[raw_scale[active] < floors[active]].tolist(),
                   near_constant_channels=active[np.ptp(x, axis=0)[active] < 1e-8].tolist(),
                   stats=dict(center=center.tolist(), scale=scale.tolist(), q01=q01.tolist(), q99=q99.tolist(),
                              minimum=x.min(0).tolist(), maximum=x.max(0).tolist()))
    profile["profile_sha256"] = profile_sha256(profile)
    FixedCameraNormalizer(profile, kind=kind)
    return profile
