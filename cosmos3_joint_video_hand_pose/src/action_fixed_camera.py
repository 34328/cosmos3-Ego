"""Fixed boundary-camera axes, frame deltas and full boundary hand latents.

All geometry is physical (metres/rotation columns), before independent 57D
state/future normalization. Hand codecs use current-frame wrist-local inputs.
No default codec, artifact, normalizer or legacy representation is selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import operator
import torch

from .action import pose_matrices, _pose9_matrices
from .codec_fixed_camera import REPRESENTATION, INPUT_FRAME

COORDINATE_SYSTEM = "fixed-camera"
ACTION_DIM = 57


def _finite(x, name):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be a finite floating tensor")


def _rigid(x):
    _finite(x, "rigid")
    if x.shape[-2:] != (4, 4):
        raise ValueError("rigid transforms must end in [4,4]")
    r = x[..., :3, :3]
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    if not torch.allclose(r.transpose(-1, -2) @ r, eye.expand_as(r), atol=1e-5, rtol=1e-5):
        raise ValueError("rotation must be orthogonal")
    if not torch.allclose(torch.linalg.det(r), torch.ones_like(r[..., 0, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("rotation determinant must be +1")
    if not torch.allclose(x[..., 3, :], x.new_tensor([0, 0, 0, 1]).expand_as(x[..., 3, :]), atol=1e-7, rtol=0):
        raise ValueError("invalid homogeneous row")


def _rigid_inverse(x):
    """Analytic SE(3) inverse; generic FP32 LU can corrupt the homogeneous row."""
    _rigid(x)
    result = torch.zeros_like(x)
    rotation = x[..., :3, :3].transpose(-1, -2)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = -(rotation @ x[..., :3, 3, None]).squeeze(-1)
    result[..., 3, 3] = 1
    return result


def _index(value):
    if isinstance(value, bool):
        raise ValueError("source index must be a non-negative integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("source index must be an integer") from exc
    if value < 0:
        raise ValueError("source index must be non-negative")
    return value


@dataclass(frozen=True)
class FixedCameraState:
    source_index: int
    rigid_camera: torch.Tensor  # [3,4,4]: identity camera, right, left
    hand_latents: torch.Tensor  # [2,15], COMPLETE wrist-local z, invariant to camera axes

    def __post_init__(self):
        _index(self.source_index)
        if self.rigid_camera.shape != (3, 4, 4) or self.hand_latents.shape != (2, 15):
            raise ValueError("state expects [3,4,4] rigid and [2,15] latent")
        _rigid(self.rigid_camera)
        _finite(self.hand_latents, "hand_latents")
        if self.rigid_camera.dtype not in (torch.float32, torch.float64):
            raise ValueError("physical geometry requires FP32 or FP64")
        if self.hand_latents.dtype != self.rigid_camera.dtype or self.hand_latents.device != self.rigid_camera.device:
            raise ValueError("state dtype/device must agree")
        if not torch.equal(self.rigid_camera[0], torch.eye(4, dtype=self.rigid_camera.dtype, device=self.rigid_camera.device)):
            raise ValueError("state camera must be identity")


@dataclass(frozen=True)
class EncodedFixedCameraChunk:
    state: FixedCameraState
    actions: torch.Tensor
    source_indices: torch.Tensor  # includes boundary, all original frames


@dataclass(frozen=True)
class DecodedFixedCameraChunk:
    rigid_chunk: torch.Tensor
    wrists_camera: torch.Tensor
    hand_latents: torch.Tensor  # accumulated wrist-local z, NOT deltas
    keypoints_chunk: torch.Tensor  # [T,2,21,3], includes wrist
    end_state: FixedCameraState  # rigid in final camera axes, z copied unchanged
    source_indices: torch.Tensor  # future frames only

    @property
    def wrist_camera(self):
        """Compatibility with the original decoder's singular property name."""
        return self.wrists_camera


def _codecs(codecs):
    pair = (codecs["right"], codecs["left"]) if isinstance(codecs, Mapping) else tuple(codecs)
    if len(pair) != 2:
        raise ValueError("explicit right and left codecs required")
    for c in pair:
        if getattr(c, "coordinate_system", None) != COORDINATE_SYSTEM:
            raise ValueError("codec must explicitly declare coordinate_system='fixed-camera'")
        if getattr(c, "representation", None) != REPRESENTATION or getattr(c, "input_frame", None) != INPUT_FRAME:
            raise ValueError("codec must declare wrist-local representation and input_frame; old camera-axis codecs are incompatible")
        if not callable(getattr(c, "encode", None)) or not callable(getattr(c, "decode", None)):
            raise ValueError("codec requires encode and decode")
    return pair


def _codec_call(c, method, x, shape):
    y = getattr(c, method)(x)
    _finite(y, "codec output")
    if y.shape != shape or y.dtype != x.dtype or y.device != x.device:
        raise ValueError(f"codec output must be {shape}, with input dtype/device")
    return y


def _encode_hands(q, codecs):
    return torch.stack([_codec_call(c, "encode", q[..., h, :, :], q.shape[:-3] + (15,))
                        for h, c in enumerate(_codecs(codecs))], dim=-2)


def _decode_hands(z, codecs):
    return torch.stack([_codec_call(c, "decode", z[..., h, :], z.shape[:-2] + (20, 3))
                        for h, c in enumerate(_codecs(codecs))], dim=-3)


def _pose9(x):
    return torch.cat((x[..., :3, 3], x[..., :3, 0], x[..., :3, 1]), -1)


def _assemble(rigid9, z):
    return torch.cat((rigid9[..., 0, :], rigid9[..., 1, :], z[..., 0, :],
                      rigid9[..., 2, :], z[..., 1, :]), -1)


def _values(values):
    _finite(values, "action/state")
    if values.shape[-1] not in (57, 64):
        raise ValueError("expected 57D or zero-padded 64D")
    if values.shape[-1] == 64 and torch.count_nonzero(values[..., 57:]):
        raise ValueError("padding must be zero")
    return values[..., :57]


def pad_action(values):
    values = _values(values)
    return torch.nn.functional.pad(values, (0, 7))


def _split(values):
    v = _values(values)
    p = torch.stack((v[..., :9], v[..., 9:18], v[..., 33:42]), -2)
    z = torch.stack((v[..., 18:33], v[..., 42:57]), -2)
    return p, z


def _matrices(p):
    # Reject degenerate 6D rotations rather than silently constructing zero axes.
    a, b = p[..., 3:6], p[..., 6:9]
    if (torch.linalg.vector_norm(a, dim=-1) < 1e-8).any() or (torch.linalg.vector_norm(torch.linalg.cross(a, b), dim=-1) < 1e-8).any():
        raise ValueError("degenerate rotation6D")
    return _pose9_matrices(p.reshape(-1, 9)).reshape(p.shape[:-1] + (4, 4))


def encode_state_physical(state):
    p = _pose9(state.rigid_camera).clone()
    p[0] = 0
    return _assemble(p, state.hand_latents)


def decode_state_physical(values, source_index):
    v = _values(values)
    if v.shape != (57,) or torch.count_nonzero(v[:9]):
        raise ValueError("state must be a vector with zero camera slot")
    p, z = _split(v)
    identity = torch.eye(4, dtype=v.dtype, device=v.device)
    return FixedCameraState(_index(source_index), torch.cat((identity[None], _matrices(p[1:]))), z.clone())


def _times(indices, length, device, start=None):
    v = torch.as_tensor(indices, device=device)
    if v.shape != (length,) or v.dtype == torch.bool or v.is_floating_point():
        raise ValueError("source_indices must contain integer frame indices")
    v = v.to(torch.int64)
    if (v < 0).any() or (length > 1 and not torch.all(v[1:] - v[:-1] == 1)):
        raise ValueError("all original consecutive action frames are required")
    if start is not None and length and v[0].item() != start:
        raise ValueError("future time must begin at boundary+1")
    return v


def _stream(x):
    """Reject missing/invalid frames; callers must exclude invalid windows.

    Never fill absent hands with identity or drop frames: both would fabricate
    valid-looking deltas across missing labels.
    """
    x = torch.as_tensor(x)
    _finite(x, "pose stream")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("pose stream requires FP32 or FP64")
    if not ((x.ndim == 2 and x.shape[-1] == 7) or
            (x.ndim == 3 and x.shape[-2:] == (4, 4))):
        raise ValueError("pose stream requires [N,7] or [N,4,4]")
    if x.ndim == 2 and x.shape[-1] == 7:
        if not torch.isfinite(x).all() or not torch.allclose(torch.linalg.vector_norm(x[:, 3:], dim=-1), torch.ones_like(x[:, 0]), atol=1e-4, rtol=1e-4):
            raise ValueError("invalid pose quaternion")
        x = torch.as_tensor(pose_matrices(x.cpu().numpy()), dtype=x.dtype, device=x.device)
    _rigid(x)
    return x


def encode_chunk_physical(head, wrists, keypoints, source_indices, codecs):
    """Inputs: head [N,4,4] or [N,7], wrists [N,2,4,4] or [N,2,7],
    keypoints [N,2,21,3] in the SAME world frame. N includes boundary.
    Returns N-1 future actions. q uses authoritative wrist-pose translation.
    Codecs operate on [...,20,3] <-> [...,15], preserving dtype/device.
    """
    _codecs(codecs)
    h = _stream(head)
    w = torch.as_tensor(wrists, dtype=h.dtype, device=h.device)
    if w.shape not in ((len(h), 2, 7), (len(h), 2, 4, 4)):
        raise ValueError("wrists require [N,2,7] or [N,2,4,4]")
    w = torch.stack((_stream(w[:, 0]), _stream(w[:, 1])), dim=1)
    n = len(h)
    k = torch.as_tensor(keypoints, dtype=h.dtype, device=h.device)
    if n < 2 or h.shape != (n, 4, 4) or w.shape != (n, 2, 4, 4) or k.shape != (n, 2, 21, 3):
        raise ValueError("chunk requires boundary plus future frames with matching shapes")
    _finite(k, "keypoints")
    times = _times(source_indices, n, h.device)
    base_inv = _rigid_inverse(h[0])
    rigid = base_inv @ torch.cat((h[:, None], w), dim=1)
    rigid[0, 0] = torch.eye(4, dtype=h.dtype, device=h.device)
    points = torch.einsum("ij,thnj->thni", base_inv[:3, :3], k) + base_inv[:3, 3]
    offsets = points[:, :, 1:] - rigid[:, 1:, None, :3, 3]
    q = torch.einsum("thji,thnj->thni", rigid[:, 1:, :3, :3], offsets)
    z = _encode_hands(q, codecs)
    delta = torch.eye(4, dtype=h.dtype, device=h.device).expand(n-1, 3, 4, 4).clone()
    delta[..., :3, 3] = rigid[1:, :, :3, 3] - rigid[:-1, :, :3, 3]
    delta[..., :3, :3] = rigid[1:, :, :3, :3] @ rigid[:-1, :, :3, :3].transpose(-1, -2)
    return EncodedFixedCameraChunk(FixedCameraState(int(times[0]), rigid[0], z[0]),
                                   _assemble(_pose9(delta), z[1:]-z[:-1]), times)


def reanchor_state(rigid_chunk, hand_latents, source_index, codecs):
    """Change rigid axes only; wrist-local z is copied with no E/D round trip."""
    _codecs(codecs)
    _rigid(rigid_chunk)
    inv = _rigid_inverse(rigid_chunk[0])
    rigid = inv @ rigid_chunk
    # Return proper rotations after finite-precision frame changes. Do not
    # relax input validation or allow drift to accumulate across a rollout.
    rigid = _matrices(_pose9(rigid))
    rigid[0] = torch.eye(4, dtype=rigid.dtype, device=rigid.device)
    return FixedCameraState(source_index, rigid, hand_latents.clone())


def decode_future_physical(state, actions, codecs, source_indices=None):
    v = _values(actions)
    if v.ndim != 2 or len(v) < 1:
        raise ValueError("future actions require nonempty [T,57/64]")
    if v.dtype != state.rigid_camera.dtype or v.device != state.rigid_camera.device:
        raise ValueError("future and state dtype/device must agree")
    t = len(v)
    times = _times(range(state.source_index+1, state.source_index+t+1) if source_indices is None else source_indices,
                   t, v.device, state.source_index+1)
    p, dz = _split(v)
    delta = _matrices(p)
    previous = state.rigid_camera
    result = []
    for row in delta:
        current = previous.clone()
        current[:, :3, 3] = previous[:, :3, 3] + row[:, :3, 3]
        current[:, :3, :3] = row[:, :3, :3] @ previous[:, :3, :3]
        # Gram-Schmidt via the shared rotation6D decoder removes round-off
        # accumulation without changing left-composition or translation.
        current = _matrices(_pose9(current))
        result.append(current)
        previous = current
    rigid = torch.stack(result)
    z = state.hand_latents[None] + dz.cumsum(0)
    q = _decode_hands(z, codecs)
    wrist_p = rigid[:, 1:, :3, 3]
    offsets = torch.einsum("thij,thnj->thni", rigid[:, 1:, :3, :3], q)
    points = torch.cat((wrist_p[:, :, None], wrist_p[:, :, None] + offsets), dim=2)
    wrists_camera = _rigid_inverse(rigid[:, 0])[:, None] @ rigid[:, 1:]
    end = reanchor_state(rigid[-1], z[-1], int(times[-1]), codecs)
    return DecodedFixedCameraChunk(rigid, wrists_camera, z, points, end, times)


def _normalize(values, normalizer, inverse, state):
    v = _values(values)
    result = getattr(normalizer, "denormalize" if inverse else "normalize")(v)
    _finite(result, "normalizer output")
    if result.shape != v.shape:
        raise ValueError("normalizer must accept and return 57D")
    if state:
        result = result.clone()
        result[..., :9] = 0  # structural conditioning sentinel, not a random variable
    return result


def normalize_state(values, normalizer):
    if torch.count_nonzero(_values(values)[..., :9]):
        raise ValueError("physical state camera must be zero")
    return _normalize(values, normalizer, False, True)


def denormalize_state(values, normalizer):
    if torch.count_nonzero(_values(values)[..., :9]):
        raise ValueError("normalized state camera must be zero")
    return _normalize(values, normalizer, True, True)


def normalize_future(values, normalizer):
    return _normalize(values, normalizer, False, False)


def denormalize_future(values, normalizer):
    return _normalize(values, normalizer, True, False)
