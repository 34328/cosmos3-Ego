"""AR boundary states: legacy F0 and versioned per-chunk camera coordinates.

The model-facing state is separate from future action rows. These helpers do
not change the v0.1 dataset, packer or sampler. The state normalizer is explicit:
the old identity-camera statistics must not be used implicitly for later states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .action import _pose9_matrices, pose_matrices
from .normalization import PiecewiseAsinhNormalizer


@dataclass(frozen=True)
class ChunkBoundary:
    chunk_id: int
    latent_start: int
    latent_stop: int
    source_start: int
    source_stop: int

    @property
    def action_count(self) -> int:
        """Predict (source_start, source_stop], excluding the boundary itself."""
        return self.source_stop - self.source_start


def chunk_boundaries(
    num_latent_frames: int, chunk_size: int, *, tokens_per_latent: int = 8
) -> tuple[ChunkBoundary, ...]:
    """Clip-relative source indexes for a [V0, C, C, ...] partition."""
    for name, value in (
        ("num_latent_frames", num_latent_frames),
        ("chunk_size", chunk_size),
        ("tokens_per_latent", tokens_per_latent),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return tuple(
        ChunkBoundary(
            chunk_id=index,
            latent_start=start,
            latent_stop=min(start + chunk_size, num_latent_frames),
            source_start=(start - 1) * tokens_per_latent,
            source_stop=(min(start + chunk_size, num_latent_frames) - 1) * tokens_per_latent,
        )
        for index, start in enumerate(range(1, num_latent_frames, chunk_size), start=1)
    )


def state_boundary_indices(boundaries: tuple[ChunkBoundary, ...], *, chunk_state_conditioning: bool) -> tuple[int, ...]:
    """The ablation removes later state tokens; it does not replace them by zero."""
    selected = boundaries if chunk_state_conditioning else boundaries[:1]
    return tuple(boundary.source_start for boundary in selected)


@dataclass(frozen=True)
class BoundaryState:
    """Physical camera/right/left transforms plus standardized hand AE latents."""

    source_index: int
    rigid_f0: torch.Tensor  # [3, 4, 4], FP32, same F0 for the complete rollout
    hand_latents: torch.Tensor  # [2, 15], existing AE standardized space

    def __post_init__(self) -> None:
        if self.source_index < 0 or self.rigid_f0.shape != (3, 4, 4) or self.hand_latents.shape != (2, 15):
            raise ValueError("boundary state requires a non-negative index, [3,4,4] rigid poses and [2,15] hands")
        if self.rigid_f0.dtype != torch.float32 or self.hand_latents.dtype != torch.float32:
            raise ValueError("physical boundary state must be retained in FP32")
        if self.rigid_f0.device != self.hand_latents.device:
            raise ValueError("boundary poses and hand latents must share a device")
        if not torch.isfinite(self.rigid_f0).all() or not torch.isfinite(self.hand_latents).all():
            raise ValueError("non-finite boundary state")
        rotation = self.rigid_f0[:, :3, :3]
        identity = torch.eye(3, device=rotation.device).expand(3, -1, -1)
        bottom = self.rigid_f0.new_tensor([0, 0, 0, 1]).expand(3, -1)
        if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4, rtol=1e-4):
            raise ValueError("boundary rotation is not orthogonal")
        if not torch.allclose(torch.linalg.det(rotation), rotation.new_ones(3), atol=1e-4, rtol=1e-4):
            raise ValueError("boundary rotation must have determinant +1")
        if not torch.allclose(self.rigid_f0[:, 3], bottom, atol=1e-6, rtol=0):
            raise ValueError("invalid homogeneous boundary transform")


def _pose9_tensor(matrices: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrices[..., :3, 3], matrices[..., :3, 0], matrices[..., :3, 1]), dim=-1)


def _normalized_pose27(values: torch.Tensor) -> torch.Tensor:
    return torch.cat((values[..., :18], values[..., 33:42]), dim=-1)


def _assemble57(pose27: torch.Tensor, hands: torch.Tensor) -> torch.Tensor:
    return torch.cat((pose27[..., :18], hands[..., 0, :], pose27[..., 18:27], hands[..., 1, :]), dim=-1)


@torch.no_grad()
def boundary_states_from_streams(
    *,
    head_pose: np.ndarray,
    right_wrist_pose: np.ndarray,
    left_wrist_pose: np.ndarray,
    hand_latents: torch.Tensor,
    boundaries: tuple[ChunkBoundary, ...],
    chunk_state_conditioning: bool = True,
) -> tuple[BoundaryState, ...]:
    """Use GT at clip-relative boundaries, never re-anchor the camera per chunk.

    hand_latents contains the existing frozen codec outputs, [source_frames,2,15].
    The dataset must supply pose streams beginning at its selected window start.
    """
    streams = (head_pose, right_wrist_pose, left_wrist_pose)
    length = len(head_pose)
    if length < 1 or any(len(stream) != length for stream in streams) or hand_latents.shape != (length, 2, 15):
        raise ValueError("pose streams and hand latents must have matching source-frame lengths")
    if boundaries and boundaries[-1].source_stop >= length:
        raise ValueError("chunk extends beyond source streams")
    matrices = np.stack([pose_matrices(stream) for stream in streams], axis=1)
    f0_from_world = np.linalg.inv(matrices[0, 0])
    indexes = state_boundary_indices(boundaries, chunk_state_conditioning=chunk_state_conditioning)
    return tuple(
        BoundaryState(
            source_index=index,
            rigid_f0=torch.as_tensor(f0_from_world @ matrices[index], dtype=torch.float32, device=hand_latents.device),
            hand_latents=hand_latents[index].detach().float().clone(),
        )
        for index in indexes
    )


@torch.no_grad()
def encode_boundary_state(state: BoundaryState, state_normalizer: PiecewiseAsinhNormalizer) -> torch.Tensor:
    """57D model condition; padding and type embeddings belong to the packer."""
    pose27 = _pose9_tensor(state.rigid_f0).reshape(27)
    encoded = _assemble57(state_normalizer.normalize(pose27), state.hand_latents)
    if not torch.isfinite(encoded).all():
        raise ValueError("non-finite normalized boundary state")
    return encoded


@torch.no_grad()
def decode_boundary_state(
    encoded: torch.Tensor, state_normalizer: PiecewiseAsinhNormalizer, *, source_index: int
) -> BoundaryState:
    """Decode a state condition explicitly; never treat it as a future increment."""
    if encoded.shape not in ((57,), (64,)) or not torch.isfinite(encoded).all():
        raise ValueError("encoded state must be a finite 57D or zero-padded 64D vector")
    if encoded.numel() == 64 and torch.count_nonzero(encoded[57:]):
        raise ValueError("state padding must be zero")
    pose27 = state_normalizer.denormalize(_normalized_pose27(encoded.float()))
    return BoundaryState(
        source_index, _pose9_matrices(pose27.reshape(3, 9)), torch.stack((encoded[18:33], encoded[42:57])).float()
    )


@dataclass(frozen=True)
class DecodedActionChunk:
    rigid_f0: torch.Tensor  # [future_rows,3,4,4], excludes input boundary
    hand_latents: torch.Tensor  # [future_rows,2,15]
    end_state: BoundaryState


@torch.no_grad()
def decode_action_chunk(
    state: BoundaryState,
    future_action: torch.Tensor,
    future_normalizer: PiecewiseAsinhNormalizer,
) -> DecodedActionChunk:
    """Integrate every future row from an explicit physical boundary.

    Accept existing normalized future 57D values (or zero-padded 64D). A caller
    using optional D calibration must invert it first. No GT is read here.
    """
    if future_action.ndim != 2 or future_action.shape[0] < 1 or future_action.shape[1] not in (57, 64):
        raise ValueError("future_action must be non-empty [T,57] or [T,64]")
    if future_action.device != state.rigid_f0.device or not torch.isfinite(future_action).all():
        raise ValueError("future action must be finite and on the boundary state's device")
    if future_action.shape[1] == 64 and torch.count_nonzero(future_action[:, 57:]):
        raise ValueError("action padding must be zero")
    values = future_action.float()
    pose27 = future_normalizer.denormalize(_normalized_pose27(values))
    if not torch.isfinite(pose27).all():
        raise ValueError("non-finite denormalized action")
    pose9 = pose27.reshape(-1, 9)
    first = pose9[:, 3:6]
    second = pose9[:, 6:9]
    first_norm = torch.linalg.vector_norm(first, dim=-1)
    unit_first = first / first_norm.clamp_min(1e-8)[:, None]
    residual = second - (unit_first * second).sum(-1, keepdim=True) * unit_first
    if (first_norm < 1e-8).any() or (torch.linalg.vector_norm(residual, dim=-1) < 1e-8).any():
        raise ValueError("degenerate predicted 6D rotation")
    increments = _pose9_matrices(pose9).reshape(-1, 3, 4, 4)
    current = state.rigid_f0.clone()
    poses = []
    for increment in increments:
        current = current @ increment
        poses.append(current)
    rigid_f0 = torch.stack(poses)
    hands = torch.stack((values[:, 18:33], values[:, 42:57]), dim=1).clone()
    end_state = BoundaryState(state.source_index + len(values), rigid_f0[-1].clone(), hands[-1].clone())
    return DecodedActionChunk(rigid_f0, hands, end_state)


CHUNK_CAMERA_STATE_SCHEMA = "ar_v02_chunk_camera_state_v2"
CHUNK_CAMERA_LAYOUT_VERSION = "joint_chunk_cond_v1"
VALID_WINDOWS_SCHEMA = "ar_v02_valid_windows_v2"


def state_profile_sha256(profile: dict) -> str:
    """Canonical profile digest, excluding only the digest field itself."""
    import hashlib
    import json

    payload = {key: value for key, value in profile.items() if key != "profile_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class ChunkCameraStateNormalizer(PiecewiseAsinhNormalizer):
    """PiecewiseAsinh transform fitted only on right/left wrist pose9 (18D).

    Camera identity is not a fitted channel. This separate loader deliberately
    rejects old 27D/F0 profiles; the legacy PiecewiseAsinhNormalizer is unchanged.
    expected_sha256 is the *file* digest saved by a checkpoint, when available.
    """

    def __init__(self, path, *, expected_sha256=None, expected_manifest_sha256=None, require_frozen=True):
        import hashlib
        import json
        from pathlib import Path

        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        profile = json.loads(raw)
        self.file_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and self.file_sha256 != expected_sha256:
            raise ValueError("state normalizer file hash mismatch")
        if profile.get("schema") != CHUNK_CAMERA_STATE_SCHEMA:
            raise ValueError("chunk-camera state requires the 18D v2 schema")
        if profile.get("layout_version") != CHUNK_CAMERA_LAYOUT_VERSION or profile.get("split") != "train":
            raise ValueError("chunk-camera state requires train-only joint_chunk_cond_v1 statistics")
        if require_frozen and profile.get("frozen") is not True:
            raise ValueError("chunk-camera state statistics must be frozen")
        if profile.get("profile_sha256") != state_profile_sha256(profile):
            raise ValueError("state normalizer profile hash mismatch")
        self.manifest_sha256 = profile.get("manifest_sha256", "")
        for key in ("manifest_sha256", "fit_samples_sha256"):
            digest = profile.get(key, "")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"invalid {key}")
        if expected_manifest_sha256 is not None and self.manifest_sha256 != expected_manifest_sha256:
            raise ValueError("state normalizer manifest hash mismatch")
        if profile.get("method") != "piecewise_asinh_rot" or profile.get("camera_encoding") != "zero_identity":
            raise ValueError("invalid chunk-camera normalization method or camera encoding")
        self.beta = float(profile.get("beta", 1.0))
        self.center = torch.tensor(profile["stats"]["center"], dtype=torch.float32)
        self.scale = torch.tensor(profile["stats"]["scale"], dtype=torch.float32)
        if self.center.shape != (18,) or self.scale.shape != (18,):
            raise ValueError("chunk-camera normalizer must contain 18 wrist channels")
        if (
            not np.isfinite(self.beta)
            or self.beta <= 0
            or not torch.isfinite(self.center).all()
            or not torch.isfinite(self.scale).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError("invalid chunk-camera normalization statistics")
        self.schema = CHUNK_CAMERA_STATE_SCHEMA
        self.profile_sha256 = profile["profile_sha256"]

    def normalize(self, values):
        if values.shape[-1] != 18:
            raise ValueError("chunk-camera normalization expects 18 wrist channels")
        return super().normalize(values)

    def denormalize(self, values):
        if values.shape[-1] != 18:
            raise ValueError("chunk-camera normalization expects 18 wrist channels")
        return super().denormalize(values)


@dataclass(frozen=True)
class ChunkCameraState:
    """FP32 poses at one boundary, expressed in that boundary's own camera.

    source_index uses the caller's index space (clip-relative for stream builders).
    rigid_camera is [camera identity, right wrist, left wrist]; no F0 is retained.
    """

    source_index: int
    rigid_camera: torch.Tensor
    hand_latents: torch.Tensor

    def __post_init__(self):
        BoundaryState(self.source_index, self.rigid_camera, self.hand_latents)
        if not torch.equal(self.rigid_camera[0], torch.eye(4, device=self.rigid_camera.device)):
            raise ValueError("chunk-camera physical camera must be identity")


def _validated_pose_streams(head_pose, right_wrist_pose, left_wrist_pose, hand_latents):
    streams = tuple(np.asarray(p) for p in (head_pose, right_wrist_pose, left_wrist_pose))
    length = len(streams[0])
    if length < 1 or any(p.shape != (length, 7) for p in streams) or hand_latents.shape != (length, 2, 15):
        raise ValueError("pose streams and hand latents must have matching source-frame lengths")
    for p in streams:
        if not np.isfinite(p).all() or (np.abs(np.linalg.norm(p[:, 3:], axis=1) - 1) > 1e-3).any():
            raise ValueError("invalid pose stream: nonfinite values or non-unit quaternion")
    if not torch.isfinite(hand_latents).all():
        raise ValueError("non-finite hand latents")
    return np.stack([pose_matrices(p) for p in streams], axis=1)


@torch.no_grad()
def chunk_camera_states_from_streams(
    *, head_pose, right_wrist_pose, left_wrist_pose, hand_latents, boundaries, chunk_state_conditioning=True
):
    """Build each candidate directly from H[b]^-1 @ wrist[b], independent of C."""
    matrices = _validated_pose_streams(head_pose, right_wrist_pose, left_wrist_pose, hand_latents)
    if any(b.source_start < 0 or b.source_stop >= len(matrices) or b.source_start >= b.source_stop for b in boundaries):
        raise ValueError("chunk extends beyond source streams or has invalid bounds")
    result = []
    for index in state_boundary_indices(boundaries, chunk_state_conditioning=chunk_state_conditioning):
        rigid = np.linalg.inv(matrices[index, 0]) @ matrices[index]
        rigid[0] = np.eye(4)
        result.append(
            ChunkCameraState(
                index,
                torch.as_tensor(rigid, dtype=torch.float32, device=hand_latents.device),
                hand_latents[index].detach().float().clone(),
            )
        )
    return tuple(result)


@torch.no_grad()
def encode_chunk_camera_state(state: ChunkCameraState, state_normalizer: ChunkCameraStateNormalizer):
    """57D condition: exact zero camera, normalized wrists, unchanged AE latents."""
    if not isinstance(state, ChunkCameraState) or not isinstance(state_normalizer, ChunkCameraStateNormalizer):
        raise ValueError("chunk-camera encoding requires ChunkCameraState and the 18D v2 normalizer")
    wrists = state_normalizer.normalize(_pose9_tensor(state.rigid_camera[1:]).reshape(18))
    encoded = torch.cat((wrists.new_zeros(9), wrists[:9], state.hand_latents[0], wrists[9:], state.hand_latents[1]))
    if not torch.isfinite(encoded).all():
        raise ValueError("non-finite normalized chunk-camera state")
    return encoded


@torch.no_grad()
def decode_chunk_camera_state(encoded, state_normalizer: ChunkCameraStateNormalizer, *, source_index):
    """Invert a 57D/64D condition; zero camera represents physical identity."""
    if not isinstance(state_normalizer, ChunkCameraStateNormalizer):
        raise ValueError("chunk-camera decoding requires the 18D v2 normalizer")
    if encoded.shape not in ((57,), (64,)) or not torch.isfinite(encoded).all():
        raise ValueError("encoded state must be a finite 57D or 64D vector")
    if torch.count_nonzero(encoded[:9]):
        raise ValueError("chunk-camera encoded camera must be zero")
    if encoded.numel() == 64 and torch.count_nonzero(encoded[57:]):
        raise ValueError("state padding must be zero")
    wrists = state_normalizer.denormalize(torch.cat((encoded[9:18], encoded[33:42])).float())
    rigid = torch.cat((torch.eye(4, device=encoded.device)[None], _pose9_matrices(wrists.reshape(2, 9))))
    return ChunkCameraState(source_index, rigid, torch.stack((encoded[18:33], encoded[42:57])).float())


@dataclass(frozen=True)
class DecodedChunkCameraAction:
    rigid_chunk: torch.Tensor  # [T,3,4,4], all poses in the input boundary camera
    wrist_camera: torch.Tensor  # [T,2,4,4], wrists in each future frame's own camera
    hand_latents: torch.Tensor  # [T,2,15]
    end_state: ChunkCameraState  # re-anchored in the final camera, ready for next chunk


@torch.no_grad()
def decode_chunk_camera_action(state: ChunkCameraState, future_action, future_normalizer):
    """Integrate unchanged future increments, then re-anchor only the next state."""
    if not isinstance(state, ChunkCameraState):
        raise ValueError("chunk-camera action decoding requires ChunkCameraState")
    decoded = decode_action_chunk(
        BoundaryState(state.source_index, state.rigid_camera, state.hand_latents), future_action, future_normalizer
    )
    rigid = decoded.rigid_f0
    camera_from_chunk = torch.linalg.inv(rigid[:, 0])
    wrists = camera_from_chunk[:, None] @ rigid[:, 1:]
    end_rigid = torch.cat((torch.eye(4, device=rigid.device)[None], wrists[-1]))
    end = ChunkCameraState(decoded.end_state.source_index, end_rigid, decoded.hand_latents[-1].clone())
    return DecodedChunkCameraAction(rigid, wrists, decoded.hand_latents, end)
