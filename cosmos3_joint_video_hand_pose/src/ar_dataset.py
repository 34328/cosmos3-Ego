"""EgoVerse clips for joint video-action AR training (lingbot-va ``frame_stride`` layout).

Each sample is one continuous window of a segment. Video keeps every
``frame_stride``-th source frame (``T = 4n + 1`` frames, ``n + 1`` VAE latents);
actions keep every source frame and are grouped ``K = 4 * frame_stride`` per latent.
V0.2 emits future actions only and separate boundary states (one state per used
chunk). Legacy V0.1 alone repeats the initial state K times. Latent group j>=1
holds source actions K(j-1)+1..Kj; hand visibility follows these action rows.
The fixed-camera representation uses C=4/K=8 with independently fitted 57D
state/future normalizers and explicit validated hand codecs.

FPS labels are the real rates times the source's speed factor (EgoVerse 0.5 slows
human motion to half speed in model time); ``fps_action == fps_video * K / 4`` holds
by construction, which the temporal-causal packer requires for ``K != 4``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from .action import Action57Builder
from .dataset import (
    PROMPT_MODE_SEGMENT_ONLY,
    PROMPT_MODES,
    CosmosActionPromptFormatter,
    CosmosTextTokenCounter,
    EgoVerseCosmosDataset,
    build_prompt_text,
    decode_rgb_video,
)
from .temporal import SPATIAL_TOKENS_PER_LATENT_FRAME

VAE_TEMPORAL_COMPRESSION = 4
DEFAULT_CLIP_FRAME_TIERS = (129, 65, 33)
DEFAULT_SPEED_FACTORS = {"egoverse": 0.5}
AR_MAX_SEQUENCE_LENGTH = 75_000
# CP1 runtime/Flex padding: two streams, two passes, 128 rows each.
JOINT_ATTENTION_ALIGNMENT = 128
JOINT_PACK_PADDING_RESERVE = 4 * JOINT_ATTENTION_ALIGNMENT
AR_V02_TOKEN_BUDGET_VERSION = "joint_chunk_cond_v1_two_pass_full_us_c4_v2"


def joint_reserved_token_budget(raw_tokens):
    """Add padding once per candidate pack; raw sample metadata stays unchanged."""
    return raw_tokens + JOINT_PACK_PADDING_RESERVE


def clip_span(clip_frames: int, frame_stride: int) -> int:
    """Source frames covered by a clip of ``clip_frames`` video frames."""
    return frame_stride * (clip_frames - 1) + 1


def select_clip_frames(source_frames: int, frame_stride: int, tiers=DEFAULT_CLIP_FRAME_TIERS) -> int | None:
    """Largest tier whose span fits in the segment, or None when the segment is too short."""
    for frames in sorted(tiers, reverse=True):
        if (frames - 1) % VAE_TEMPORAL_COMPRESSION:
            raise ValueError(f"clip tiers must satisfy T = 4n + 1, got {frames}")
        if clip_span(frames, frame_stride) <= source_frames:
            return frames
    return None


def ar_latent_frames(clip_frames: int) -> int:
    return 1 + (clip_frames - 1) // VAE_TEMPORAL_COMPRESSION


def ar_token_count(text_tokens: int, clip_frames: int, action_tokens_per_latent: int) -> int:
    """Packed length of one pass: text, generation specials, video patches and action tokens."""
    latents = ar_latent_frames(clip_frames)
    return int(text_tokens + 1 + SPATIAL_TOKENS_PER_LATENT_FRAME * latents + 2 + latents * action_tokens_per_latent)


def ar_v02_token_count(
    text_tokens: int,
    clip_frames: int,
    action_tokens_per_latent: int = 8,
    *,
    chunk_size: int = 4,
    spatial_tokens_per_latent: int = SPATIAL_TOKENS_PER_LATENT_FRAME,
) -> int:
    """Conservative *sum of both passes*, including unsupervised noisy U/S rows.

    C=4 is the fixed training layout. Explicit other C values are only for
    layout regression calculations, not checkpoint budget compatibility.
    Each pass uses text + 2 (EOS and generation start, no BOS)
    and all U/S/V/A tokens. The cap is exclusive in PackingDataLoader, so a
    pack must have joint_reserved_token_budget(sum(ar_num_tokens)) < the cap.
    It is a compute budget, not the length of either individual model forward.
    """
    if type(text_tokens) is not int or text_tokens < 0:
        raise ValueError("text_tokens must be a non-negative integer")
    if type(clip_frames) is not int or clip_frames < 5 or (clip_frames - 1) % 4:
        raise ValueError("clip_frames must contain 1+4N sampled RGB frames, N>=1")
    if type(action_tokens_per_latent) is not int or action_tokens_per_latent < 1:
        raise ValueError("action_tokens_per_latent must be positive")
    if type(chunk_size) is not int or chunk_size not in (1, 2, 3, 4):
        raise ValueError("chunk_size must be one of 1,2,3,4")
    n = (clip_frames - 1) // 4
    boundaries = (n + chunk_size - 1) // chunk_size
    if type(spatial_tokens_per_latent) is not int or spatial_tokens_per_latent < 1:
        raise ValueError("spatial_tokens_per_latent must be positive")
    generation = spatial_tokens_per_latent * (n + boundaries) + action_tokens_per_latent * n + boundaries
    return 2 * (text_tokens + 2 + generation)


class ARV02BudgetTransform:
    """Refresh budget after the ordinary transform has actually tokenized text.

    SequencePlan retains its raw RGB/future-action geometry. No fake frames or
    states are inserted to manipulate the generic single-pass cost estimator;
    the v0.2 dataloader subclass reads ar_num_tokens and its version explicitly.
    """

    def __init__(self, transform, max_sequence_length):
        self.transform = transform
        self.max_sequence_length = max_sequence_length

    def __call__(self, sample, resolution=None):
        sample = self.transform(sample, resolution=resolution)
        if sample.get("ar_layout_version") != "joint_chunk_cond_v1":
            return sample
        if sample.get("ar_token_budget_version") != AR_V02_TOKEN_BUDGET_VERSION:
            raise ValueError("dataset token budget version mismatch; old C1 metadata must not be reinterpreted")
        tokens = sample.get("text_token_ids")
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim not in (1, 2)
            or (tokens.ndim == 2 and tokens.shape[0] != 1)
        ):
            raise ValueError("v0.2 budget requires one tokenized caption per sample")
        frames = int(sample["clip_frames"])
        span = len(sample["action_source_frame_indices"])
        k = (span - 1) // ((frames - 1) // 4)
        sample["ar_num_tokens"] = ar_v02_token_count(tokens.numel(), frames, k)
        sample["ar_token_budget_version"] = AR_V02_TOKEN_BUDGET_VERSION
        if joint_reserved_token_budget(sample["ar_num_tokens"]) >= self.max_sequence_length:
            raise ValueError("tokenized v0.2 sample exceeds the padding-inclusive double-pass token budget")
        return sample


class EgoVerseARCosmosDataset(EgoVerseCosmosDataset):
    """Expose exact-window reconstruction through the training map wrapper."""

    def get_item_at_window(self, index, *, window_start=None, source_frame_indices=None):
        raw = self.dataset.get_item_at_window(
            index, window_start=window_start, source_frame_indices=source_frame_indices
        )
        sample = self.transform(raw, resolution=None)
        sample["dataset_index"] = int(index)
        sample["image_size"] = torch.tensor([368, 640, 368, 640], dtype=torch.float32)
        if sample["action"].shape[-1] != 64 or int(sample["raw_action_dim"]) != 57:
            raise AssertionError("Cosmos action padding contract is not 57D -> 64D")
        return sample


def ar_fps_labels(source_fps: float, frame_stride: int, speed_factor: float) -> tuple[float, float]:
    """(video, action) FPS labels: real rates times the speed factor."""
    if source_fps <= 0 or frame_stride < 1 or speed_factor <= 0:
        raise ValueError("source_fps, frame_stride and speed_factor must be positive")
    return float(source_fps) / frame_stride * speed_factor, float(source_fps) * speed_factor


def expand_state_group(per_frame: torch.Tensor, tokens_per_latent: int) -> torch.Tensor:
    """``[1 + (L-1)K, ...]`` source-frame rows -> ``[L*K, ...]`` with row 0 repeated K times."""
    if (per_frame.shape[0] - 1) % tokens_per_latent:
        raise ValueError(f"{per_frame.shape[0]} rows are not 1 + a multiple of K={tokens_per_latent}")
    repeat = [tokens_per_latent] + [1] * (per_frame.ndim - 1)
    return torch.cat([per_frame[:1].repeat(*repeat), per_frame[1:]], dim=0)


class EgoVerseARSegmentDataset(Dataset):
    """Map-style manifest dataset producing unpadded 57D AR clips."""

    def __init__(
        self,
        episodes_manifest: str | Path,
        segments_manifest: str | Path,
        *,
        split: str = "train",
        frame_stride: int = 2,
        clip_frame_tiers: tuple[int, ...] = DEFAULT_CLIP_FRAME_TIERS,
        source_name: str = "egoverse",
        speed_factors: dict[str, float] | None = None,
        random_window: bool = True,
        token_counter: Callable[[str], int] | None = None,
        prompt_formatter: Callable[[str, int, float], str] | None = None,
        action_builder: Action57Builder | None = None,
        max_sequence_length: int = AR_MAX_SEQUENCE_LENGTH,
        prompt_mode: str = PROMPT_MODE_SEGMENT_ONLY,
        chunk_state_normalizer: str | Path | None = None,
        valid_windows_manifest: str | Path | None = None,
        action_representation: str | None = None,
        future_normalizer: str | Path | None = None,
        right_codec: str | Path | None = None,
        left_codec: str | Path | None = None,
    ):
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"unsupported prompt mode {prompt_mode!r}; expected one of {PROMPT_MODES}")
        speed_factors = dict(DEFAULT_SPEED_FACTORS if speed_factors is None else speed_factors)
        if source_name not in speed_factors:
            raise ValueError(f"no speed factor configured for source {source_name!r}")
        with Path(episodes_manifest).open(newline="", encoding="utf-8") as handle:
            episodes = {row["episode_hash"]: row for row in csv.DictReader(handle) if row["split"] == split}
        with Path(segments_manifest).open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split and row["episode_hash"] in episodes]
        if not rows:
            raise ValueError(f"no {split!r} segments found")
        self.episodes = episodes
        self.frame_stride = int(frame_stride)
        self.tokens_per_latent = VAE_TEMPORAL_COMPRESSION * self.frame_stride
        self.clip_frame_tiers = tuple(int(t) for t in clip_frame_tiers)
        self.speed_factor = float(speed_factors[source_name])
        self.random_window = bool(random_window)
        self.token_counter = token_counter or CosmosTextTokenCounter()
        self.prompt_formatter = prompt_formatter or CosmosActionPromptFormatter()
        from .action_fixed_normalization import REPRESENTATION, NORMALIZER_SCHEMA
        fixed_profile = (json.loads(Path(chunk_state_normalizer).read_text())
                         if chunk_state_normalizer is not None else {})
        if fixed_profile.get("schema") == NORMALIZER_SCHEMA:
            inferred = fixed_profile.get("representation")
            if inferred != REPRESENTATION:
                raise ValueError("retired or missing action representation in state statistics; refit required")
        else:
            inferred = "legacy_local_delta_absolute_hand_v1"
        self.action_representation = action_representation or inferred
        if self.action_representation != inferred:
            raise ValueError("action representation and state statistics schema mismatch")
        self.fixed_camera_mode = inferred == REPRESENTATION
        self.fixed_hand_codecs = None
        self.action_builder = (None if self.fixed_camera_mode else
                               action_builder or Action57Builder(rigid_pose_frame_delta=True))
        self.chunk_state_normalizer = None
        self.chunk_camera_mode = False
        self.state_normalizer_sha256 = None
        self.valid_windows_sha256 = None
        self.valid_windows = None
        if self.fixed_camera_mode:
            if action_builder is not None:
                raise ValueError("fixed-camera data cannot use a legacy Action57Builder")
            self._init_fixed(chunk_state_normalizer, future_normalizer, valid_windows_manifest,
                             (right_codec, left_codec), episodes_manifest, segments_manifest, split)
        elif chunk_state_normalizer is not None:
            import hashlib
            from .normalization import PiecewiseAsinhNormalizer

            profile = json.loads(Path(chunk_state_normalizer).read_text())
            from .ar_chunk_state import (
                CHUNK_CAMERA_STATE_SCHEMA,
                CHUNK_CAMERA_LAYOUT_VERSION,
                VALID_WINDOWS_SCHEMA,
                ChunkCameraStateNormalizer,
            )

            self.chunk_camera_mode = profile.get("schema") == CHUNK_CAMERA_STATE_SCHEMA
            if (
                profile.get("schema") not in (CHUNK_CAMERA_STATE_SCHEMA, "ar_v02_boundary_state_f0_v1")
                or profile.get("split") != "train"
            ):
                raise ValueError("per-chunk state requires versioned train-only boundary statistics")
            if self.chunk_camera_mode and not getattr(self.action_builder, "rigid_pose_frame_delta", False):
                raise ValueError("chunk-camera states require unchanged frame-delta future actions")
            if not profile.get("frozen") or not profile.get("manifest_sha256"):
                raise ValueError("state statistics must be frozen and tied to the training manifest")
            if valid_windows_manifest is None:
                raise ValueError("per-chunk states require the audited common valid-window manifest")
            manifest_bytes = Path(valid_windows_manifest).read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != profile["manifest_sha256"]:
                raise ValueError("valid-window manifest does not match the frozen state normalizer")
            manifest = json.loads(manifest_bytes)
            expected_schema = VALID_WINDOWS_SCHEMA if self.chunk_camera_mode else "ar_v02_valid_windows_v1"
            if manifest.get("schema") != expected_schema or manifest["frame_stride"] != self.frame_stride:
                raise ValueError("invalid v0.2 window manifest or stride mismatch")
            if self.chunk_camera_mode and (
                manifest.get("state_schema") != CHUNK_CAMERA_STATE_SCHEMA
                or manifest.get("layout_version") != CHUNK_CAMERA_LAYOUT_VERSION
            ):
                raise ValueError("window manifest state/layout version mismatch")
            prefix = "train" if split == "train" else "heldout"
            for kind, path in (("episodes", episodes_manifest), ("segments", segments_manifest)):
                if (
                    hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    != manifest["source_hashes"][prefix + "_" + kind]
                ):
                    raise ValueError("dataset source differs from the audited manifest")
            self.valid_windows = manifest["windows"]
            self.valid_windows_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            self.state_normalizer_sha256 = hashlib.sha256(Path(chunk_state_normalizer).read_bytes()).hexdigest()
            loader = ChunkCameraStateNormalizer if self.chunk_camera_mode else PiecewiseAsinhNormalizer
            self.chunk_state_normalizer = loader(chunk_state_normalizer)
        self.max_sequence_length = int(max_sequence_length)
        self.prompt_mode = prompt_mode
        self.tier_counts = {str(t): 0 for t in sorted(self.clip_frame_tiers, reverse=True)} | {"dropped": 0}
        self.dropped_segments = []
        self.rows = []
        for row in rows:
            plan = self._build_clip_plan(row, episodes[row["episode_hash"]])
            if plan is None:
                continue
            row.update(plan)
            if self.valid_windows is not None:
                sid = f"{row['episode_hash']}:{row['span_index']}:{row['start_idx']}:{row['end_idx']}"
                window = self.valid_windows.get(sid)
                if window is None or window["frames"] != row["_clip_frames"] or window["split"] != split:
                    raise ValueError(f"dataset window differs from audited plan: {sid}")
                if not window["starts"]:
                    self._drop(sid, int(row["end_idx"]) - int(row["start_idx"]), "invalid_tracking")
                    continue
                starts = window["starts"]
                if (
                    window["span"] != row["_clip_span"]
                    or window["episode"] != row["episode_hash"]
                    or any(
                        type(x) is not int or x < int(row["start_idx"]) or x + row["_clip_span"] > int(row["end_idx"])
                        for x in starts
                    )
                    or starts != sorted(set(starts))
                ):
                    raise ValueError(f"invalid audited source window indexes: {sid}")
                row["_valid_starts"] = starts
            self.rows.append(row)
            self.tier_counts[str(row["_clip_frames"])] += 1
        if not self.rows:
            raise ValueError(f"no {split!r} segment is long enough for the smallest clip tier")

    def _init_fixed(self, state_path, future_path, manifest_path, codec_paths,
                    episodes_path, segments_path, split):
        import hashlib
        from .action_fixed_normalization import (FixedCameraNormalizer, load_fixed_codecs,
                                                REPRESENTATION, NORMALIZER_SCHEMA, VALID_WINDOWS_SCHEMA)
        if self.frame_stride != 2 or future_path is None or manifest_path is None:
            raise ValueError("fixed-camera dataset requires K=8, separate future statistics and manifest")
        self.fixed_hand_codecs = load_fixed_codecs(codec_paths)
        hashes = tuple(c.checkpoint_sha256 for c in self.fixed_hand_codecs)
        self.chunk_state_normalizer = FixedCameraNormalizer(state_path, kind="state", codec_sha256=hashes)
        self.future_normalizer = FixedCameraNormalizer(future_path, kind="future", codec_sha256=hashes)
        data = Path(manifest_path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if any(n.profile["manifest_sha256"] != digest for n in
               (self.chunk_state_normalizer, self.future_normalizer)):
            raise ValueError("fixed-camera state/future manifest hash mismatch")
        manifest = json.loads(data)
        expected = dict(schema=VALID_WINDOWS_SCHEMA, state_schema=NORMALIZER_SCHEMA,
                        representation=REPRESENTATION, layout_version="joint_chunk_cond_v1",
                        chunk_size=4, tokens_per_latent=8, frame_stride=2, codec_sha256=list(hashes))
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("fixed-camera manifest representation/layout/codec mismatch")
        if manifest.get("tracking_validation") != dict(
            version="fixed_camera_float32_v1", quaternion_norm_atol=1e-4,
            so3_atol=1e-5, so3_rtol=1e-5, missing_hand="exclude_entire_window",
            finite_dtype="float32", scope="all_source_frames"):
            raise ValueError("fixed-camera manifest lacks matching strict tracking audit")
        prefix = "train" if split == "train" else "heldout"
        for name, path in (("episodes", episodes_path), ("segments", segments_path)):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest["source_hashes"][prefix+"_"+name]:
                raise ValueError("dataset source differs from fixed-camera audit")
        self.valid_windows = manifest["windows"]
        self.valid_windows_sha256 = digest
        self.state_normalizer_sha256 = hashlib.sha256(Path(state_path).read_bytes()).hexdigest()
        self.future_normalizer_sha256 = hashlib.sha256(Path(future_path).read_bytes()).hexdigest()
        self.chunk_camera_mode = True

    def _build_clip_plan(self, row: dict, episode: dict) -> dict | None:
        sample_id = f"{row['episode_hash']}:{row['span_index']}:{row['start_idx']}:{row['end_idx']}"
        source_frames = int(row["end_idx"]) - int(row["start_idx"])
        frames = select_clip_frames(source_frames, self.frame_stride, self.clip_frame_tiers)
        if frames is None:
            self._drop(sample_id, source_frames, "shorter_than_smallest_clip_tier")
            return None
        fps_video, fps_action = ar_fps_labels(float(episode["fps"]), self.frame_stride, self.speed_factor)
        caption = build_prompt_text(row["text_normalized"], episode["task_description"], mode=self.prompt_mode)
        prompt = self.prompt_formatter(caption, frames, fps_video)
        text_tokens = self.token_counter(prompt)
        budget = ar_token_count(text_tokens, frames, self.tokens_per_latent)
        if self.chunk_camera_mode:
            budget = ar_v02_token_count(text_tokens, frames, self.tokens_per_latent)
        admission_budget = joint_reserved_token_budget(budget) if self.chunk_camera_mode else budget
        if admission_budget >= self.max_sequence_length:
            self._drop(sample_id, source_frames, "exceeds_packing_cap")
            return None
        return {
            "_clip_frames": frames,
            "_clip_span": clip_span(frames, self.frame_stride),
            "_fps_video": fps_video,
            "_fps_action": fps_action,
            "_structured_prompt": prompt,
            "_ar_num_tokens": budget,
        }

    def _drop(self, sample_id: str, source_frames: int, reason: str) -> None:
        self.tier_counts["dropped"] += 1
        self.dropped_segments.append({"sample_id": sample_id, "source_frames": source_frames, "reason": reason})

    def __len__(self) -> int:
        return len(self.rows)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return [(index, 1) for index in range(len(self))]

    def window_start(self, row: dict) -> int:
        if "_valid_starts" in row:
            return random.choice(row["_valid_starts"]) if self.random_window else row["_valid_starts"][0]
        """Source index of the clip's first frame (random offset in training, fixed at 0 otherwise)."""
        start, end = int(row["start_idx"]), int(row["end_idx"])
        slack = (end - start) - int(row["_clip_span"])
        return start + (random.randint(0, slack) if self.random_window and slack > 0 else 0)

    def __getitem__(self, index: int) -> dict:
        return self.get_item_at_window(index, window_start=self.window_start(self.rows[index]))

    def get_item_at_window(self, index: int, *, window_start=None, source_frame_indices=None) -> dict:
        """Rebuild all modalities from a saved absolute window without drawing RNG.

        Accept unbatched [T] or singleton-collated [1,T] RGB source indexes.
        Never silently clamp, shift, or substitute another audited window.
        The caller restores saved CFG/text metadata only after verifying the
        regenerated source indexes, states, and normalizer/manifest hashes.
        """
        row = self.rows[index]
        episode = self.episodes[row["episode_hash"]]
        frames, span, stride = int(row["_clip_frames"]), int(row["_clip_span"]), self.frame_stride
        saved_indexes = None
        if source_frame_indices is not None:
            saved_indexes = torch.as_tensor(source_frame_indices).detach().cpu()
            if saved_indexes.ndim == 2 and saved_indexes.shape[0] == 1:
                saved_indexes = saved_indexes[0]
            if saved_indexes.shape != (frames,) or saved_indexes.dtype not in (torch.int32, torch.int64):
                raise ValueError("saved source_frame_indices must contain the full integer RGB frame list")
            first_from_indexes = int(saved_indexes[0])
            expected = first_from_indexes + torch.arange(frames) * stride
            if not torch.equal(saved_indexes, expected):
                raise ValueError("saved RGB source indexes do not match the continuous window stride")
            if window_start is None:
                window_start = first_from_indexes
        if isinstance(window_start, torch.Tensor):
            if window_start.numel() != 1 or window_start.dtype not in (torch.int32, torch.int64):
                raise ValueError("saved window_start must be one integer")
            window_start = window_start.item()
        if not isinstance(window_start, (int, np.integer)) or isinstance(window_start, (bool, np.bool_)):
            raise ValueError("exact reconstruction requires an integer window_start or source_frame_indices")
        first = int(window_start)
        if saved_indexes is not None and first != int(saved_indexes[0]):
            raise ValueError("saved window_start and source indexes disagree")
        if first < int(row["start_idx"]) or first + span > int(row["end_idx"]):
            raise ValueError("saved window exceeds the manifest segment")
        if "_valid_starts" in row and first not in row["_valid_starts"]:
            raise ValueError("saved window is absent from the frozen valid-window manifest")
        action_indexes = first + np.arange(span, dtype=np.int64)
        video_indexes = action_indexes[::stride]
        group = zarr.open_group(episode["abs_zarr_path"], mode="r")
        if action_indexes[-1] >= min(int(row["end_idx"]), int(group.attrs["total_frames"])):
            raise IndexError("clip window exceeds the manifest segment or episode")

        def read(name: str, dtype=np.float64):
            return np.asarray(group[name][action_indexes], dtype=dtype)

        head, right, left = read("obs_head_pose"), read("right.obs_wrist_pose"), read("left.obs_wrist_pose")
        right_points, left_points = read("right.obs_keypoints"), read("left.obs_keypoints")
        if self.chunk_camera_mode:
            from .ar_v02_prepare_data import invalid_frames

            invalid = invalid_frames(
                {
                    "obs_head_pose": head,
                    "right.obs_wrist_pose": right,
                    "left.obs_wrist_pose": left,
                    "right.obs_keypoints": right_points,
                    "left.obs_keypoints": left_points,
                }, fixed_camera=self.fixed_camera_mode,
            )
            reasons = [name for name, flags in invalid.items() if flags.any()]
            if reasons:
                raise ValueError(f"audited window tracking changed or is invalid: {reasons}")
        if self.fixed_camera_mode:
            from .action_fixed_normalization import encode_fixed_window
            physical_states, physical_future = encode_fixed_window(
                head, right, left, right_points, left_points, self.fixed_hand_codecs)
            normalized_future = self.future_normalizer.normalize(physical_future)
            # Internal compatibility row only; never emitted as a future action.
            per_frame_action = torch.cat((normalized_future.new_zeros(1, 57), normalized_future))
        else:
            per_frame_action = self.action_builder.build(
                head_pose=head, right_wrist_pose=right, left_wrist_pose=left,
                right_keypoints=right_points, left_keypoints=left_points,
            )  # [span,57]

        def visibility_of(side: str) -> np.ndarray:
            name = f"{side}.obs_palm_in_fov_front_1"
            if name in group:
                return read(name, np.uint8)
            # Episodes outside the materialized training subsets: same projection, computed on the fly.
            from .materialize_visibility import compute_visibility

            return compute_visibility(group, int(action_indexes[-1]) + 1, side)[action_indexes]

        per_frame_visibility = torch.from_numpy(
            np.stack((visibility_of("right"), visibility_of("left")), axis=-1).astype(np.bool_)
        )  # [span,2]
        video = decode_rgb_video(group["images.front_1"][video_indexes])
        if self.chunk_state_normalizer is None:
            # v0.1 compatibility only; repetition was a packer choice, not FM.
            action = expand_state_group(per_frame_action, self.tokens_per_latent)
            visibility = expand_state_group(per_frame_visibility, self.tokens_per_latent)
            rows = ar_latent_frames(frames) * self.tokens_per_latent
        else:
            action, visibility, rows = per_frame_action[1:], per_frame_visibility[1:], span - 1
        assert video.shape == (3, frames, 368, 640)
        assert action.shape == (rows, 57) and visibility.shape == (rows, 2)
        sample = {
            "ai_caption": json.loads(row["_structured_prompt"]),
            "video": video,
            "action": action,
            "hand_visibility": visibility,
            "conditioning_fps": float(row["_fps_video"]),
            "conditioning_fps_action": float(row["_fps_action"]),
            "mode": "wam",
            "domain_id": torch.tensor(3, dtype=torch.long),
            "viewpoint": "ego_view",
            "sample_id": f"{row['episode_hash']}:{row['span_index']}:{row['start_idx']}:{row['end_idx']}",
            "clip_frames": frames,
            "window_start": first,
            "source_frame_indices": torch.from_numpy(video_indexes.copy()),
            "action_source_frame_indices": torch.from_numpy(action_indexes.copy()),
            # Raw metric target: metres, world frame, right then left, includes boundary.
            "ar_source_keypoints_world": torch.from_numpy(np.stack(
                (right_points.reshape(span, 21, 3), left_points.reshape(span, 21, 3)), axis=1).copy()),
        }
        if self.fixed_camera_mode:
            offsets = torch.arange(0, span - 1, 8, dtype=torch.long)
            sample.update(
                ar_boundary_states=torch.nn.functional.pad(
                    self.chunk_state_normalizer.normalize(physical_states), (0, 7)),
                ar_layout_version="joint_chunk_cond_v1",
                ar_action_representation=self.action_representation,
                ar_chunk_size=4,
                ar_num_tokens=int(row["_ar_num_tokens"]),
                ar_token_budget_version=AR_V02_TOKEN_BUDGET_VERSION,
                ar_boundary_source_offsets=offsets,
                ar_boundary_source_indices=offsets + first,
                ar_boundary_times=(offsets + first).double() / float(episode["fps"]),
                ar_condition_source="gt",
                future_action_source_frame_indices=torch.from_numpy(action_indexes[1:].copy()),
                ar_source_poses=torch.from_numpy(np.stack((head, right, left), axis=1).copy()),
                ar_state_schema=self.chunk_state_normalizer.schema,
                ar_state_normalizer_sha256=self.state_normalizer_sha256,
                ar_future_normalizer_sha256=self.future_normalizer_sha256,
                ar_codec_sha256=self.chunk_state_normalizer.codec_sha256,
                ar_right_hand_codec_sha256=self.chunk_state_normalizer.codec_sha256[0],
                ar_left_hand_codec_sha256=self.chunk_state_normalizer.codec_sha256[1],
                ar_valid_windows_sha256=self.valid_windows_sha256,
            )
            return sample
        if self.chunk_state_normalizer is not None:
            from .ar_chunk_state import (
                boundary_states_from_streams,
                chunk_boundaries,
                encode_boundary_state,
                chunk_camera_states_from_streams,
                encode_chunk_camera_state,
                CHUNK_CAMERA_LAYOUT_VERSION,
            )

            # V0.2 training fixes C=4 in the model. Keep all candidate
            # boundaries for layout compatibility; packing selects the C=4 states.
            boundaries = chunk_boundaries(ar_latent_frames(frames), 1, tokens_per_latent=self.tokens_per_latent)
            hands = torch.stack((per_frame_action[:, 18:33], per_frame_action[:, 42:57]), dim=1)
            state_builder = chunk_camera_states_from_streams if self.chunk_camera_mode else boundary_states_from_streams
            encode_state = encode_chunk_camera_state if self.chunk_camera_mode else encode_boundary_state
            states = state_builder(
                head_pose=head, right_wrist_pose=right, left_wrist_pose=left, hand_latents=hands, boundaries=boundaries
            )
            sample["ar_boundary_states"] = torch.nn.functional.pad(
                torch.stack([encode_state(state, self.chunk_state_normalizer) for state in states]), (0, 7)
            )
            sample["ar_layout_version"] = (
                CHUNK_CAMERA_LAYOUT_VERSION if self.chunk_camera_mode else "joint_state_single_v1"
            )
            if self.chunk_camera_mode:
                # Preserve legacy full source indexes, including the initial boundary.
                # Future action rows have a separate, unambiguous one-to-one index.
                offsets = torch.tensor([state.source_index for state in states], dtype=torch.long)
                sample.update(
                    ar_num_tokens=int(row["_ar_num_tokens"]),
                    ar_token_budget_version=AR_V02_TOKEN_BUDGET_VERSION,
                    ar_boundary_source_offsets=offsets,
                    ar_boundary_source_indices=offsets + first,
                    ar_boundary_times=(offsets + first).double() / float(episode["fps"]),
                    ar_condition_source="gt",
                    future_action_source_frame_indices=torch.from_numpy(action_indexes[1:].copy()),
                    ar_source_poses=torch.from_numpy(np.stack((head, right, left), axis=1).copy()),
                    ar_hand_latents=hands.clone(),
                    ar_state_schema=self.chunk_state_normalizer.schema,
                    ar_state_normalizer_sha256=self.state_normalizer_sha256,
                    ar_valid_windows_sha256=self.valid_windows_sha256,
                )
            sample["action"] = per_frame_action[1:]
            sample["hand_visibility"] = per_frame_visibility[1:]
        return sample


def get_egoverse_ar_dataset(
    *,
    episodes_manifest: str,
    segments_manifest: str,
    tokenizer_config: dict,
    cfg_dropout_rate: float = 0.1,
    iterable_shuffle: bool = True,
    seed: int = 42,
    max_sequence_length: int = AR_MAX_SEQUENCE_LENGTH,
    prompt_mode: str = PROMPT_MODE_SEGMENT_ONLY,
    state_normalizer: str | None = None,
    future_normalizer: str | None = None,
    frame_stride: int = 2,
    clip_frame_tiers: tuple[int, ...] = DEFAULT_CLIP_FRAME_TIERS,
    speed_factors: dict[str, float] | None = None,
    random_window: bool = True,
    split: str = "train",
    chunk_state_normalizer: str | None = None,
    valid_windows_manifest: str | None = None,
    action_representation: str | None = None,
    right_codec: str | None = None,
    left_codec: str | None = None,
):
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

    builder_kwargs = {"rigid_pose_frame_delta": True}
    if state_normalizer is not None:
        builder_kwargs["state_normalizer"] = state_normalizer
    if future_normalizer is not None:
        builder_kwargs["future_normalizer"] = future_normalizer
    from .action_fixed_normalization import NORMALIZER_SCHEMA, REPRESENTATION
    fixed = (action_representation == REPRESENTATION or
             (chunk_state_normalizer is not None and
              json.loads(Path(chunk_state_normalizer).read_text()).get("schema") == NORMALIZER_SCHEMA))
    raw = EgoVerseARSegmentDataset(
        episodes_manifest,
        segments_manifest,
        split=split,
        frame_stride=frame_stride,
        clip_frame_tiers=tuple(clip_frame_tiers),
        speed_factors=None if speed_factors is None else dict(speed_factors),
        random_window=random_window,
        action_builder=None if fixed else Action57Builder(**builder_kwargs),
        max_sequence_length=max_sequence_length,
        prompt_mode=prompt_mode,
        chunk_state_normalizer=chunk_state_normalizer,
        valid_windows_manifest=valid_windows_manifest,
        action_representation=action_representation,
        future_normalizer=future_normalizer,
        right_codec=right_codec,
        left_codec=left_codec,
    )
    transform = ActionTransformPipeline(
        pad_keys=[],
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=64,
        action_channel_masking=True,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        # The adapter already applied Cosmos' JSON formatter with the real canvas and slowed FPS.
        format_prompt_as_json=False,
    )
    if raw.chunk_camera_mode:
        transform = ARV02BudgetTransform(transform, max_sequence_length)
    dataset = EgoVerseARCosmosDataset(raw, transform)
    return ActionIterableShuffleDataset(dataset, seed=seed) if iterable_shuffle else dataset
