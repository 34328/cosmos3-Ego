"""Single-block V0.2 control API; storage and per-step work are episode-length independent.

Usage:
    stream = StreamingJointSampler(model, text_ids=ids, latent_shape=(4, 24, 40),
        state_normalizer=state_norm, future_normalizer=action_norm)
    result = stream.step(u, state, gt_video=v_gt, gt_action=a_gt)  # history="gt"
    # Consume result.video/action/decoded synchronously. The stream keeps no output list.
    stream.reset(seed=42)  # mandatory after an interrupted/failed call

U is [1,D,1,H,W] latent (or one RGB frame with input_is_latent=False);
state is a ChunkCameraState or normalized [57]/[64] tensor. Current GT V/A
are required only for gt/oracle. For generated, pass U/S only on the first call.
C=4 is the deployment default; C=1..3 and partial final blocks support correctness
tests. The caller owns model/VAE, frozen normalizers, tokenizer and output sinks.
"""

from dataclasses import dataclass
import math
import time

import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence, PackedSequenceBuilder

from .ar_chunk_state import (
    ChunkCameraState,
    decode_chunk_camera_action,
    decode_chunk_camera_state,
    encode_chunk_camera_state,
)
from .ar_v02_cache import BoundedJointKVCache
from .ar_v02_inference import HISTORY_MODES, JOINT_STEPS, _schedule
from .ar_v02_layout import ACTION, CONDITION_VIDEO, LAYOUT_VERSION, STATE, VIDEO


@dataclass(frozen=True)
class StreamingChunk:
    video: torch.Tensor
    action: torch.Tensor
    decoded: object
    condition_video: torch.Tensor
    condition_state: torch.Tensor
    report: dict


class ChunkPackWorkspace:
    """Prebuilt text/condition/target packs for at most C frames; no remapping.

    Shapes, indexes, modality masks and token buffers are allocated once.
    Only the temporal position row changes at chunk boundaries. Spatial IDs
    keep the same text/modality offset used by the training packer.
    """

    def __init__(
        self,
        *,
        text_ids,
        special_tokens,
        latent_shape,
        chunk_size,
        patch_size,
        device,
        dtype,
        fps_action=15.0,
        base_fps=24.0,
        reset_spatial=True,
        modality_margin=0.0,
        initial_temporal_offset=0.0,
        action_domain_id=0,
    ):
        self.device, self.dtype = torch.device(device), dtype
        channels, height, width = latent_shape
        if min(channels, height, width, patch_size) < 1:
            raise ValueError("positive latent geometry and patch_size required")
        self.latent_shape = tuple(latent_shape)
        self.ph, self.pw = math.ceil(height / patch_size), math.ceil(width / patch_size)
        self.vision_tokens = self.ph * self.pw
        self.time_scale = (base_fps / 4) / fps_action
        builder = PackedSequenceBuilder(uses_single_timestep=False)
        builder.begin_sample(initial_temporal_offset)
        nt = builder.pack_text_tokens(list(text_ids), special_tokens, True, use_float_positions=True)
        self.offset = builder.mrope_temporal_offset + modality_margin
        self.text = PackedSequence(
            sample_lens=[nt],
            split_lens=[nt, 0],
            attn_modes=["causal", "full"],
            is_image_batch=False,
            uses_single_timestep=False,
            sequence_length=nt,
            text_ids=torch.tensor(builder.text_ids, dtype=torch.long),
            text_indexes=torch.arange(nt),
            position_ids=torch.cat(builder.position_ids, dim=1),
            text_caption_lens=builder.text_caption_lens,
            text_caption_view_ids=builder.text_caption_view_ids,
        )
        spatial, _ = get_3d_mrope_ids_vae_tokens(
            1,
            self.ph,
            self.pw,
            self.offset,
            reset_spatial_indices=reset_spatial,
            fps=fps_action / 2,
            base_fps=base_fps,
            temporal_compression_factor=4,
            start_frame_offset=0,
        )
        self.packs, self.sources = {}, {}
        for phase, frames in [("condition", 0)] + [
            (phase, frames) for frames in range(1, chunk_size + 1) for phase in ("noisy", "refresh")
        ]:
            condition, clean = phase == "condition", phase != "noisy"
            roles = (
                [CONDITION_VIDEO] * self.vision_tokens + [STATE]
                if condition
                else ([ACTION] * 8 + [VIDEO] * self.vision_tokens) * frames
            )
            roles = torch.tensor(roles)
            vi = torch.where((roles == VIDEO) | (roles == CONDITION_VIDEO))[0]
            ai = torch.where((roles == ACTION) | (roles == STATE))[0]
            nv, na = (1, 1) if condition else (frames, 8 * frames)
            source = torch.zeros(len(roles), dtype=torch.float32)
            if not condition:
                source[ai] = torch.arange(1, na + 1).float()
                source[vi] = (torch.arange(1, nv + 1) * 8).repeat_interleave(self.vision_tokens).float()
            positions = torch.full((3, len(roles)), 0.0 if reset_spatial else self.offset)
            positions[1:, vi] = spatial[1:].repeat(1, nv)
            positions[0] = self.offset + source * self.time_scale

            def modality(indexes, shape, payload, rows):
                return ModalityData(
                    sequence_indexes=indexes,
                    mse_loss_indexes=torch.empty(0, dtype=torch.long) if clean else indexes.clone(),
                    timesteps=torch.zeros(0 if clean else len(indexes)),
                    token_shapes=[shape],
                    tokens=[payload],
                    condition_mask=[
                        torch.full(
                            (rows, 1, 1) if len(shape) == 3 else (rows, 1),
                            float(clean),
                            device=self.device,
                            dtype=dtype,
                        )
                    ],
                    noisy_frame_indexes=[torch.empty(0, dtype=torch.long) if clean else torch.arange(rows)],
                )

            vision = modality(
                vi,
                (nv, self.ph, self.pw),
                torch.empty(1, channels, nv, height, width, device=self.device, dtype=dtype),
                nv,
            )
            action = modality(ai, (na,), torch.empty(na, 64, device=self.device, dtype=dtype), na)
            action.domain_id = [torch.tensor([action_domain_id])]
            action.raw_action_dim = [torch.tensor(57)]
            vision.seconds_per_frame = [8 / fps_action]
            action.seconds_per_frame = [1 / fps_action]
            key = (phase, frames)
            self.packs[key] = PackedSequence(
                sample_lens=[len(roles)],
                split_lens=[len(roles)],
                attn_modes=["full"],
                is_image_batch=False,
                uses_single_timestep=False,
                sequence_length=len(roles),
                text_ids=torch.empty(0, dtype=torch.long),
                text_indexes=torch.empty(0, dtype=torch.long),
                position_ids=positions,
                vision=vision,
                action=action,
                action_state_mask=torch.full((na,), condition),
                vision_condition_type_mask=torch.full((nv * self.vision_tokens,), condition),
                vision_item_split_lens=[[len(roles)]],
            )
            self.sources[key] = source.to(self.device)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                for packed in [self.text, *self.packs.values()]:
                    packed.to_cuda()

    def set_boundary(self, source_index, frames):
        for key in (("condition", 0), ("noisy", frames), ("refresh", frames)):
            self.packs[key].position_ids[0].copy_(self.offset + (self.sources[key] + source_index) * self.time_scale)

    def load(self, phase, video, action, *, video_t=0.0, action_t=0.0):
        key = (phase, 0 if phase == "condition" else video.shape[2])
        packed = self.packs[key]
        packed.vision.tokens[0].copy_(video)
        packed.action.tokens[0].copy_(action)
        packed.vision.timesteps.fill_(video_t)
        packed.action.timesteps.fill_(action_t)
        return packed


class StreamingJointSampler:
    @torch.no_grad()
    def __init__(
        self,
        model,
        *,
        text_ids,
        latent_shape,
        state_normalizer,
        future_normalizer,
        chunk_size=4,
        history="gt",
        seed=42,
        source_fps=30.0,
        speed_factor=0.5,
        source_start=0,
        initial_temporal_offset=0.0,
        action_domain_id=0,
        video_schedule=None,
        action_schedule=None,
        steps=JOINT_STEPS,
    ):
        if steps != JOINT_STEPS:
            raise ValueError("AR v0.2 requires exactly 30 joint Euler steps")
        if chunk_size not in (1, 2, 3, 4) or history not in HISTORY_MODES:
            raise ValueError("requires C=1..4 and a supported history mode")
        if not all(math.isfinite(x) and x > 0 for x in (source_fps, speed_factor)):
            raise ValueError("source_fps and speed_factor must be finite and positive")
        self.model, self.history, self.chunk_size = model, history, chunk_size
        self.state_normalizer, self.future_normalizer = state_normalizer, future_normalizer
        self.source_fps = float(source_fps)
        self.device = torch.device(model.tensor_kwargs["device"])
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = model.tensor_kwargs["dtype"]
        cfg = model.config.diffusion_expert_config
        self.workspace = ChunkPackWorkspace(
            text_ids=text_ids,
            special_tokens=model.llm_special_tokens,
            latent_shape=latent_shape,
            chunk_size=chunk_size,
            patch_size=cfg.patch_spatial,
            device=self.device,
            dtype=self.dtype,
            fps_action=source_fps * speed_factor,
            base_fps=cfg.base_fps,
            reset_spatial=cfg.unified_3d_mrope_reset_spatial_ids,
            modality_margin=cfg.unified_3d_mrope_temporal_modality_margin,
            initial_temporal_offset=initial_temporal_offset,
            action_domain_id=action_domain_id,
        )
        net = model.net
        self.cache = BoundedJointKVCache(
            vision_tokens=self.workspace.vision_tokens,
            chunk_size=chunk_size,
            num_layers=net.num_hidden_layers,
            num_kv_heads=net.num_kv_heads,
            head_dim=net.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        channels, height, width = latent_shape
        self._video = torch.empty(1, channels, chunk_size, height, width, device=self.device)
        self._action = torch.empty(8 * chunk_size, 64, device=self.device)
        self._previous = torch.empty(1, channels, chunk_size + 1, height, width, device=self.device)
        self._condition = torch.empty(1, channels, 1, height, width, device=self.device)
        self._state = torch.empty(1, 64, device=self.device)
        self.sv, self.sa = _schedule(video_schedule, self.device), _schedule(action_schedule, self.device)
        # Keep scalar timesteps on the host; no scalar GPU synchronization in each step.
        self._sv, self._sa = self.sv.cpu().tolist(), self.sa.cpu().tolist()
        self.reset(seed=seed, source_start=source_start)

    @torch.no_grad()
    def reset(self, *, seed=42, source_start=0):
        if not isinstance(source_start, int) or source_start < 0:
            raise ValueError("source_start must be a nonnegative integer")
        self.cache.reset()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.chunk = 0
        self.source_index = source_start
        self._terminal = None
        self._previous_frames = 0
        self._failed = self._closed = False

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward(self, packed):
        result = self.model.denoise(data_batch_packed=packed, memory=self.cache)
        self.cache.ensure_complete()
        return result

    def _check(self, tensor, shape, name, *, padding=False):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        if tensor.device != self.device or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite and on {self.device}")
        if padding and torch.count_nonzero(tensor[:, 57:]):
            raise ValueError("action padding must be zero")

    @torch.no_grad()
    def step(
        self,
        u=None,
        state=None,
        *,
        gt_video=None,
        gt_action=None,
        frames=None,
        input_is_latent=True,
        condition_source="gt",
    ):
        """Return only this block. A partial block closes the episode until reset.

        Physical action integration and clean refresh finish before return.
        generated retains only one [U,V] block and its terminal state; next U is
        decoded/re-encoded at the next call, never from a future GT payload.
        """
        if self._failed or self._closed:
            raise RuntimeError("reset required after a failed call or a partial final block")
        frames = self.chunk_size if frames is None else frames
        if not isinstance(frames, int) or not 1 <= frames <= self.chunk_size:
            raise ValueError("frames must be in 1..chunk_size")
        if condition_source not in ("gt", "observation"):
            raise ValueError("condition_source must be gt or observation")
        self._sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        try:
            result = self._step(u, state, gt_video, gt_action, frames, input_is_latent, condition_source, started)
        except Exception:
            # A partial model/cache write cannot be retried as though it never happened.
            self._failed = True
            raise
        return result

    def _step(self, u, state, gt_video, gt_action, frames, input_is_latent, condition_source, started):
        video, action = self._video[:, :, :frames], self._action[: 8 * frames]
        if self.history in ("gt", "oracle"):
            self._check(gt_video, video.shape, "current GT video")
            self._check(gt_action, action.shape, "current GT action", padding=True)
        elif gt_video is not None or gt_action is not None:
            raise ValueError("pred_history/generated must not receive future GT")
        if self.history == "generated" and self.chunk:
            if u is not None or state is not None:
                raise ValueError("generated continuation must not receive external U/S")
            block = self._previous[:, :, : self._previous_frames + 1]
            decoded = self.model.decode(block.to(self.dtype))
            if (
                decoded.ndim != 5
                or decoded.shape[2] != 1 + 4 * self._previous_frames
                or not torch.isfinite(decoded).all()
            ):
                raise ValueError("VAE must decode complete [U,V] to 1+4C finite frames")
            u = self.model.encode(decoded[:, :, -1:].contiguous()).float()
            state = self._terminal
            condition_source = "prediction"
        else:
            if u is None or state is None:
                raise ValueError("current U and S are required")
            if not input_is_latent:
                if u.ndim != 5 or u.shape[0] != 1 or u.shape[2] != 1 or not torch.isfinite(u).all():
                    raise ValueError("condition RGB must be one finite [1,channels,1,H,W] frame")
                u = self.model.encode(u.to(device=self.device, dtype=self.dtype)).float()
        self._check(u, self._condition.shape, "condition latent")
        self._condition.copy_(u)
        if isinstance(state, torch.Tensor):
            state = decode_chunk_camera_state(state, self.state_normalizer, source_index=self.source_index)
        if not isinstance(state, ChunkCameraState) or state.source_index != self.source_index:
            raise ValueError("state must describe the current absolute source boundary")
        encoded = encode_chunk_camera_state(state, self.state_normalizer)
        self._check(encoded, (57,), "encoded state")
        self._state.zero_()
        self._state[0, :57].copy_(encoded)
        self.workspace.set_boundary(self.source_index, frames)
        text_seconds = 0.0
        if not self.cache.text_ready:
            self._sync()
            text_started = time.perf_counter()
            self.cache.begin_phase("text")
            self._forward(self.workspace.text)
            self._sync()
            text_seconds = time.perf_counter() - text_started
        chunk = self.chunk + 1
        calls_before = self.cache.forward_calls
        self.cache.begin_phase("condition", chunk=chunk, frames=frames)
        self._forward(self.workspace.load("condition", self._condition, self._state))
        video.normal_(generator=self.generator)
        action.normal_(generator=self.generator)
        action[:, 57:] = 0
        if self.history == "oracle":
            video.copy_(gt_video)
        vmax = float(self.model.rectified_flow_video.noise_scheduler.config.num_train_timesteps)
        amax = float(self.model.rectified_flow_action.noise_scheduler.config.num_train_timesteps)
        self.cache.begin_phase("noisy", chunk=chunk, frames=frames)
        visible_cache_tokens = int((self.cache.roles >= 0).sum())
        for step in range(JOINT_STEPS):
            packed = self.workspace.load(
                "noisy",
                video,
                action,
                video_t=0.0 if self.history == "oracle" else self._sv[step] * vmax,
                action_t=self._sa[step] * amax,
            )
            out = self._forward(packed)
            pv = out["preds_vision"][0].float().reshape_as(video)
            pa = out["preds_action"][0].float().reshape_as(action)
            if not torch.isfinite(pv).all() or not torch.isfinite(pa).all():
                raise ValueError(f"non-finite flow at chunk={chunk} step={step}")
            if self.history != "oracle":
                video.add_(pv, alpha=self._sv[step + 1] - self._sv[step])
            action.add_(pa, alpha=self._sa[step + 1] - self._sa[step])
            action[:, 57:] = 0
        if not torch.isfinite(video).all() or not torch.isfinite(action).all():
            raise ValueError("non-finite Euler result")
        # Independent return buffers: GT refresh and subsequent calls cannot overwrite them.
        predicted_v, predicted_a = video.clone(), action.clone()
        decoded_action = decode_chunk_camera_action(state, action, self.future_normalizer)
        self.cache.begin_phase("refresh", chunk=chunk, frames=frames)
        refresh_v, refresh_a = (gt_video, gt_action) if self.history in ("gt", "oracle") else (video, action)
        self._forward(self.workspace.load("refresh", refresh_v, refresh_a))
        if self.cache.forward_calls - calls_before != 32:
            raise AssertionError("each chunk requires 1 condition + 30 noisy + 1 refresh calls")
        if self.history == "generated":
            self._previous[:, :, :1].copy_(self._condition)
            self._previous[:, :, 1 : frames + 1].copy_(video)
            # Own the tiny terminal state; user mutation of returned decoded output is harmless.
            end = decoded_action.end_state
            self._terminal = ChunkCameraState(end.source_index, end.rigid_camera.clone(), end.hand_latents.clone())
            self._previous_frames = frames
        result_u, result_s = self._condition.clone(), self._state[0].clone()
        self.chunk, self.source_index = chunk, self.source_index + 8 * frames
        self._closed = frames < self.chunk_size
        self._sync()
        report = dict(
            chunk=chunk,
            history=self.history,
            condition_source=condition_source,
            boundary_source_index=state.source_index,
            boundary_time=state.source_index / self.source_fps,
            source_stop=self.source_index,
            layout_version=LAYOUT_VERSION,
            action_count=8 * frames,
            bounded_control_api=True,
            end_to_end_seconds=time.perf_counter() - started,
            timing_scope="single_block_input_ready_through_physical_action_and_clean_refresh",
            text_prefill_seconds=text_seconds,
            text_prefill_calls=int(chunk == 1),
            condition_prefill_calls=1,
            noisy_calls=30,
            clean_refresh_calls=1,
            forward_calls=32,
            denoise_steps=30,
            query_tokens=frames * (self.workspace.vision_tokens + 8),
            visible_key_tokens=self.cache.text_len + visible_cache_tokens + frames * (self.workspace.vision_tokens + 8),
            cache_tokens=int((self.cache.roles >= 0).sum()),
            cache_capacity_tokens=self.cache.capacity,
            cache_bytes=sum(t.numel() * t.element_size() for pair in self.cache.kv for t in pair),
            text_cache_bytes=sum(t.numel() * t.element_size() for pair in self.cache.text_kv for t in pair),
            allocated_bytes=(torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else None),
            peak_allocated_bytes=(torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None),
            peak_reserved_bytes=(torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None),
        )
        return StreamingChunk(predicted_v, predicted_a, decoded_action, result_u, result_s, report)
