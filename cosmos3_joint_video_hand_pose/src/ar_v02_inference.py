"""Fixed-clip, 30-step U/S-conditioned sampling and full causal-prefix reference.

Only persistent K/V capacity is bounded. Full-clip payloads/output buffers and
per-step select_joint_pack remapping are not a bounded streaming control API.
Reported sampler timings exclude initial GT preparation/VAE and offline export.
"""

import dataclasses
import time

import torch

from .ar_chunk_state import encode_chunk_camera_state
from .action_representation import ActionRepresentationAdapter, FIXED_CAMERA
from .ar_inference import flow_sigmas
from .ar_v02_layout import ACTION, CONDITION_VIDEO, STATE, VIDEO, LAYOUT_VERSION
from .ar_v02_cache import JointKVCache
from .ar_v02_packing import pack_joint_sequence, select_joint_pack

JOINT_STEPS = 30
HISTORY_MODES = ("gt", "oracle", "pred_history", "generated")


def assert_numerically_close(got, reference, *, fp32, context=""):
    """Both per-element and relative error limits, with the documented near-zero rule."""
    got, reference = got.float(), reference.float()
    if not torch.isfinite(got).all() or not torch.isfinite(reference).all():
        raise AssertionError(f"non-finite comparison: {context}")
    atol, rtol, relative_limit = (1e-5, 1e-4, 1e-4) if fp32 else (0.01, 0.03, 0.01)
    torch.testing.assert_close(got, reference, atol=atol, rtol=rtol, msg=context)
    if not reference.numel():
        return
    error = got - reference
    if reference.square().mean().sqrt() < 1e-6:
        assert error.abs().max() <= atol, context
    else:
        relative = error.norm() / reference.norm().clamp_min(1e-6)
        assert relative <= relative_limit, f"{context}: relative L2={relative}"


def _condition_encoding(state, normalizer):
    encoded = encode_chunk_camera_state(state, normalizer)
    if torch.count_nonzero(encoded[:9]):
        raise ValueError("chunk-camera state encoder must produce zero camera slots; old F0 statistics are unsupported")
    return encoded


def _schedule(values, device):
    values = flow_sigmas(JOINT_STEPS, 5) if values is None else torch.as_tensor(values)
    values = values.to(device=device, dtype=torch.float32)
    if (
        values.shape != (JOINT_STEPS + 1,)
        or not torch.isfinite(values).all()
        or values[0] != 1
        or values[-1] != 0
        or not (values[:-1] > values[1:]).all()
    ):
        raise ValueError("schedule must contain 31 finite, strictly decreasing sigmas from 1 to 0")
    return values


class JointARSampler:
    @torch.no_grad()
    def __init__(self, model, batch, *, state_normalizer, future_normalizer, chunk_size=4, source_fps=30.0, hand_codecs=None):
        if chunk_size not in (1, 2, 3, 4) or not 0 < source_fps < float("inf"):
            raise ValueError("v0.2 requires C=1..4 and positive finite source_fps")
        self.model, self.chunk_size, self.source_fps = model, chunk_size, float(source_fps)
        self.state_normalizer, self.future_normalizer = state_normalizer, future_normalizer
        self.action_adapter = ActionRepresentationAdapter(state_normalizer, future_normalizer, hand_codecs)
        self.action_adapter.validate_model(model)
        if self.action_adapter.representation == FIXED_CAMERA and chunk_size != 4:
            raise ValueError("fixed-camera statistics require C=4")
        states = batch["ar_boundary_states"]
        while isinstance(states, list):
            if len(states) != 1:
                raise ValueError("joint inference requires exactly one clip")
            states = states[0]
        while states.ndim > 2 and states.shape[0] == 1:
            states = states.squeeze(0)
        if states.ndim != 2:
            raise ValueError("joint inference requires exactly one boundary-state payload")
        with model.ar_context(chunk_size, 15):
            self.text, self.plans, self.gen, self.memory_info, _, _ = model._prepare_training_data(batch, 0)
        if self.gen.batch_size != 1 or len(self.plans) != 1 or len(self.gen.x0_tokens_vision) != 1:
            raise ValueError("joint inference requires exactly one clip")
        layout = model._joint_layout
        if isinstance(layout, (list, tuple)):
            if len(layout) != 1:
                raise ValueError("joint inference requires exactly one layout")
            layout = layout[0]
        self.layout = layout
        self.gt_video = self.gen.x0_tokens_vision[0].float()
        self.gt_action = self.gen.x0_tokens_action[0].float().reshape(-1, 64)
        self.gt_states = states.to(self.gt_action)
        if self.gt_video.shape[2] != layout.num_video_frames or self.gt_states.shape != (layout.num_frames - 1, 64):
            raise ValueError("video/state payload does not match joint_chunk_cond_v1")
        if torch.count_nonzero(self.gt_states[:, :9]) or torch.count_nonzero(self.gt_states[:, 57:]):
            raise ValueError("chunk-camera states require zero camera and padding slots")
        if not torch.isfinite(self.gt_video).all() or not torch.isfinite(self.gt_action).all():
            raise ValueError("non-finite model payload")
        self.roles, self.chunks, self.sources = layout.action_metadata(device=self.gt_action.device)
        self.chunk_reports = []

    @torch.no_grad()
    def forward(
        self, video, action, *, first_noisy, end, video_sigmas, action_sigmas, clean_video=None, clean_action=None
    ):
        """Recompute the COMPLETE causal prefix, preserving deep historical context."""
        if end not in [b.latent_stop for b in self.layout.boundaries]:
            raise ValueError("prefix must end at a declared chunk boundary")
        layout = dataclasses.replace(self.layout, num_frames=end)
        boundary = next((b for b in layout.boundaries if b.latent_start == first_noisy), None)
        if boundary is None:
            raise ValueError("first_noisy must be a declared chunk boundary")
        n, nv = layout.num_action_rows, layout.num_video_frames
        cv = video if clean_video is None else clean_video
        ca = action if clean_action is None else clean_action
        gen = dataclasses.replace(
            self.gen, x0_tokens_vision=[cv[:, :, :nv].contiguous()], x0_tokens_action=[ca[:n].contiguous()]
        )
        _, video_chunks, _ = layout.video_metadata()
        history_frames = torch.where(video_chunks < boundary.chunk_id)[0].tolist()
        model, old = self.model, self.model._joint_layout
        model._joint_layout = layout
        try:
            with model.ar_context(self.chunk_size, 15):
                tmax = float(model.rectified_flow_video.noise_scheduler.config.num_train_timesteps)
                cfg = model.config.diffusion_expert_config
                packed = pack_joint_sequence(
                    layout=layout,
                    gen_data_clean=gen,
                    text_ids=self.text[0],
                    special_tokens=model.llm_special_tokens,
                    timesteps=(video_sigmas[:nv] * tmax).cpu(),
                    latent_patch_size=cfg.patch_spatial,
                    condition_frames=history_frames,
                    base_fps=cfg.base_fps,
                    reset_spatial=cfg.unified_3d_mrope_reset_spatial_ids,
                    modality_margin=cfg.unified_3d_mrope_temporal_modality_margin,
                    initial_temporal_offset=self.memory_info["initial_temporal_offset"],
                )
                info = model.pre_noise_memory_hook(packed, gen, dict(self.memory_info))
                packed.vision.tokens = [video[:, :, :nv].contiguous()]
                packed.action.tokens = [action[:n].contiguous()]
                noisy_rows = packed.action.noisy_frame_indexes[0].to(action_sigmas.device)
                amax = float(model.rectified_flow_action.noise_scheduler.config.num_train_timesteps)
                packed.action.timesteps = (action_sigmas[:n][noisy_rows] * amax).float().cpu()
                packed.uses_single_timestep = False
                packed.to_cuda()
                model._cast_generated_tokens_to_precision(packed)
                out = model.denoise(data_batch_packed=packed, memory=model.build_memory_state(packed, info))
            return (
                out["preds_vision"][0].float().reshape_as(video[:, :, :nv]),
                out["preds_action"][0].float().reshape(n, 64),
            )
        finally:
            model._joint_layout = old

    @torch.no_grad()
    def _cache_forward(self, video, action, indexes, *, chunk, phase, video_sigma=0.0, action_sigma=0.0):
        from cosmos_framework.model.generator.teacher_forcing import mark_modality_as_clean_condition

        if phase not in ("text", "condition", "noisy", "refresh"):
            raise ValueError("unknown cache phase")
        capture, initial = phase != "noisy", phase == "text"
        self._cache_template.vision.tokens = [video]
        self._cache_template.action.tokens = [action]
        packed = select_joint_pack(self._cache_template, self.layout, indexes, include_text=initial)
        for modality, sigma, flow in (
            (packed.vision, video_sigma, self.model.rectified_flow_video),
            (packed.action, action_sigma, self.model.rectified_flow_action),
        ):
            if modality is not None:
                if capture:
                    mark_modality_as_clean_condition(modality)
                else:
                    modality.timesteps.fill_(float(sigma) * flow.noise_scheduler.config.num_train_timesteps)
        if phase == "refresh" and getattr(self, "history_video_sigma", 0) > 0:
            from .ar_v02_history_noise import history_noise_pack
            generator = torch.Generator(device=video.device).manual_seed(self._history_noise_seed + chunk)
            packed = history_noise_pack(packed,
                [torch.full((packed.vision.tokens[0].shape[2],), self.history_video_sigma, device=video.device)],
                generator=generator, max_timestep=self.model.rectified_flow_video.noise_scheduler.config.num_train_timesteps)
        if self._cache_phase != (chunk, phase):
            self.cache.begin(indexes, chunk=chunk, capture=capture, include_text=initial)
            self._cache_phase = (chunk, phase)
        packed.to_cuda()
        self.model._cast_generated_tokens_to_precision(packed)
        out = self.model.denoise(data_batch_packed=packed, memory=self.cache)
        self.cache.ensure_complete()
        return out

    @torch.no_grad()
    def _next_condition_video(self, block):
        # A future-only decode changes causal VAE context and is not equivalent.
        dtype = self.model.tensor_kwargs["dtype"]
        decoded = self.model.decode(block.to(dtype))
        expected = 1 + 4 * (block.shape[2] - 1)
        if decoded.ndim != 5 or decoded.shape[2] != expected or not torch.isfinite(decoded).all():
            raise ValueError("VAE must decode the complete [U,V] block to 1+4C frames")
        encoded = self.model.encode(decoded[:, :, -1:].contiguous()).float()
        if encoded.shape != block[:, :, :1].shape or not torch.isfinite(encoded).all():
            raise ValueError("single-frame re-encode must produce exactly one finite condition latent")
        return encoded

    @torch.no_grad()
    def sample(
        self,
        *,
        history="gt",
        seed=42,
        steps=JOINT_STEPS,
        use_cache=True,
        verify_cache=False,
        video_schedule=None,
        action_schedule=None,
        history_video_sigma=0.0,
    ):
        if steps != JOINT_STEPS:
            raise ValueError("AR v0.2 requires exactly 30 joint Euler steps")
        if history not in HISTORY_MODES:
            raise ValueError("unsupported history mode")
        if verify_cache and not use_cache:
            raise ValueError("verify_cache requires use_cache")
        if not 0 <= history_video_sigma <= 1:
            raise ValueError("history_video_sigma must be finite and in [0,1]")
        if history_video_sigma and (not use_cache or verify_cache):
            raise ValueError("history noise currently requires persistent cache without clean-reference verification")
        self.history_video_sigma = float(history_video_sigma)
        self._history_noise_seed = int(seed) * 1_000_003 + 17
        device = self.gt_video.device
        generator = torch.Generator(device=device).manual_seed(seed)
        vr, vc, _ = self.layout.video_metadata(device=device)
        gr, gc, _ = self.layout.metadata()
        video = torch.zeros_like(self.gt_video)
        vf = vr == VIDEO
        video[:, :, vf] = torch.randn(video[:, :, vf].shape, device=device, generator=generator)
        action = torch.zeros_like(self.gt_action)
        future = self.roles == ACTION
        action[future] = torch.randn((int(future.sum()), 64), device=device, generator=generator)
        action[:, 57:] = 0
        output_v, output_a = torch.zeros_like(video), torch.zeros_like(action)
        sv, sa = _schedule(video_schedule, device), _schedule(action_schedule, device)
        self.chunk_reports, self.condition_reports = [], []
        self.cache_prefill_seconds = 0.0
        if use_cache:
            model, net = self.model, self.model.net
            self.cache = JointKVCache(
                self.layout,
                num_layers=net.num_hidden_layers,
                num_kv_heads=net.num_kv_heads,
                head_dim=net.head_dim,
                device=device,
                dtype=model.tensor_kwargs["dtype"],
            )
            self._cache_phase = None
            with model.ar_context(self.chunk_size, 15):
                self._cache_template = model._pack_input_sequence(
                    self.plans,
                    self.text,
                    self.gen,
                    torch.zeros(1, self.layout.num_video_frames),
                    initial_mrope_temporal_offset=self.memory_info["initial_temporal_offset"],
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            self._cache_forward(video, action, [], chunk=0, phase="text")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.cache_prefill_seconds = time.perf_counter() - started
        state = previous_block = None
        for b in self.layout.boundaries:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            calls_before = self.cache.forward_calls if use_cache else 0
            u = self.layout.video_indexes(b.chunk_id, condition=True).to(device)
            vi = self.layout.video_indexes(b.chunk_id, condition=False).to(device)
            block_indexes = self.layout.video_indexes(b.chunk_id).to(device)
            rows = torch.where((self.roles == ACTION) & (self.chunks == b.chunk_id))[0]
            sr = (self.roles == STATE) & (self.chunks == b.chunk_id)
            target = torch.where((gc == b.chunk_id) & ((gr == ACTION) | (gr == VIDEO)))[0]
            if history != "generated" or b.chunk_id == 1:
                state = self.action_adapter.decode_state(
                    self.gt_states[b.latent_start - 1], source_index=b.source_start
                )
                video[:, :, u] = self.gt_video[:, :, u]
            else:
                video[:, :, u] = self._next_condition_video(previous_block)
            encoded = self.action_adapter.encode_state(state)
            action[sr, :57] = encoded
            action[sr, 57:] = 0
            output_a[sr], output_v[:, :, u] = action[sr], video[:, :, u]
            condition_source = "prediction" if history == "generated" and b.chunk_id > 1 else "gt"
            condition = dict(
                chunk=b.chunk_id,
                condition_source=condition_source,
                boundary_source_index=b.source_start,
                boundary_time=b.source_start / self.source_fps,
            )
            self.condition_reports.append(condition)
            if use_cache:
                self._cache_forward(
                    video,
                    action,
                    self.layout.condition_prefill_indexes(b.chunk_id),
                    chunk=b.chunk_id,
                    phase="condition",
                )
            if history == "oracle":
                video[:, :, vi] = self.gt_video[:, :, vi]
            if not use_cache or verify_cache:
                vs = torch.zeros(self.layout.num_video_frames, device=device)
                acs = torch.zeros(self.layout.num_action_rows, device=device)
            for i in range(JOINT_STEPS):
                v_sigma = 0.0 if history == "oracle" else sv[i]
                if not use_cache or verify_cache:
                    vs[vi], acs[rows] = v_sigma, sa[i]
                if use_cache:
                    out = self._cache_forward(
                        video, action, target, chunk=b.chunk_id, phase="noisy", video_sigma=v_sigma, action_sigma=sa[i]
                    )
                    pv = out["preds_vision"][0].float().reshape_as(video[:, :, vi])
                    pa = out["preds_action"][0].float().reshape(-1, 64)
                    if verify_cache:
                        rv, ra = self.forward(
                            video,
                            action,
                            first_noisy=b.latent_start,
                            end=b.latent_stop,
                            video_sigmas=vs,
                            action_sigmas=acs,
                        )
                        for name, got, ref in (("video", pv, rv[:, :, vi]), ("action", pa, ra[rows])):
                            assert_numerically_close(
                                got,
                                ref,
                                fp32=self.model.tensor_kwargs["dtype"] == torch.float32,
                                context=f"{name} chunk={b.chunk_id} step={i}",
                            )
                else:
                    pv, pa = self.forward(
                        video, action, first_noisy=b.latent_start, end=b.latent_stop, video_sigmas=vs, action_sigmas=acs
                    )
                    pv, pa = pv[:, :, vi], pa[rows]
                if not torch.isfinite(pv).all() or not torch.isfinite(pa).all():
                    raise ValueError(f"non-finite flow at chunk={b.chunk_id} step={i}")
                if history != "oracle":
                    video[:, :, vi] += (sv[i + 1] - sv[i]) * pv
                action[rows] += (sa[i + 1] - sa[i]) * pa
                action[rows, 57:] = 0
            output_v[:, :, vi], output_a[rows] = video[:, :, vi], action[rows]
            terminal = self.action_adapter.decode_action(state, action[rows]).end_state
            if history == "generated":
                state = terminal
                previous_block = video[:, :, block_indexes].clone()
            # Save predictions BEFORE replacing the working history by GT.
            if history in ("gt", "oracle"):
                video[:, :, vi], action[rows] = self.gt_video[:, :, vi], self.gt_action[rows]
            if use_cache:
                self._cache_forward(video, action, target, chunk=b.chunk_id, phase="refresh")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if use_cache and self.cache.forward_calls - calls_before != 32:
                raise AssertionError("each chunk must perform exactly 1 condition + 30 noisy + 1 refresh forwards")
            self.chunk_reports.append(
                dict(
                    **condition,
                    source_start=b.source_start,
                    source_stop=b.source_stop,
                    layout_version=LAYOUT_VERSION,
                    history=history,
                    action_count=b.action_count,
                    bounded_control_api=False,
                    timing_scope="fixed_clip_sampler_excludes_initial_GT_preparation_and_offline_export",
                    end_to_end_seconds=time.perf_counter() - started,
                    denoise_steps=JOINT_STEPS,
                    forward_calls=self.cache.forward_calls - calls_before if use_cache else 2 * JOINT_STEPS,
                    condition_prefill_calls=1 if use_cache else 0,
                    noisy_calls=JOINT_STEPS,
                    clean_refresh_calls=1 if use_cache else 0,
                    text_prefill_calls=0,
                    cache_mode="persistent" if use_cache else "prefix_recompute",
                    includes_reference_checks=verify_cache,
                    verification_scope="same_input_flow_only" if verify_cache else None,
                    reference_forward_calls=2 * JOINT_STEPS if verify_cache else 0,
                    peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                )
            )
        return output_v, output_a

    def future_action(self, payload):
        _, future, source = self.layout.unpack_action(payload)
        return future, source
