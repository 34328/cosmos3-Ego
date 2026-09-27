"""Joint video-action autoregressive model: lingbot-va teacher forcing on Cosmos' replayed TF.

Cosmos' ``OmniMoTCausalModel`` runs teacher forcing as two passes over the same pack:
Pass 1 forwards the clean sequence and records its K/V, Pass 2 denoises the noisy
sequence against them. Because clean tokens never read noisy ones, this equals
lingbot-va's single sequence ``[clean, noisy]``. ``EgoVerseARModel`` keeps that
machinery and changes three things:

* action groups of non-conditioning frames are noisy, loss-supervised targets
  (``supervise_temporal_causal_actions``);
* both passes use the lingbot-va visibility rules (``ar_attention``) with a chunk
  size and window drawn per step;
* video and action noise levels are drawn independently per chunk.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F

from .ar_attention import ARChunkLayout, LingbotTeacherForcingAttention, frame_chunk_ids
from .model import EgoVerseLossMixin

try:
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
except ImportError as error:  # pragma: no cover - exercised only outside the Cosmos environment
    raise ImportError(
        "EgoVerseARModel requires PYTHONPATH=<repository-root>:<repository-root>/packages/cosmos3"
    ) from error


@dataclass
class ARStepContext:
    """Chunk/window choice of one forward and the per-chunk noise levels derived from it."""

    chunk_size: int
    window: int | None
    chunk_ids: torch.Tensor | None = None  # [T] chunk index per latent frame
    action_sigmas: torch.Tensor | None = None  # [B,num_chunks]
    action_timesteps: list[torch.Tensor] | None = None  # per action sample, [rows]


class EgoVerseARModel(EgoVerseLossMixin, OmniMoTCausalModel):
    """Lingbot-va style joint video-action AR training with the EgoVerse 57D action loss."""

    def __init__(
        self,
        config,
        lambda_out_of_fov: float = 0.0,
        subblock_equal_weight: bool = False,
        train_chunk_sizes: tuple[int, ...] = (1, 2, 3, 4),
        train_window_range: tuple[int, int] = (4, 64),
        seed: int = 0,
    ):
        super().__init__(config, lambda_out_of_fov=lambda_out_of_fov, subblock_equal_weight=subblock_equal_weight)
        self._validate_ar_config()
        chunk_sizes = tuple(int(size) for size in train_chunk_sizes)
        if not chunk_sizes or min(chunk_sizes) < 1:
            raise ValueError(f"train_chunk_sizes must be positive integers, got {train_chunk_sizes}")
        low, high = (int(value) for value in train_window_range)
        if not 0 <= low <= high:
            raise ValueError(f"train_window_range must satisfy 0 <= low <= high, got {train_window_range}")
        self.train_chunk_sizes = chunk_sizes
        self.train_window_range = (low, high)
        self.ar_seed = int(seed)
        self._ar_step: ARStepContext | None = None

    def _validate_ar_config(self) -> None:
        config = self.config
        required = {
            "video_temporal_causal=True": bool(config.video_temporal_causal),
            "causal_training_strategy='teacher_forcing'": config.causal_training_strategy == "teacher_forcing",
            "teacher_forcing_kv_implementation='singleview_threeway_kv'": (
                self._get_teacher_forcing_kv_implementation() == "singleview_threeway_kv"
            ),
            # Chunking is owned by the lingbot mask; Cosmos' fixed-C truncation must stay off.
            "teacher_forcing_frames_per_chunk=1": int(config.teacher_forcing_frames_per_chunk) == 1,
            "supervise_temporal_causal_actions=True": bool(config.supervise_temporal_causal_actions),
            "action_tokens_per_latent set": config.action_tokens_per_latent is not None,
            "enable_moba=False": not bool(config.enable_moba),
            "teacher_forcing_target_only_no_text_pass2=False": not bool(
                config.teacher_forcing_target_only_no_text_pass2
            ),
            "context_parallel_shard_degree=1": int(config.parallelism.context_parallel_shard_degree) == 1,
        }
        missing = [name for name, ok in required.items() if not ok]
        if missing:
            raise ValueError("EgoVerseARModel requires " + ", ".join(missing))

    @contextlib.contextmanager
    def ar_context(self, chunk_size: int, window: int | None = None):
        """Chunk size and window for teacher-forcing passes outside ``training_step`` (inference)."""
        previous = self._ar_step
        self._ar_step = ARStepContext(chunk_size=int(chunk_size), window=window)
        try:
            yield self._ar_step
        finally:
            self._ar_step = previous

    def _sample_step_context(self, iteration: int) -> ARStepContext:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        rng = random.Random((self.ar_seed * 1_000_003 + int(iteration)) * 4099 + rank)
        return ARStepContext(
            chunk_size=rng.choice(self.train_chunk_sizes),
            window=rng.randint(*self.train_window_range),
        )

    def training_step(self, data_batch, iteration: int):
        self._ar_step = self._sample_step_context(iteration)
        try:
            return super().training_step(data_batch, iteration)
        finally:
            self._ar_step = None

    # ------------------------------------------------------------------ noise
    def _get_train_noise_level_vision(
        self,
        batch_size,
        is_image_batch,
        num_vision_latent_frames,
        resolutions=None,
        num_tokens=None,
        iteration=None,
    ):
        step = self._ar_step
        if step is None or is_image_batch:
            return super()._get_train_noise_level_vision(
                batch_size=batch_size,
                is_image_batch=is_image_batch,
                num_vision_latent_frames=num_vision_latent_frames,
                resolutions=resolutions,
                num_tokens=num_tokens,
                iteration=iteration,
            )
        chunk_ids = frame_chunk_ids(max(num_vision_latent_frames), step.chunk_size)  # [T]
        num_chunks = int(chunk_ids.max()) + 1

        def repeat(values):
            if values is None or isinstance(values, str):
                return values
            return [value for value in values for _ in range(num_chunks)]

        timesteps, sigmas = super()._get_train_noise_level_vision(
            batch_size=batch_size * num_chunks,
            is_image_batch=False,
            num_vision_latent_frames=repeat(list(num_vision_latent_frames)),
            resolutions=repeat(resolutions),
            num_tokens=repeat(num_tokens),
            iteration=iteration,
        )  # [B*num_chunks,1]
        step.chunk_ids = chunk_ids
        index = chunk_ids.to(sigmas.device)
        return (
            timesteps.reshape(batch_size, num_chunks)[:, index],
            sigmas.reshape(batch_size, num_chunks)[:, index],
        )  # [B,T] each

    def _get_train_noise_level_action(self, batch_size, iteration=None):
        step = self._ar_step
        if step is None or step.chunk_ids is None:
            return super()._get_train_noise_level_action(batch_size=batch_size, iteration=iteration)
        num_chunks = int(step.chunk_ids.max()) + 1
        timesteps, sigmas = super()._get_train_noise_level_action(batch_size=batch_size * num_chunks, iteration=iteration)
        step.action_sigmas = sigmas.reshape(batch_size, num_chunks)
        # The caller keeps [B,1] bookkeeping only; the per-chunk values are applied in
        # _add_noise_to_input (noise and timestep embedding) and _compute_losses.
        return timesteps.reshape(batch_size, num_chunks)[:, 1:2], sigmas.reshape(batch_size, num_chunks)[:, 1:2]

    def _add_noise_to_input(
        self,
        gen_data_clean,
        packed_sequence,
        sigmas,
        sigmas_action=None,
        sigmas_sound=None,
        sigmas_lidar=None,
        iteration=None,
    ):
        step = self._ar_step
        if step is None or step.action_sigmas is None:
            return super()._add_noise_to_input(
                gen_data_clean,
                packed_sequence,
                sigmas,
                sigmas_action=sigmas_action,
                sigmas_sound=sigmas_sound,
                sigmas_lidar=sigmas_lidar,
                iteration=iteration,
            )
        actions = gen_data_clean.x0_tokens_action
        if actions is None or len(actions) != step.action_sigmas.shape[0]:
            raise ValueError("per-chunk action noise expects one action tensor per batch sample")
        tokens_per_latent = int(packed_sequence.num_action_tokens_per_supertoken)
        per_row: list[torch.Tensor] = []
        for sample, action in enumerate(actions):
            rows = action.shape[0]
            frames, remainder = divmod(rows, tokens_per_latent)
            if remainder or frames > step.chunk_ids.numel():
                raise ValueError(f"action rows {rows} do not match K={tokens_per_latent} groups of the clip")
            chunk_of_row = step.chunk_ids[:frames].repeat_interleave(tokens_per_latent).to(step.action_sigmas.device)
            per_row.append(step.action_sigmas[sample][chunk_of_row])  # [rows]
        result = super()._add_noise_to_input(
            gen_data_clean,
            packed_sequence,
            sigmas,
            sigmas_action=per_row,
            sigmas_sound=sigmas_sound,
            sigmas_lidar=sigmas_lidar,
            iteration=iteration,
        )
        # Every noisy action token embeds the timestep of its own chunk.
        max_timestep = float(self.rectified_flow_action.noise_scheduler.config.num_train_timesteps)
        step.action_timesteps = [sigma * max_timestep for sigma in per_row]
        noisy_timesteps = [
            step.action_timesteps[sample][rows.to(step.action_timesteps[sample].device)]
            for sample, rows in enumerate(packed_sequence.action.noisy_frame_indexes)
        ]
        packed_sequence.action.timesteps = torch.cat(noisy_timesteps).float().cpu()  # [N_noisy_action]
        packed_sequence.uses_single_timestep = False
        return result

    def _compute_losses(
        self,
        out_net,
        data_batch_packed,
        gen_data_noised,
        timesteps,
        is_image_batch,
        timesteps_action=None,
        timesteps_sound=None,
        timesteps_lidar=None,
    ):
        step = self._ar_step
        if step is not None and step.action_timesteps is not None:
            rows = max(t.numel() for t in step.action_timesteps)
            timesteps_action = torch.stack(
                [F.pad(t, (0, rows - t.numel())) for t in step.action_timesteps]
            ).to(timesteps.device)  # [n_action,rows]
        total_loss, losses = super()._compute_losses(
            out_net=out_net,
            data_batch_packed=data_batch_packed,
            gen_data_noised=gen_data_noised,
            timesteps=timesteps,
            is_image_batch=is_image_batch,
            timesteps_action=timesteps_action,
            timesteps_sound=timesteps_sound,
            timesteps_lidar=timesteps_lidar,
        )
        if step is not None:
            device = total_loss.device
            losses["egoverse_ar_chunk_size"] = torch.tensor(float(step.chunk_size), device=device)
            losses["egoverse_ar_window"] = torch.tensor(float(step.window or 0), device=device)
            if step.action_sigmas is not None:
                losses["egoverse_sigma_action_mean"] = step.action_sigmas[:, 1:].mean().to(device)
        return total_loss, losses

    # ---------------------------------------------------------- attention
    def _build_tf_memory_state(
        self,
        packed_sequence,
        memory_info,
        net=None,
        detach_clean_kv=None,
        selected_clean_gen_token_indexes=None,
    ):
        tf_memory = super()._build_tf_memory_state(
            packed_sequence=packed_sequence,
            memory_info=memory_info,
            net=net,
            detach_clean_kv=detach_clean_kv,
            selected_clean_gen_token_indexes=selected_clean_gen_token_indexes,
        )
        step = self._ar_step
        if step is None:
            raise RuntimeError("lingbot teacher forcing needs a chunk size and window; use training_step or ar_context()")
        vision = packed_sequence.vision
        if vision is None or len(vision.token_shapes) != 1 or len(packed_sequence.sample_lens) != 1:
            raise ValueError("lingbot teacher forcing packs exactly one sample with one video item")
        if packed_sequence.null_action_supertokens:
            raise ValueError("lingbot teacher forcing expects a real (state) action group for frame 0")
        num_frames, patch_h, patch_w = vision.token_shapes[0]
        layout = ARChunkLayout(
            num_frames=int(num_frames),
            action_tokens=int(packed_sequence.num_action_tokens_per_supertoken),
            vision_tokens=int(patch_h) * int(patch_w),
            chunk_size=step.chunk_size,
            window=step.window,
        )
        attention = LingbotTeacherForcingAttention(layout, device=torch.device("cuda", torch.cuda.current_device()))
        read_base = tf_memory.read_for_layer

        def read_for_layer(layer_idx: int):
            value = read_base(layer_idx)
            value.gen_attention_override = attention
            return value

        tf_memory.read_for_layer = read_for_layer
        return tf_memory
