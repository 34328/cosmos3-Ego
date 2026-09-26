# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Temporal-causal supertoken packing helpers."""

import math

import torch

from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens
from cosmos_framework.data.generator.sequence_packing.runtime import to_device_nonblocking
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder


def pack_supertokens_temporal_causal(
    seq_builder: PackedSequenceBuilder,
    input_vision_tokens: torch.Tensor,
    input_action_tokens: torch.Tensor | None,
    condition_frame_indexes_vision: list[int],
    input_timestep: float | torch.Tensor,
    latent_patch_size: int,
    temporal_compression_factor: int,
    action_dim: int,
    vision_fps: float | None = None,
    action_fps: float | None = None,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    pack_action_tokens: bool = True,
    action_tokens_per_latent: int | None = None,
    supervise_action_tokens: bool = False,
) -> tuple[int, bool]:
    """Pack vision and (optionally) action tokens in supertoken order for temporal causal attention.

    Buffer layout per frame:
        pack_action_tokens=True:  [action_t (K), vision_t (H*W)]    — supertoken size K + H*W
        pack_action_tokens=False: [vision_t (H*W)]                  — supertoken size H*W

    ``K = action_tokens_per_latent`` is the number of action tokens per latent
    frame; ``None`` (default) means ``K = tcf``, the historical layout. ``tcf``
    itself remains the video clock unit of the mRoPE positions.

    Use ``pack_action_tokens=False`` when ``config.action_gen=False``; the resulting
    ``num_action_tokens_per_supertoken=0`` is stamped on the pack and read by the
    attention builder so NATTEN metadata stays in sync automatically.

    mRoPE layout (with actions, unified_3d_mrope only). The layout is inferred from the
    action tensor shape:
        - Whole-clip training (frame 0 is the clean conditioning frame, so
          ``real_actions`` has ``(T-1)*K`` rows): null action for supertoken 0, real
          actions for frames 1..T-1 with ``start_frame_offset=1`` so the last action in
          group i co-locates with vision frame i; vision uses ``start_frame_offset=0``.
        - AR generation, single frame OR chunk (every frame carries a real action, so
          ``real_actions`` has ``latent_t*K`` rows): vision AND action both use
          ``start_frame_offset=1``, generalizing the single-frame AR supertoken to
          ``latent_t`` frames. The caller (``pack_input_sequence_autoregressive``)
          seeds ``temporal_offset`` one frame-stride back to compensate, so the unit
          lands at the same absolute positions as the whole-clip training pack.
        - Interleaved per frame as cat([action_ids, vision_ids]).

    With FPS modulation, the last action of each group co-locates with its vision
    latent only if ``action_fps == vision_fps * K / tcf``; for ``K != tcf`` this is
    validated whenever both FPS values are given.

    ``input_timestep`` is float (TF/none) or Tensor(T_max,) (DF, per-frame sigma).
    Conditioning frames are excluded from mse_loss_indexes either way.

    Args:
        seq_builder: Mutable sequence builder receiving packed spans and metadata.
        input_vision_tokens: Vision latent tokens with shape ``[1, C, T, H, W]``.
        input_action_tokens: Optional action tokens. Whole-clip training uses
            ``(T - 1) * K`` rows; AR chunks use ``T * K`` rows.
        condition_frame_indexes_vision: Vision frame indexes treated as clean conditioning.
        input_timestep: Diffusion timestep as a scalar float or per-frame tensor.
        latent_patch_size: Spatial patch size used to derive the vision patch grid.
        temporal_compression_factor: VAE temporal compression factor (video frames
            per latent frame); the mRoPE clock unit. Also the default ``K``.
        action_dim: Action feature dimension used when null action tokens are created.
        vision_fps: Optional video FPS for FPS-modulated vision mRoPE positions.
        action_fps: Optional action FPS for FPS-modulated action mRoPE positions.
        enable_fps_modulation: If True, scale temporal position IDs using FPS values.
        base_fps: Base FPS for temporal position normalization.
        pack_action_tokens: If True, append action tokens before each vision supertoken.
            If False, append only vision tokens and report no null-action supertokens.
        action_tokens_per_latent: Action tokens per latent frame (``K``). ``None``
            uses ``temporal_compression_factor``.
        supervise_action_tokens: If False (default), every action token is a clean
            condition (forward dynamics). If True, the action group of every
            non-conditioning vision frame is a noisy, loss-supervised target
            (joint video-action generation) with that frame's timestep; the
            groups of conditioning frames stay clean.

    Returns:
        Tuple of ``(total_split_len, null_action_flag)``. ``null_action_flag`` is False
        when ``pack_action_tokens=False``.
    """
    _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    tcf = temporal_compression_factor
    K = tcf if action_tokens_per_latent is None else int(action_tokens_per_latent)
    if K < 1:
        raise ValueError(f"action_tokens_per_latent must be a positive integer or None, got {action_tokens_per_latent}.")
    if pack_action_tokens and K != tcf and vision_fps is not None and action_fps is not None:
        expected_action_fps = float(vision_fps) * K / tcf
        if not math.isclose(float(action_fps), expected_action_fps, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"action_tokens_per_latent={K} with temporal_compression_factor={tcf} requires "
                f"action_fps == vision_fps * K / tcf = {vision_fps} * {K} / {tcf} = {expected_action_fps}, "
                f"got action_fps={action_fps}. Each group of K action tokens spans one video latent "
                "(tcf video frames), so the last action of every group must end at the same time as its "
                "vision latent."
            )
    if supervise_action_tokens and (not pack_action_tokens or input_action_tokens is None):
        raise ValueError("supervise_action_tokens=True requires pack_action_tokens=True and real action tokens.")
    patches_per_frame = patch_h * patch_w

    vision = seq_builder.ensure_vision()
    action = seq_builder.ensure_action() if pack_action_tokens else None

    device = input_vision_tokens.device
    dtype = input_vision_tokens.dtype

    null_action_flag: bool
    if pack_action_tokens:
        # Build all_action_tokens: shape (latent_t * K, action_dim)
        #
        # Cases (token assembly; mRoPE start_frame_offset is chosen separately below,
        # inferred from the same action shape):
        #   1. Whole-clip training with conditioning frame (latent_t > 1, real_actions
        #      has (T-1)*K rows): prepend K null tokens for frame 0, then real
        #      actions for frames 1..T-1.
        #   2. AR generation (every frame has a real action, real_actions has
        #      latent_t*K rows — single frame OR chunk): no null prefix.
        #   3. AR frame 0 / image2video (action is None): all null tokens.
        if input_action_tokens is not None:
            # input_action_tokens shape: (1, T*K, D) or (T*K, D) for training; (T*K, D) for AR units
            if input_action_tokens.dim() == 3:
                real_actions = input_action_tokens.squeeze(0)  # [T*K,action_dim] or [N,action_dim]
            else:
                real_actions = input_action_tokens  # [N,action_dim]
            null_tokens = real_actions.new_zeros((K, action_dim))  # [K,action_dim]
            if real_actions.shape[0] == latent_t * K:
                # AR generation (single frame: K == 1*K, or chunk: latent_t*K):
                # every supertoken carries a real action, no null prefix.
                all_action_tokens = real_actions
                null_action_flag = False
            elif real_actions.shape[0] == (latent_t - 1) * K:
                # Conditioning frame present: null for supertoken 0, real for 1..T-1
                all_action_tokens = torch.cat([null_tokens, real_actions], dim=0)  # [T*K,action_dim]
                null_action_flag = True
            else:
                raise ValueError(
                    "Temporal-causal action tokens must have either latent_t*K rows for AR chunks "
                    f"or (latent_t-1)*K rows for whole-clip training; got {real_actions.shape[0]} rows "
                    f"for latent_t={latent_t}, K(action_tokens_per_latent)={K}, tcf={tcf}."
                )
        else:
            # AR frame 0 or image2video: all action tokens are null
            all_action_tokens = torch.zeros(
                latent_t * K, action_dim, device=device, dtype=dtype
            )  # [T*K,action_dim]
            null_action_flag = True
    else:
        # pack_action_tokens=False: action tokens must not be supplied.
        assert input_action_tokens is None, (
            "pack_action_tokens=False requires input_action_tokens=None; got a non-None tensor."
        )
        null_action_flag = False

    # Record vision token shapes and tokens
    vision.token_shapes.append((latent_t, patch_h, patch_w))
    vision.tokens.append(input_vision_tokens)
    vision_payload_index = len(vision.tokens) - 1

    # Vision conditioning mask: (T, 1, 1)
    condition_set_vision = {idx for idx in condition_frame_indexes_vision if 0 <= idx < latent_t}
    # Built on the host and moved asynchronously: writing Python scalars into a CUDA tensor
    # element by element synchronises the host with the device on every call.
    vision_condition_mask = torch.zeros((latent_t, 1, 1), dtype=dtype)  # [T,1,1]
    for fidx in condition_set_vision:
        vision_condition_mask[fidx, 0, 0] = 1.0
    vision.condition_mask.append(to_device_nonblocking(vision_condition_mask, device))

    vision_noisy_frame_indexes = to_device_nonblocking(
        torch.tensor([idx for idx in range(latent_t) if idx not in condition_set_vision], dtype=torch.long),
        device,
    )  # [N_noisy_frames]
    vision.noisy_frame_indexes.append(vision_noisy_frame_indexes)

    if pack_action_tokens:
        assert action is not None
        # Action token shapes: latent_t * K total (including null tokens)
        action.token_shapes.append((latent_t * K,))
        action.tokens.append(all_action_tokens)
        action_payload_index = len(action.tokens) - 1

        if supervise_action_tokens:
            # Joint video-action layout: an action group is a clean condition iff its
            # vision frame is; every other group is a noisy, supervised target.
            if null_action_flag and 0 not in condition_set_vision:
                raise ValueError("supervise_action_tokens requires the null action frame 0 to be a condition frame.")
            action_condition_mask = torch.zeros((latent_t * K, 1), dtype=dtype)  # [T*K,1]
            for fidx in condition_set_vision:
                action_condition_mask[fidx * K : (fidx + 1) * K, 0] = 1.0
            action.condition_mask.append(to_device_nonblocking(action_condition_mask, device))
            action_noisy_rows = torch.tensor(
                [
                    row
                    for fidx in range(latent_t)
                    if fidx not in condition_set_vision
                    for row in range(fidx * K, (fidx + 1) * K)
                ],
                dtype=torch.long,
            )  # [N_noisy_action_rows]
            action.noisy_frame_indexes.append(to_device_nonblocking(action_noisy_rows, device))
        else:
            # Action conditioning mask: all action tokens are conditioning (not supervised)
            # Null tokens are always conditioning; real actions are conditioning too (they are inputs)
            action_condition_mask = torch.ones((latent_t * K, 1), device=device, dtype=dtype)  # [T*K,1]
            action.condition_mask.append(action_condition_mask)

    # Pack in interleaved supertoken order: [action_t, vision_t] for each frame t
    # (or just [vision_t] per frame when pack_action_tokens=False)
    total_split_len = 0

    # Snapshot the offset before this sample and compute mRoPE IDs.
    temporal_offset = seq_builder._mrope_temporal_offset
    effective_vision_fps = vision_fps if enable_fps_modulation else None

    # AR generation (single frame OR chunk) is detected by every frame carrying a
    # real action (``real_actions`` has ``latent_t*K`` rows). There, vision AND
    # action both use start_frame_offset=1 so the last action in each group
    # co-locates with its vision frame, mirroring whole-clip training; the caller
    # (pack_input_sequence_autoregressive) seeds temporal_offset one frame-stride
    # back to compensate. Whole-clip training (frame 0 is the null conditioning
    # frame, ``real_actions`` has ``(T-1)*K`` rows) keeps vision start_frame_offset=0.
    all_frames_have_real_action = (
        pack_action_tokens and input_action_tokens is not None and real_actions.shape[0] == latent_t * K
    )
    vision_sfo = 1 if all_frames_have_real_action else 0

    # Vision mRoPE keeps the VAE temporal compression factor (video clock), not K.
    vision_ids_flat, new_offset = get_3d_mrope_ids_vae_tokens(
        grid_t=latent_t,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=temporal_offset,
        reset_spatial_indices=seq_builder._mrope_reset_spatial,
        fps=effective_vision_fps,
        base_fps=base_fps,
        temporal_compression_factor=tcf,
        start_frame_offset=vision_sfo,
    )  # vision_ids_flat: [3,T*patch_h*patch_w]
    vision_ids_3d = vision_ids_flat.reshape(3, latent_t, patches_per_frame)  # [3,T,patch_h*patch_w]

    action_ids_3d: torch.Tensor | None = None
    if pack_action_tokens:
        effective_action_fps = action_fps if enable_fps_modulation else None

        # Action IDs. Real action tokens use start_frame_offset=1 so the last
        # sub-token of a group co-locates with its vision frame. Whole-clip training
        # has a null action at frame 0 (the conditioning frame); AR units have a real
        # action for every frame.
        fps_active = effective_action_fps is not None
        t_dtype = torch.float32 if fps_active else torch.long
        t_offset = float(temporal_offset) if fps_active else int(temporal_offset)
        null_t = torch.full((K,), t_offset, dtype=t_dtype)  # [K]
        null_hw = torch.zeros(K, dtype=t_dtype)  # [K]
        null_ids = torch.stack([null_t, null_hw, null_hw])  # [3,K]

        def _real_action_ids(n_frames: int, start_frame_offset: int) -> torch.Tensor:
            # K action tokens per latent frame, each advancing one action period;
            # base_temporal_compression_factor stays tcf so the mRoPE unit matches vision.
            flat, _ = get_3d_mrope_ids_vae_tokens(
                grid_t=n_frames * K,
                grid_h=1,
                grid_w=1,
                temporal_offset=temporal_offset,
                reset_spatial_indices=seq_builder._mrope_reset_spatial,
                fps=effective_action_fps,
                base_fps=base_fps,
                temporal_compression_factor=1,
                base_temporal_compression_factor=tcf,
                start_frame_offset=start_frame_offset,
            )
            return flat.reshape(3, n_frames, K)  # [3,n_frames,K]

        if all_frames_have_real_action:
            # AR generation (single frame: K == 1*K, or chunk: latent_t*K):
            # every supertoken carries a real action. start_frame_offset=1 puts
            # a_{j-1}'s last sub-token on vision frame j -- the whole-clip TF
            # training layout. The caller seeds temporal_offset (N-1) frame-strides
            # back to compensate.
            action_ids_3d = _real_action_ids(latent_t, start_frame_offset=1)  # [3,T,K]
        elif latent_t > 1:
            # Whole-clip training: supertoken 0 = null (conditioning frame), frames
            # 1..T-1 = real with start_frame_offset=1. Covers real-action training
            # (real_actions has (T-1)*K rows) and the architectural all-null layout
            # (input_action_tokens is None); the tokens differ but the IDs match.
            null_ids_3d = null_ids.reshape(3, 1, K)  # [3,1,K]
            real_ids_3d = _real_action_ids(latent_t - 1, start_frame_offset=1)  # [3,T-1,K]
            action_ids_3d = torch.cat([null_ids_3d, real_ids_3d], dim=1)  # [3,T,K]
        else:
            # AR frame 0 / image2video (latent_t == 1, no action): only null.
            action_ids_3d = null_ids.reshape(3, 1, K)  # [3,1,K]

    seq_builder._mrope_temporal_offset = new_offset

    for frame_t in range(latent_t):
        if pack_action_tokens:
            assert action is not None
            assert action_ids_3d is not None
            # Pack action tokens for this frame (indexes only; tokens already stored in seq_builder.action.tokens)
            action_position_ids = action_ids_3d[:, frame_t, :]  # [3,K]
            action_frame_indexes = seq_builder.append_action_span(
                K,
                action_position_ids,
                payload_index=action_payload_index,
                payload_start=frame_t * K,
                payload_shape=(K,),
            )
            total_split_len += K
            # Action tokens are in the MSE loss only when supervised (never for condition frames).
            if supervise_action_tokens and frame_t not in condition_set_vision:
                action.mse_loss_indexes.extend(action_frame_indexes)
                action_ts = (
                    input_timestep[frame_t].item() if isinstance(input_timestep, torch.Tensor) else input_timestep
                )
                action.timesteps.extend([action_ts] * K)

        # Pack vision tokens for this frame
        vision_position_ids = vision_ids_3d[:, frame_t, :]  # [3,patch_h*patch_w]
        frame_indexes = seq_builder.append_vision_span(
            patches_per_frame,
            vision_position_ids,
            payload_index=vision_payload_index,
            payload_start=frame_t * patches_per_frame,
            payload_shape=(1, patch_h, patch_w),
        )
        total_split_len += patches_per_frame

        # Vision MSE loss: supervise non-conditioning frames
        if frame_t not in condition_set_vision:
            vision.mse_loss_indexes.extend(frame_indexes)
            frame_ts = input_timestep[frame_t].item() if isinstance(input_timestep, torch.Tensor) else input_timestep
            vision.timesteps.extend([frame_ts] * patches_per_frame)

    return total_split_len, null_action_flag
