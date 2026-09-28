"""Native Cosmos builder with explicit per-chunk conditions and sample isolation."""

import math
import dataclasses
import torch
from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder
from .ar_v02_layout import ACTION, STATE, VIDEO, CONDITION_VIDEO, JointChunkLayout, LAYOUT_VERSION


def pack_joint_sequence(
    *,
    layout,
    gen_data_clean,
    text_ids,
    special_tokens,
    timesteps,
    latent_patch_size,
    condition_frames=(),
    base_fps=24.0,
    reset_spatial=True,
    modality_margin=0,
    initial_temporal_offset=0
):
    single = isinstance(layout, JointChunkLayout)
    layouts = [layout] if single else list(layout)
    texts = [text_ids] if single else text_ids
    times = [timesteps] if single else timesteps
    conds = [condition_frames] if single else condition_frames
    if len(layouts) != gen_data_clean.batch_size or len(texts) != len(layouts):
        raise ValueError("sample payloads and explicit layouts must match")
    builder = PackedSequenceBuilder(uses_single_timestep=False)
    builder._mrope_reset_spatial = reset_spatial
    state_masks = []
    video_masks = []
    text_lengths = []
    for i, lay in enumerate(layouts):
        video = gen_data_clean.x0_tokens_vision[i]
        actions = gen_data_clean.x0_tokens_action[i].reshape(-1, 64)
        _, _, t, h, w = video.shape
        ph, pw = math.ceil(h / latent_patch_size), math.ceil(w / latent_patch_size)
        if (t, ph * pw) != (lay.num_video_frames, lay.vision_tokens) or actions.shape != (lay.num_action_rows, 64):
            raise ValueError("payload geometry differs from declared joint_chunk_cond_v1 layout")
        vf = float(gen_data_clean.fps_vision.reshape(-1)[i])
        af = float(gen_data_clean.fps_action.reshape(-1)[i])
        if vf <= 0 or not math.isclose(af, vf * 2, rel_tol=1e-5):
            raise ValueError("video/action timestamps require action FPS = 2 * RGB FPS")
        builder.begin_sample(initial_temporal_offset)
        nt = builder.pack_text_tokens(texts[i], special_tokens, has_generation=True, use_float_positions=True)
        text_lengths.append(nt)
        builder.advance_mrope_temporal_offset(modality_margin)
        offset = builder.mrope_temporal_offset
        vision = builder.ensure_vision()
        action = builder.ensure_action()
        vision.tokens.append(video)
        vision.token_shapes.append((t, ph, pw))
        action.tokens.append(actions)
        action.token_shapes.append((lay.num_action_rows,))
        vr, vc, vs = lay.video_metadata(device=video.device)
        ar, ac, src = lay.action_metadata(device=actions.device)
        vm = vr == CONDITION_VIDEO
        am = ar == STATE
        extra = set(conds[i])
        if any(f < 0 or f >= t for f in extra):
            raise ValueError("conditioned frame outside payload")
        for f in extra:
            vm[f] = True
            if int(vr[f]) == VIDEO:
                am |= (ar == ACTION) & (src > vs[f] - 8) & (src <= vs[f])
        vision.condition_mask.append(vm[:, None, None].to(video.dtype))
        vision.noisy_frame_indexes.append(torch.where(~vm)[0])
        action.condition_mask.append(am[:, None].to(actions.dtype))
        action.noisy_frame_indexes.append(torch.where(~am)[0])
        state_masks.append(ar == STATE)
        video_masks.append((vr == CONDITION_VIDEO).repeat_interleave(ph * pw))
        vpos, _ = get_3d_mrope_ids_vae_tokens(
            t,
            ph,
            pw,
            offset,
            reset_spatial_indices=reset_spatial,
            fps=vf,
            base_fps=base_fps,
            temporal_compression_factor=4,
            start_frame_offset=0,
        )
        vpos[0] = offset + vs.cpu().float().repeat_interleave(ph * pw) * (base_fps / 4) / af
        apos = torch.zeros(3, lay.num_action_rows, dtype=torch.float32)
        if not reset_spatial:
            apos.fill_(offset)
        apos[0] = offset + src.cpu().float() * (base_fps / 4) / af
        ts = torch.as_tensor(times[i]).reshape(-1)
        # Noise sampling uses a padded [batch,max_frames] matrix. Preserve each
        # sample's actual frame count; padding is metadata, not an extra frame.
        if ts.numel() > t:
            if torch.count_nonzero(ts[t:]):
                raise ValueError("nonzero timestep beyond sample video length")
            ts = ts[:t]
        if ts.numel() not in (1, t):
            raise ValueError("one timestep per packed video frame required")
        for role, chunk, start, count, source in lay.spans():
            if role in (VIDEO, CONDITION_VIDEO):
                f = start // lay.vision_tokens
                indexes = builder.append_vision_span(
                    count,
                    vpos[:, start : start + count],
                    payload_index=i,
                    payload_start=start,
                    payload_shape=(1, ph, pw),
                )
                if not vm[f]:
                    vision.mse_loss_indexes.extend(indexes)
                    vision.timesteps.extend([float(ts[0 if ts.numel() == 1 else f])] * count)
            else:
                indexes = builder.append_action_span(
                    count, apos[:, start : start + count], payload_index=i, payload_start=start, payload_shape=(count,)
                )
                if role == ACTION and not am[start]:
                    f = int(torch.where((vc == chunk) & (vr == VIDEO) & (vs >= source))[0][0])
                    action.mse_loss_indexes.extend(indexes)
                    action.timesteps.extend([float(ts[0 if ts.numel() == 1 else f])] * count)
        builder.vision_item_split_lens.append([lay.num_tokens])
        builder.finish_sample(lay.num_tokens, nt + lay.num_tokens)
    packed = builder.finalize(gen_data_clean)
    packed.action_state_mask = torch.cat(state_masks).cpu()
    packed.vision_condition_type_mask = torch.cat(video_masks).cpu()
    packed.joint_layout_version = LAYOUT_VERSION
    packed.joint_layouts = layouts
    packed.joint_text_lengths = text_lengths
    return packed


def select_joint_pack(packed, layout, gen_indexes, *, include_text):
    """Select explicit GEN tokens for prefill/denoise/refresh, preserving absolute RoPE.

    Vision selections must contain complete latent frames. State-only prefill
    has no vision payload. No mode is inferred from the action row count.
    """
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence

    selected = torch.as_tensor(gen_indexes, dtype=torch.long).cpu()
    if (selected.numel() == 0 and not include_text) or not torch.equal(selected, selected.unique(sorted=True)):
        raise ValueError("GEN selection must be non-empty, sorted and unique")
    if selected.numel() and (selected[0] < 0 or selected[-1] >= layout.num_tokens):
        raise ValueError("GEN selection exceeds its declared layout")
    text_length = packed.text_indexes.numel()
    global_indexes = selected + text_length
    if include_text:
        global_indexes = torch.cat((packed.text_indexes.cpu(), global_indexes))
    remap = torch.full((packed.sequence_length,), -1, dtype=torch.long)
    remap[global_indexes] = torch.arange(global_indexes.numel())

    def select_modality(modality, vision=False):
        if modality is None:
            return None, None
        original = modality.sequence_indexes.cpu()
        keep = remap[original] >= 0
        if not keep.any():
            return None, None
        if vision:
            t, ph, pw = modality.token_shapes[0]
            frame_keep = keep.reshape(t, ph * pw)
            if not (frame_keep.all(1) == frame_keep.any(1)).all():
                raise ValueError("a selected video latent must include every spatial patch")
            rows = torch.where(frame_keep.all(1))[0]
            payload = modality.tokens[0].index_select(2, rows.to(modality.tokens[0].device))
            shape = (rows.numel(), ph, pw)
        else:
            rows = torch.where(keep)[0]
            payload = modality.tokens[0].index_select(0, rows.to(modality.tokens[0].device))
            shape = (rows.numel(),)
        condition = modality.condition_mask[0].index_select(0, rows.to(modality.condition_mask[0].device))
        noisy = torch.where(condition.reshape(rows.numel(), -1)[:, 0] == 0)[0]
        mse_keep = remap[modality.mse_loss_indexes.cpu()] >= 0
        domain_id = modality.domain_id
        if domain_id and domain_id[0].numel() > 1:
            domain_id = [domain_id[0].reshape(-1).index_select(0, rows.to(domain_id[0].device))]
        return (
            dataclasses.replace(
                modality,
                sequence_indexes=remap[original[keep]],
                mse_loss_indexes=remap[modality.mse_loss_indexes.cpu()[mse_keep]],
                timesteps=modality.timesteps[mse_keep.to(modality.timesteps.device)],
                token_shapes=[shape],
                tokens=[payload],
                condition_mask=[condition],
                noisy_frame_indexes=[noisy],
                domain_id=domain_id,
            ),
            rows,
        )

    vision, vision_rows = select_modality(packed.vision, vision=True)
    action, action_rows = select_modality(packed.action)
    nt = text_length if include_text else 0
    return PackedSequence(
        sample_lens=[global_indexes.numel()],
        split_lens=([nt] if include_text else []) + [selected.numel()],
        attn_modes=(["causal"] if include_text else []) + ["full"],
        is_image_batch=False,
        uses_single_timestep=False,
        sequence_length=global_indexes.numel(),
        text_ids=packed.text_ids.clone() if include_text else torch.empty(0, dtype=torch.long),
        text_indexes=torch.arange(nt),
        position_ids=packed.position_ids[:, global_indexes.to(packed.position_ids.device)],
        vision=vision,
        action=action,
        action_state_mask=packed.action_state_mask[action_rows] if action is not None else None,
        vision_condition_type_mask=(
            packed.vision_condition_type_mask.reshape(-1, layout.vision_tokens)[vision_rows].reshape(-1)
            if vision is not None
            else None
        ),
        vision_item_split_lens=[[selected.numel()]] if vision is not None else [],
        text_caption_lens=packed.text_caption_lens if include_text else [],
        text_caption_view_ids=packed.text_caption_view_ids if include_text else [],
    )
