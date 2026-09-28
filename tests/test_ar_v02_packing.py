"""Real builder contracts for mixed lengths, sample order, and physical time."""

import pytest
import torch

from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.teacher_forcing import make_teacher_forcing_clean_pack
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import (
    ACTION,
    CONDITION_VIDEO,
    STATE,
    VIDEO,
    LAYOUT_VERSION,
    JointChunkLayout,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence, select_joint_pack

SPECIAL = {"eos_token_id": 5, "start_of_generation": 6}


def make_pack(layouts, texts=None, tags=None, fps=None, *, timesteps=None):
    texts = texts or [[3, 4 + i] for i in range(len(layouts))]
    tags = tags or list(range(len(layouts)))
    fps = fps or [7.5] * len(layouts)
    data = GenerationDataClean(
        batch_size=len(layouts),
        is_image_batch=False,
        x0_tokens_vision=[torch.full((1, 4, x.num_video_frames, 4, 6), float(tag)) for x, tag in zip(layouts, tags)],
        x0_tokens_action=[torch.full((x.num_action_rows, 64), float(tag)) for x, tag in zip(layouts, tags)],
        fps_vision=torch.tensor(fps),
        fps_action=torch.tensor(fps) * 2,
        raw_action_dim=[torch.tensor(57) for _ in layouts],
        action_domain_id=[torch.tensor([tag]) for tag in tags],
    )
    if timesteps is None:
        timesteps = [torch.arange(x.num_video_frames) * 10 + tag * 100 for x, tag in zip(layouts, tags)]
    return pack_joint_sequence(
        layout=layouts,
        gen_data_clean=data,
        text_ids=texts,
        special_tokens=SPECIAL,
        timesteps=timesteps,
        condition_frames=[() for _ in layouts],
        latent_patch_size=2,
        modality_margin=15000,
    )


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_pack_preserves_conditions_targets_and_absolute_time(c):
    layout = JointChunkLayout(10, 6, c)
    packed = make_pack([layout])
    vr, _, vs = layout.video_metadata()
    ar, _, source = layout.action_metadata()
    assert packed.joint_layout_version == LAYOUT_VERSION
    assert packed.vision.token_shapes == [(layout.num_video_frames, 2, 3)]
    assert packed.action.mse_loss_indexes.numel() == 72
    assert packed.vision.mse_loss_indexes.numel() == 54
    torch.testing.assert_close(packed.action_state_mask, ar == STATE)
    torch.testing.assert_close(packed.vision_condition_type_mask, (vr == CONDITION_VIDEO).repeat_interleave(6))
    torch.testing.assert_close(packed.action.noisy_frame_indexes[0], torch.where(ar == ACTION)[0])
    torch.testing.assert_close(packed.vision.noisy_frame_indexes[0], torch.where(vr == VIDEO)[0])
    assert packed.position_ids.dtype == torch.float32
    apos = packed.position_ids[0, packed.action.sequence_indexes]
    vpos = packed.position_ids[0, packed.vision.sequence_indexes].reshape(-1, 6)[:, 0]
    torch.testing.assert_close(apos, 15004 + source.float() * 0.4, atol=1e-3, rtol=0)
    torch.testing.assert_close(vpos, 15004 + vs.float() * 0.4, atol=1e-3, rtol=0)
    torch.testing.assert_close(apos[ar == ACTION][7::8], vpos[vr == VIDEO], atol=1e-3, rtol=0)
    assert (apos[ar == ACTION].diff() > 0).all()
    for chunk in range(1, len(layout.boundaries) + 1):
        part = select_joint_pack(packed, layout, layout.condition_prefill_indexes(chunk), include_text=False)
        assert part.text_ids.numel() == 0
        assert part.vision.token_shapes == [(1, 2, 3)]
        assert part.action.token_shapes == [(1,)]
        assert part.vision.mse_loss_indexes.numel() == part.action.mse_loss_indexes.numel() == 0
        expected_time = 15004 + (chunk - 1) * c * 8 * 0.4
        torch.testing.assert_close(part.position_ids[0], torch.full((7,), expected_time), atol=1e-3, rtol=0)
    clean = make_teacher_forcing_clean_pack(packed)
    assert clean.action.mse_loss_indexes.numel() == clean.vision.mse_loss_indexes.numel() == 0
    torch.testing.assert_close(clean.action_state_mask, packed.action_state_mask)
    torch.testing.assert_close(clean.vision_condition_type_mask, packed.vision_condition_type_mask)
    assert packed.action.mse_loss_indexes.numel() == 72


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_mixed_clip_pack_equals_individual_and_reordered_packs(c):
    layouts = [JointChunkLayout(n, 6, c) for n in (9, 17, 33)]
    texts = [[3], [4, 3, 4], [3, 4]]
    tags, fps = [1, 2, 3], [7.5, 15.0, 7.5]
    packed = make_pack(layouts, texts, tags, fps)
    assert packed.attn_modes == ["causal", "full"] * 3
    assert packed.num_vision_items_per_sample in (None, [1, 1, 1])
    start = 0
    for i, layout in enumerate(layouts):
        single = make_pack([layout], [texts[i]], [tags[i]], [fps[i]])
        end = start + packed.sample_lens[i]
        torch.testing.assert_close(packed.position_ids[:, start:end], single.position_ids)
        for attr in ("vision", "action"):
            batched, one = getattr(packed, attr), getattr(single, attr)
            torch.testing.assert_close(batched.tokens[i], one.tokens[0])
            torch.testing.assert_close(batched.condition_mask[i], one.condition_mask[0])
            select = (batched.mse_loss_indexes >= start) & (batched.mse_loss_indexes < end)
            torch.testing.assert_close(batched.mse_loss_indexes[select] - start, one.mse_loss_indexes)
            torch.testing.assert_close(batched.timesteps[select], one.timesteps)
        start = end
    assert start == packed.sequence_length
    order = [2, 0, 1]
    reordered = make_pack(
        [layouts[i] for i in order], [texts[i] for i in order], [tags[i] for i in order], [fps[i] for i in order]
    )
    starts = [0] + list(torch.tensor(packed.sample_lens).cumsum(0).tolist())
    offset = 0
    for i, original in enumerate(order):
        n = reordered.sample_lens[i]
        torch.testing.assert_close(
            reordered.position_ids[:, offset : offset + n],
            packed.position_ids[:, starts[original] : starts[original + 1]],
        )
        torch.testing.assert_close(reordered.vision.tokens[i], packed.vision.tokens[original])
        offset += n


def test_text_only_prefill_and_partial_patch_rejection():
    layout = JointChunkLayout(6, 6, 4)
    packed = make_pack([layout])
    text = select_joint_pack(packed, layout, [], include_text=True)
    assert text.vision is None and text.action is None
    torch.testing.assert_close(text.position_ids, packed.position_ids[:, packed.text_indexes])
    with pytest.raises(ValueError, match="spatial patch"):
        select_joint_pack(packed, layout, [0], include_text=False)


def test_official_rope_keeps_fractional_time_with_bfloat16_hidden_states():
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRotaryEmbedding

    # Exercise the actual official forward without loading a backbone/config/checkpoint.
    rotary = object.__new__(Qwen3VLTextRotaryEmbedding)
    torch.nn.Module.__init__(rotary)
    rotary.rope_type = "default"
    rotary.mrope_section = [24, 20, 20]
    rotary.attention_scaling = 1.0
    rotary.register_buffer("inv_freq", 10000.0 ** (-torch.arange(0, 128, 2).float() / 128))
    layout = JointChunkLayout(33, 6, 3)
    pack = make_pack([layout])
    action = pack.position_ids[:, pack.action.sequence_indexes][:, ~pack.action_state_mask]
    pos = action[:, None, :]
    cos32, sin32 = rotary(torch.empty(1, 1, 128), pos)
    cos16, sin16 = rotary(torch.empty(1, 1, 128, dtype=torch.bfloat16), pos)
    torch.testing.assert_close(cos16, cos32.bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(sin16, sin32.bfloat16(), atol=0, rtol=0)
    # 15000 modality margin must not quantize the 0.4-spaced action timestamps.
    assert (cos16[:, 1:] != cos16[:, :-1]).any(-1).all()
    assert pos.dtype == torch.float32


def _mixed_timestep_rows(layouts):
    """Distinct sample/chunk times, with every U frame clean, as emitted by the sampler."""
    rows = []
    for sample, layout in enumerate(layouts):
        roles, chunks, _ = layout.video_metadata()
        rows.append(
            torch.where(
                roles == CONDITION_VIDEO,
                torch.zeros_like(chunks, dtype=torch.float32),
                100.0 * (sample + 1) + 10.0 * chunks.float(),
            )
        )
    return rows


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_mixed_length_padded_timestep_matrix_matches_unpadded_rows(c, order):
    # RGB T=33/65/129; C=3 also exercises partial final chunks.
    original = [JointChunkLayout(n, 6, c) for n in (9, 17, 33)]
    original_rows = _mixed_timestep_rows(original)
    layouts = [original[i] for i in order]
    rows = [original_rows[i] for i in order]
    matrix = torch.zeros(len(layouts), max(x.num_video_frames for x in layouts))
    for sample, row in enumerate(rows):
        matrix[sample, : len(row)] = row
    before = matrix.clone()
    padded = make_pack(layouts, timesteps=matrix)
    reference = make_pack(layouts, timesteps=rows)
    torch.testing.assert_close(matrix, before, atol=0, rtol=0)
    assert padded.sample_lens == reference.sample_lens
    assert padded.split_lens == reference.split_lens
    assert padded.sequence_length == reference.sequence_length
    torch.testing.assert_close(padded.position_ids, reference.position_ids, atol=0, rtol=0)
    for name in ("vision", "action"):
        actual, expected = getattr(padded, name), getattr(reference, name)
        assert actual.token_shapes == expected.token_shapes
        torch.testing.assert_close(actual.mse_loss_indexes, expected.mse_loss_indexes)
        torch.testing.assert_close(actual.timesteps, expected.timesteps, atol=0, rtol=0)
        for got, want in zip(actual.noisy_frame_indexes, expected.noisy_frame_indexes, strict=True):
            torch.testing.assert_close(got, want)
        for got, want in zip(actual.condition_mask, expected.condition_mask, strict=True):
            torch.testing.assert_close(got, want)
    # Independent expected supervision: exclude U, then repeat each future time
    # over its six video patches and eight action rows. No padded tail is a target.
    future_times = [row[layout.video_metadata()[0] == VIDEO] for layout, row in zip(layouts, rows, strict=True)]
    torch.testing.assert_close(padded.vision.timesteps, torch.cat([row.repeat_interleave(6) for row in future_times]))
    torch.testing.assert_close(padded.action.timesteps, torch.cat([row.repeat_interleave(8) for row in future_times]))


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("sample", [0, 1])
@pytest.mark.parametrize("tail_position", ["first", "last"])
def test_mixed_length_timestep_matrix_rejects_nonzero_padding(c, sample, tail_position):
    layouts = [JointChunkLayout(n, 6, c) for n in (9, 17, 33)]
    rows = _mixed_timestep_rows(layouts)
    matrix = torch.zeros(3, layouts[-1].num_video_frames)
    for i, row in enumerate(rows):
        matrix[i, : len(row)] = row
    own_length = layouts[sample].num_video_frames
    assert own_length < matrix.shape[1]
    tail_index = own_length if tail_position == "first" else matrix.shape[1] - 1
    matrix[sample, tail_index] = 0.125 if tail_position == "first" else -0.25
    with pytest.raises(ValueError, match="nonzero timestep beyond sample video length"):
        make_pack(layouts, timesteps=matrix)
