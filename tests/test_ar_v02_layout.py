"""Source-frame ownership and leakage contracts for joint_chunk_cond_v1."""

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_v02_layout import (
    ACTION,
    CONDITION_VIDEO,
    STATE,
    VIDEO,
    JointChunkLayout,
    joint_mask_mod,
)


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("future_groups", [8, 16, 32])
def test_action_and_condition_source_coverage(c, future_groups):
    layout = JointChunkLayout(future_groups + 1, 3, c)
    roles, chunks, source = layout.metadata()
    ar, ac, sources = layout.action_metadata()
    vr, vc, vs = layout.video_metadata()
    expected_boundaries = torch.arange(0, future_groups, c) * 8
    assert roles.numel() == layout.num_tokens
    torch.testing.assert_close(sources[ar == ACTION], torch.arange(1, 8 * future_groups + 1))
    torch.testing.assert_close(sources[ar == STATE], expected_boundaries)
    torch.testing.assert_close(vs[vr == CONDITION_VIDEO], expected_boundaries)
    torch.testing.assert_close(vs[vr == VIDEO], torch.arange(1, future_groups + 1) * 8)
    assert layout.num_video_frames == future_groups + len(expected_boundaries)
    assert layout.num_action_rows == 8 * future_groups + len(expected_boundaries)
    assert chunks.min() == 1
    future = torch.zeros(future_groups * 8, 64)
    future[:, 0] = torch.arange(1, future_groups * 8 + 1)
    boundary = torch.zeros(future_groups, 64)
    boundary[:, 9] = torch.arange(future_groups) * 8
    values, visible = layout.assemble_action(future, boundary, torch.ones(len(future), 2, dtype=torch.bool))
    torch.testing.assert_close(values[ar == ACTION], future)
    torch.testing.assert_close(values[ar == STATE, 9], expected_boundaries.float())
    assert not visible[ar == STATE].any() and visible[ar == ACTION].all()
    states, restored, source_rows = layout.unpack_action(values)
    torch.testing.assert_close(restored, future)
    torch.testing.assert_close(source_rows, torch.arange(1, len(future) + 1))
    for chunk in range(1, len(expected_boundaries) + 1):
        indexes = layout.condition_prefill_indexes(chunk)
        assert len(indexes) == 4  # three U patches and one S
        assert (chunks[indexes] == chunk).all()
        assert set(roles[indexes].tolist()) == {CONDITION_VIDEO, STATE}
        assert (source[indexes] == expected_boundaries[chunk - 1]).all()


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("noisy", [False, True])
def test_visibility_at_first_eviction_and_partial_tail(c, noisy):
    layout = JointChunkLayout(1 + 16 * c + 1, 2, c)
    roles, chunks, _ = layout.metadata()
    n = len(roles)
    predicate = joint_mask_mod(roles, chunks, text_pad_len=3, text_len=2, noisy=noisy)
    allowed = predicate(0, 0, torch.arange(n)[:, None], torch.arange(3 + n * (2 if noisy else 1))[None])
    current = allowed[:, 3 : 3 + n]
    clean = allowed[:, 3 + n :] if noisy else current
    cond = (roles == STATE) | (roles == CONDITION_VIDEO)
    index = lambda role, chunk: int(torch.where((roles == role) & (chunks == chunk))[0][0])
    for chunk in (1, 15, 16, 17):
        v, a = index(VIDEO, chunk), index(ACTION, chunk)
        assert current[v, a] and current[a, v]
        assert clean[v, cond & (chunks == chunk)].all()
        assert clean[a, cond & (chunks == chunk)].all()
        for old_chunk in range(1, chunk):
            assert bool(clean[v, chunks == old_chunk].all()) == (chunk - old_chunk <= 15)
        if noisy:
            assert not clean[v, (~cond) & (chunks == chunk)].any()
            assert not current[v, cond].any()
        assert not clean[v, chunks > chunk].any()
    assert clean[index(VIDEO, 16), chunks == 1].all()
    assert not clean[index(VIDEO, 17), chunks == 1].any()
    for row in torch.where(cond)[0]:
        torch.testing.assert_close(clean[row], cond & (chunks == chunks[row]))
        if noisy:
            assert not current[row].any()
    assert allowed[:, :2].all() and not allowed[:, 2].any()


@pytest.mark.parametrize("noisy", [False, True])
def test_two_samples_and_padding_are_isolated(noisy):
    layouts = [JointChunkLayout(4, 1, 2), JointChunkLayout(6, 2, 2)]
    roles = torch.cat([x.metadata()[0] for x in layouts] + [torch.tensor([-1, -1])])
    chunks = torch.cat([x.metadata()[1] for x in layouts] + [torch.tensor([-1, -1])])
    samples = torch.cat([torch.full((x.num_tokens,), i) for i, x in enumerate(layouts)] + [torch.tensor([-1, -1])])
    text_ids = torch.tensor([0, 0, 1, 1, 1, -1])
    n = len(roles)
    mask = joint_mask_mod(
        roles, chunks, text_pad_len=6, text_len=5, noisy=noisy, sample_ids=samples, text_sample_ids=text_ids
    )
    allowed = mask(0, 0, torch.arange(n)[:, None], torch.arange(6 + n * (2 if noisy else 1))[None])
    for sample in (0, 1):
        q = samples == sample
        torch.testing.assert_close(allowed[q, :6], (text_ids == sample)[None].expand(int(q.sum()), -1))
        for begin in ([6, 6 + n] if noisy else [6]):
            assert not allowed[q, begin : begin + n][:, samples != sample].any()
    assert not allowed[roles >= 0, :6][:, text_ids < 0].any()


def test_no_future_paths_through_multiple_clean_layers():
    roles, chunks, _ = JointChunkLayout(6, 1, 2).metadata()
    n = len(roles)
    clean = joint_mask_mod(roles, chunks, text_pad_len=0, text_len=0, noisy=False)(
        0, 0, torch.arange(n)[:, None], torch.arange(n)[None]
    )
    reach = clean.float()
    for _ in range(5):
        reach = ((reach @ clean.float()) > 0).float()
    assert not reach[chunks == 1][:, chunks > 1].any()
    conditions = (roles == STATE) | (roles == CONDITION_VIDEO)
    for row in torch.where(conditions)[0]:
        torch.testing.assert_close(reach[row].bool(), conditions & (chunks == chunks[row]))


@pytest.mark.parametrize("bad", ["padding", "camera", "nan"])
def test_assemble_action_rejects_invalid_input(bad):
    layout = JointChunkLayout(3, 2, 2)
    future, states = torch.zeros(16, 64), torch.zeros(2, 64)
    if bad == "padding":
        future[0, 63] = 1
    elif bad == "camera":
        states[0, 0] = 1
    else:
        states[0, 10] = float("nan")
    with pytest.raises(ValueError):
        layout.assemble_action(future, states, torch.ones(16, 2, dtype=torch.bool))


def test_window_and_action_rate_are_fixed():
    with pytest.raises(ValueError, match="15"):
        JointChunkLayout(5, 2, 4, history_chunks=30)
    with pytest.raises(ValueError, match="K=8"):
        JointChunkLayout(5, 2, 4, action_tokens=1)
