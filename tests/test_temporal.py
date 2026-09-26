import numpy as np

from cosmos3_joint_video_hand_pose.src.temporal import (
    cosmos_wam_token_count,
    retained_fps,
    retention_candidates,
    select_frame_indices,
)


def test_tail_trim_preserves_dense_order_and_4n_plus_1():
    indexes = select_frame_indices(12, text_tokens=20)
    np.testing.assert_array_equal(indexes, np.arange(9))
    assert (len(indexes) - 1) % 4 == 0


def test_oversized_sampling_preserves_first_frame_and_cap():
    indexes = select_frame_indices(2401, text_tokens=64, cap=110_000)
    assert indexes[0] == 0
    assert np.all(np.diff(indexes) > 0)
    assert (len(indexes) - 1) % 4 == 0
    assert cosmos_wam_token_count(64, len(indexes)) <= 110_000


def test_no_future_is_skipped():
    assert select_frame_indices(1, text_tokens=10) is None


def test_sampler_keeps_native_packer_cap_strict():
    text_tokens = 15
    # Use the exact token count of a legal clip as the cap. Cosmos rejects
    # current + sample >= cap, so the selected result must be smaller.
    exact_cap = cosmos_wam_token_count(text_tokens, 101)
    indexes = select_frame_indices(101, text_tokens=text_tokens, cap=exact_cap)
    assert indexes is not None
    assert cosmos_wam_token_count(text_tokens, len(indexes)) < exact_cap


def test_oversized_sampler_uses_discrete_retention_ladder():
    text_tokens = 20
    source_frames = 401
    candidates = dict(retention_candidates(source_frames))
    assert candidates == {1.0: 401, 0.8: 321, 0.7: 281, 0.6: 241, 0.5: 201}

    for ratio in (0.8, 0.7, 0.6, 0.5):
        cap = cosmos_wam_token_count(text_tokens, candidates[ratio]) + 1
        indexes = select_frame_indices(source_frames, text_tokens, cap=cap)
        assert indexes is not None
        assert len(indexes) == candidates[ratio]
        assert indexes[0] == 0
        assert indexes[-1] == source_frames - 1

    assert select_frame_indices(
        source_frames,
        text_tokens,
        cap=cosmos_wam_token_count(text_tokens, candidates[0.5]),
    ) is None


def test_retained_fps_preserves_sampled_clip_timespan():
    assert retained_fps(30.0, source_frames=401, target_frames=321) == 24.0
    assert retained_fps(30.0, source_frames=401, target_frames=201) == 15.0
