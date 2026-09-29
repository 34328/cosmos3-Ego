"""Visibility gaps are diagnostic, never equivalent to invalid tracking."""
import numpy as np
import pytest
from cosmos3_joint_video_hand_pose.src.ar_v02_prepare_data import visibility_window_report, invalid_frames


def test_retained_window_exposures_and_recovery():
    v = np.ones((9, 2), dtype=np.uint8)
    v[2:4, 0] = 0
    v[0, 1] = 0  # unseen boundary, but supervised first future delta
    result = visibility_window_report(v, [0, 2], 5)
    assert result["future_row_exposures"] == 8
    assert result["unique_future_frames"] == 6
    right = result["hands"]["right"]
    assert right["masked_future_exposures"] == 3
    assert right["windows_with_masked_future"] == 2
    assert right["supervised_recovery_exposures"] == 2
    assert right["masked_run_histogram"] == {"2": 1}
    left = result["hands"]["left"]
    assert left["masked_future_exposures"] == 0
    assert left["supervised_recovery_exposures"] == 1


def test_empty_or_disjoint_windows_do_not_count_uncovered_gaps():
    v = np.zeros((20, 2), dtype=np.uint8)
    result = visibility_window_report(v, [0, 10], 3)
    assert result["hands"]["right"]["masked_run_histogram"] == {"2": 2}
    assert visibility_window_report(v, [], 3)["future_row_exposures"] == 0


def test_recovery_supervision_does_not_repair_missing_delta():
    # An unpenalized delta at t=2 still shifts all following integrated positions,
    # including the visible t=3 whose own delta is supervised correctly.
    truth_delta = np.array([1., 1., 1., 1.])
    predicted_delta = np.array([1., 8., 1., 1.])
    visible = np.array([True, False, True, True])
    assert np.square(predicted_delta[visible] - truth_delta[visible]).sum() == 0
    assert (predicted_delta.cumsum() - truth_delta.cumsum())[2] == 7
    v = np.stack([np.r_[True, visible]] * 2, axis=1)
    assert visibility_window_report(v, [0], 5)["hands"]["right"]["supervised_recovery_exposures"] == 1


def test_fov_is_not_tracking_and_true_missing_pose_rejected():
    pose = np.zeros((5, 7), dtype=np.float32)
    pose[:, 3] = 1
    assert not np.logical_or.reduce(list(invalid_frames({"obs_head_pose": pose}, fixed_camera=True).values())).any()
    pose[2] = np.nan
    assert invalid_frames({"obs_head_pose": pose}, fixed_camera=True)["obs_head_pose:nonfinite"][2]
    with pytest.raises(ValueError, match="binary"):
        visibility_window_report(np.full((5, 2), 2), [0], 5)


def test_retired_representation_never_falls_through_to_legacy_preparation():
    from types import SimpleNamespace
    from cosmos3_joint_video_hand_pose.src.ar_v02_prepare_data import prepare
    with pytest.raises(ValueError, match="retired"):
        prepare(SimpleNamespace(representation="fixed_camera_delta_latent_v1"))


@pytest.mark.parametrize("tier,source,expected_chunks", [(129,257,8),(257,513,16),(273,545,17)])
def test_model_frames_are_not_source_frames(tier, source, expected_chunks):
    from cosmos3_joint_video_hand_pose.src.ar_dataset import select_clip_frames, clip_span, ar_latent_frames
    from cosmos3_joint_video_hand_pose.src.ar_chunk_state import chunk_boundaries
    assert clip_span(tier, 2) == source
    assert select_clip_frames(source - 1, 2, tiers=(tier,)) is None
    assert select_clip_frames(source, 2, tiers=(tier,)) == tier
    assert len(chunk_boundaries(ar_latent_frames(tier), 4)) == expected_chunks
