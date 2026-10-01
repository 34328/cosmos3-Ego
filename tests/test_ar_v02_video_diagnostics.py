from types import SimpleNamespace
import numpy as np
import pytest
from cosmos3_joint_video_hand_pose.src.ar_v02_video_diagnostics import (
    best_time_offset, flow_sums, summarize_flow, future_rgb, sequence_flow, summarize,
)


def test_positive_lag_means_prediction_is_late_and_support_is_fixed():
    rng=np.random.default_rng(7)
    gt=rng.integers(0,256,(31,8,8,3),dtype=np.uint8)
    pred=np.roll(gt,3,axis=0)
    result=best_time_offset(pred,gt,sample_seconds=2/30,max_lag=5)
    assert result["lag_samples"]==3
    assert result["lag_seconds"]==pytest.approx(.2)
    assert result["mse"]==0 and result["zero_lag_mse"]>0
    assert result["common_frames"]==21
    assert not result["search_boundary_hit"]
    assert len(result["curve"])==11
    early=best_time_offset(np.roll(gt,-2,axis=0),gt,sample_seconds=2/30,max_lag=5)
    assert early["lag_samples"]==-2


def test_static_video_is_ambiguous_not_evidence_of_correct_speed():
    rgb=np.zeros((20,8,8,3),dtype=np.uint8)
    result=best_time_offset(rgb,rgb,sample_seconds=2/30,max_lag=3)
    assert result["flat_curve"] and result["tied_minima"]==7
    assert result["lag_samples"]==0


def test_invalid_lag_inputs_fail():
    rgb=np.zeros((4,8,8,3),dtype=np.uint8)
    with pytest.raises(ValueError,match="support"):
        best_time_offset(rgb,rgb,sample_seconds=.1,max_lag=2)
    with pytest.raises(ValueError):
        best_time_offset(rgb,rgb,sample_seconds=float('nan'),max_lag=1)


def test_different_modes_and_duplicate_windows_are_not_pooled():
    rows=[dict(history='gt',checkpoint='one',sample_id='a'),
          dict(history='generated',checkpoint='one',sample_id='b')]
    with pytest.raises(ValueError,match='one checkpoint'):
        summarize(rows)
    rows[1].update(history='gt',sample_id='a')
    with pytest.raises(ValueError,match='duplicate'):
        summarize(rows)


def test_flow_direction_amplitude_and_stationary_cases():
    gt=np.zeros((3,4,2));gt[...,0]=2
    same=flow_sums(gt*.5,gt)
    result=summarize_flow([same])
    assert result["magnitude_ratio"]==.5
    assert result["direction_cosine"]==1 and result["direction_coverage"]==1
    opposite=summarize_flow([flow_sums(-gt,gt)])
    assert opposite["direction_cosine"]==-1
    frozen=summarize_flow([flow_sums(gt*0,gt)])
    assert frozen["magnitude_ratio"]==0
    assert frozen["direction_cosine"] is None and frozen["direction_coverage"]==0
    stationary=summarize_flow([flow_sums(gt,gt*0)])
    assert stationary["magnitude_ratio"] is None
    assert stationary["direction_cosine"] is None
    assert stationary["all_pred_magnitude_sum"]>0


def test_pooling_uses_sums_not_mean_of_ratios():
    a=np.zeros((1,1,2));a[...,0]=1
    b=np.zeros((1,3,2));b[...,0]=1
    result=summarize_flow([flow_sums(a*2,a),flow_sums(b*0,b)])
    assert result["magnitude_ratio"]==.5  # mean of ratios would incorrectly be 1
    assert result["direction_coverage"]==.25


def test_future_sampling_excludes_condition_duplicates_and_keeps_partial_tail():
    bounds=[SimpleNamespace(source_start=0,source_stop=32,chunk_id=1),
            SimpleNamespace(source_start=32,source_stop=40,chunk_id=2)]
    gt=np.broadcast_to(np.arange(41,dtype=np.uint8)[:,None,None,None],(41,4,4,3)).copy()
    blocks=[np.concatenate([np.full((1,4,4,3),255,np.uint8),gt[np.arange(b.source_start+2,b.source_stop+1,2)]]) for b in bounds]
    arrays=dict(gt_rgb=gt,generated_rgb=np.concatenate(blocks),
                generated_offsets=np.array([0,17,22]),
                gt_pixel_transform=np.eye(3),generated_pixel_transform=np.eye(3))
    pred,target,src,chunks=future_rgb(SimpleNamespace(boundaries=bounds),dict(valid_image_rect=[0,0,4,4]),arrays)
    np.testing.assert_array_equal(pred,target)
    np.testing.assert_array_equal(src,np.arange(2,41,2))
    np.testing.assert_array_equal(chunks,[1]*16+[2]*4)
    assert len(pred)==20 and 255 not in pred


def test_real_flow_identity_and_excluding_cross_chunk_pairs():
    import cv2
    cv2.setNumThreads(1)
    rng=np.random.default_rng(3)
    base=rng.integers(0,256,(48,48,3),dtype=np.uint8)
    rgb=np.stack([np.roll(base,k,axis=1) for k in range(4)])
    result=sequence_flow(rgb,rgb,[1,1,17,17])
    assert result["all"]["frame_pairs"]==2
    assert result["chunk17plus"]["frame_pairs"]==1
    assert result["all"]["magnitude_ratio"]==pytest.approx(1)
    assert result["all"]["direction_cosine"]==pytest.approx(1)
    assert result["all"]["direction_coverage"]==1
