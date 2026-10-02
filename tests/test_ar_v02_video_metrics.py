"""Temporal, historical flow, and whole-vector endpoint video metrics."""

from types import SimpleNamespace

import numpy as np
import pytest

from cosmos3_joint_video_hand_pose.src import (
    ar_v02_video_diagnostics as diagnostics,
    ar_v02_noise_grid_metrics as noise_grid,
    ar_v02_endpoint_video_metrics as endpoint_metrics,
)


def test_positive_lag_means_prediction_is_late_and_support_is_fixed():
    rng=np.random.default_rng(7)
    gt=rng.integers(0,256,(31,8,8,3),dtype=np.uint8)
    pred=np.roll(gt,3,axis=0)
    result=diagnostics.best_time_offset(pred,gt,sample_seconds=2/30,max_lag=5)
    assert result["lag_samples"]==3
    assert result["lag_seconds"]==pytest.approx(.2)
    assert result["mse"]==0 and result["zero_lag_mse"]>0
    assert result["common_frames"]==21
    assert not result["search_boundary_hit"]
    assert len(result["curve"])==11
    early=diagnostics.best_time_offset(np.roll(gt,-2,axis=0),gt,sample_seconds=2/30,max_lag=5)
    assert early["lag_samples"]==-2


def test_static_video_is_ambiguous_not_evidence_of_correct_speed():
    rgb=np.zeros((20,8,8,3),dtype=np.uint8)
    result=diagnostics.best_time_offset(rgb,rgb,sample_seconds=2/30,max_lag=3)
    assert result["flat_curve"] and result["tied_minima"]==7
    assert result["lag_samples"]==0


def test_invalid_lag_inputs_fail():
    rgb=np.zeros((4,8,8,3),dtype=np.uint8)
    with pytest.raises(ValueError,match="support"):
        diagnostics.best_time_offset(rgb,rgb,sample_seconds=.1,max_lag=2)
    with pytest.raises(ValueError):
        diagnostics.best_time_offset(rgb,rgb,sample_seconds=float('nan'),max_lag=1)


def test_different_modes_and_duplicate_windows_are_not_pooled():
    rows=[dict(history='gt',checkpoint='one',sample_id='a'),
          dict(history='generated',checkpoint='one',sample_id='b')]
    with pytest.raises(ValueError,match='one checkpoint'):
        diagnostics.summarize(rows)
    rows[1].update(history='gt',sample_id='a')
    with pytest.raises(ValueError,match='duplicate'):
        diagnostics.summarize(rows)


def test_flow_direction_amplitude_and_stationary_cases():
    gt=np.zeros((3,4,2));gt[...,0]=2
    same=diagnostics.flow_sums(gt*.5,gt)
    result=diagnostics.summarize_flow([same])
    assert result["magnitude_ratio"]==.5
    assert result["direction_cosine"]==1 and result["direction_coverage"]==1
    opposite=diagnostics.summarize_flow([diagnostics.flow_sums(-gt,gt)])
    assert opposite["direction_cosine"]==-1
    frozen=diagnostics.summarize_flow([diagnostics.flow_sums(gt*0,gt)])
    assert frozen["magnitude_ratio"]==0
    assert frozen["direction_cosine"] is None and frozen["direction_coverage"]==0
    stationary=diagnostics.summarize_flow([diagnostics.flow_sums(gt,gt*0)])
    assert stationary["magnitude_ratio"] is None
    assert stationary["direction_cosine"] is None
    assert stationary["all_pred_magnitude_sum"]>0


def test_pooling_uses_sums_not_mean_of_ratios():
    a=np.zeros((1,1,2));a[...,0]=1
    b=np.zeros((1,3,2));b[...,0]=1
    result=diagnostics.summarize_flow([diagnostics.flow_sums(a*2,a),diagnostics.flow_sums(b*0,b)])
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
    pred,target,src,chunks=diagnostics.future_rgb(SimpleNamespace(boundaries=bounds),dict(valid_image_rect=[0,0,4,4]),arrays)
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
    result=diagnostics.sequence_flow(rgb,rgb,[1,1,17,17])
    assert result["all"]["frame_pairs"]==2
    assert result["chunk17plus"]["frame_pairs"]==1
    assert result["all"]["magnitude_ratio"]==pytest.approx(1)
    assert result["all"]["direction_cosine"]==pytest.approx(1)
    assert result["all"]["direction_coverage"]==1


def test_dense_source_frame_shifts_equal_support_and_sign():
    rng = np.random.default_rng(42)
    gt = rng.integers(0, 255, (24, 4, 5, 3), dtype=np.uint8)
    source = np.arange(2, 24, 2)
    pred = gt[np.minimum(source+2, len(gt)-1)]
    report = noise_grid.source_shift_mse(pred, gt, source, np.ones(len(source), dtype=int))
    assert report["all"]["best_gt_shift_frames"] == 2
    assert report["all"]["best_mse"] == 0
    assert len({r["values"] for r in report["all"]["curve"]}) == 1
    assert [r["gt_shift_frames"] for r in report["all"]["curve"]] == [-4, -2, 0, 2, 4]
    assert report["all"]["curve"][0]["values"] == 8*4*5*3


def test_static_video_ties_choose_zero():
    gt = np.zeros((20, 4, 4, 3), dtype=np.uint8)
    report = noise_grid.source_shift_mse(gt[::2], gt, np.arange(0, 20, 2), np.ones(10))
    assert report["all"]["best_gt_shift_frames"] == 0
    assert report["all"]["flat_curve"]


def test_shift_pool_uses_sums_and_counts():
    rows = [dict(curve=[dict(gt_shift_frames=s, squared_error_sum=x, values=n)
                       for s in [-4,-2,0,2,4]]) for x,n in [(10,1),(20,10)]]
    assert noise_grid.pool_shift(rows)["zero_mse"] == pytest.approx(30/11)


def test_endpoint_flow_uses_condition_to_last_future(monkeypatch):
    import cosmos3_joint_video_hand_pose.src.ar_v02_noise_grid_metrics as module
    calls = []
    def fake(a,b,*args,**kwargs):
        calls.append((int(a[0,0]),int(b[0,0])))
        return np.ones((*a.shape,2),dtype=np.float32)
    monkeypatch.setattr(module.cv2,"calcOpticalFlowFarneback",fake)
    block = np.stack([np.full((4,4,3), v,np.uint8) for v in [7,10,20]])
    gt = np.stack([np.full((4,4,3), v,np.uint8) for v in range(5)])
    layout = SimpleNamespace(boundaries=[SimpleNamespace(source_start=0,source_stop=4,chunk_id=1)])
    report = noise_grid.endpoint_flow(layout,dict(valid_image_rect=[0,0,4,4]),
                           dict(generated_offsets=[0,3],generated_rgb=block,gt_rgb=gt))
    assert calls == [(7,20),(0,4)]
    assert report["all"]["frame_pairs"] == 1
    assert report["all"]["magnitude_ratio"] == pytest.approx(1)


def test_reject_mixed_groups():
    s=dict(sample_id="a",source_offset=0,checkpoint="x",history="gt",history_video_sigma=0)
    with pytest.raises(ValueError): noise_grid.summarize([s,dict(s,sample_id="b",history="generated")])
    with pytest.raises(ValueError): noise_grid.summarize([s,s])


def test_nonfinite_input_rejected():
    gt=np.zeros((20,4,4,3));pred=gt[::2].copy();pred[0,0,0,0]=np.nan
    with pytest.raises(ValueError):
        noise_grid.source_shift_mse(pred,gt,np.arange(0,20,2),np.ones(10))


def test_flat_cosine_and_mean_magnitude():
    g = np.array([[[1., 0.], [0., 2.]]])
    for scale, ratio, cosine in [(1, 1, 1), (2, 2, 1), (-1, 1, -1)]:
        r = endpoint_metrics.pool_flow([endpoint_metrics.flow_sums(scale*g, g)])
        assert r["magnitude_ratio"] == pytest.approx(ratio)
        assert r["direction_cosine"] == pytest.approx(cosine)


def test_pooling_is_concatenated_vector_cosine_not_mean_pixel_cosine():
    p = np.array([[[10., 0.], [-1., 0.]]])
    g = np.array([[[10., 0.], [1., 0.]]])
    full = endpoint_metrics.pool_flow([endpoint_metrics.flow_sums(p, g)])
    split = endpoint_metrics.pool_flow([endpoint_metrics.flow_sums(p[:, :1], g[:, :1]),
                       endpoint_metrics.flow_sums(p[:, 1:], g[:, 1:])])
    assert full["direction_cosine"] == pytest.approx(99/101)
    assert split["direction_cosine"] == full["direction_cosine"]
    assert full["direction_cosine"] != 0  # Mean of pixel cosines would be 0.


def test_stationary_flow_is_not_fabricated():
    zero = np.zeros((3, 4, 2))
    assert endpoint_metrics.pool_flow([endpoint_metrics.flow_sums(zero, zero)])["direction_cosine"] is None
    assert endpoint_metrics.pool_flow([endpoint_metrics.flow_sums(np.ones_like(zero), zero)])["magnitude_ratio"] is None
    assert endpoint_metrics.pool_flow([endpoint_metrics.flow_sums(zero, np.ones_like(zero))])["magnitude_ratio"] == 0
    with pytest.raises(ValueError):
        endpoint_metrics.flow_sums(np.full_like(zero, np.nan), zero)


def test_endpoint_resize_and_source_boundaries(monkeypatch):
    import cosmos3_joint_video_hand_pose.src.ar_v02_endpoint_video_metrics as m
    seen = []
    def fake(first, last, _, **kwargs):
        assert first.shape == last.shape == (180, 320)
        seen.append((int(first[0, 0]), int(last[0, 0])))
        return np.ones((180, 320, 2), np.float32)
    monkeypatch.setattr(m.cv2, "calcOpticalFlowFarneback", fake)
    boundaries = [SimpleNamespace(chunk_id=1, source_start=0, source_stop=32),
                  SimpleNamespace(chunk_id=2, source_start=32, source_stop=64)]
    gt = np.stack([np.full((360, 640, 3), i, np.uint8) for i in range(65)])
    generated = np.stack([gt[i] for i in [0, 16, 32, 32, 48, 64]])
    r = endpoint_metrics.endpoint_flow(SimpleNamespace(boundaries=boundaries),
                      {"valid_image_rect": [0, 0, 640, 360]},
                      {"generated_offsets": [0, 3, 6], "generated_rgb": generated,
                       "gt_rgb": gt})
    assert seen == [(0, 32), (0, 32), (32, 64), (32, 64)]
    assert r["all"]["pairs"] == 2
    assert r["all"]["pixels"] == 2*320*180
    assert r["all"]["direction_cosine"] == pytest.approx(1)
