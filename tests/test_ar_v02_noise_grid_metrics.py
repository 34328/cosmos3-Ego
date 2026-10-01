import numpy as np
import pytest
from types import SimpleNamespace

from cosmos3_joint_video_hand_pose.src.ar_v02_noise_grid_metrics import (
    endpoint_flow, pool_shift, source_shift_mse, summarize,
)


def test_dense_source_frame_shifts_equal_support_and_sign():
    rng = np.random.default_rng(42)
    gt = rng.integers(0, 255, (24, 4, 5, 3), dtype=np.uint8)
    source = np.arange(2, 24, 2)
    pred = gt[np.minimum(source+2, len(gt)-1)]
    report = source_shift_mse(pred, gt, source, np.ones(len(source), dtype=int))
    assert report["all"]["best_gt_shift_frames"] == 2
    assert report["all"]["best_mse"] == 0
    assert len({r["values"] for r in report["all"]["curve"]}) == 1
    assert [r["gt_shift_frames"] for r in report["all"]["curve"]] == [-4, -2, 0, 2, 4]
    assert report["all"]["curve"][0]["values"] == 8*4*5*3


def test_static_video_ties_choose_zero():
    gt = np.zeros((20, 4, 4, 3), dtype=np.uint8)
    report = source_shift_mse(gt[::2], gt, np.arange(0, 20, 2), np.ones(10))
    assert report["all"]["best_gt_shift_frames"] == 0
    assert report["all"]["flat_curve"]


def test_shift_pool_uses_sums_and_counts():
    rows = [dict(curve=[dict(gt_shift_frames=s, squared_error_sum=x, values=n)
                       for s in [-4,-2,0,2,4]]) for x,n in [(10,1),(20,10)]]
    assert pool_shift(rows)["zero_mse"] == pytest.approx(30/11)


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
    report = endpoint_flow(layout,dict(valid_image_rect=[0,0,4,4]),
                           dict(generated_offsets=[0,3],generated_rgb=block,gt_rgb=gt))
    assert calls == [(7,20),(0,4)]
    assert report["all"]["frame_pairs"] == 1
    assert report["all"]["magnitude_ratio"] == pytest.approx(1)


def test_reject_mixed_groups():
    s=dict(sample_id="a",source_offset=0,checkpoint="x",history="gt",history_video_sigma=0)
    with pytest.raises(ValueError): summarize([s,dict(s,sample_id="b",history="generated")])
    with pytest.raises(ValueError): summarize([s,s])


def test_nonfinite_input_rejected():
    gt=np.zeros((20,4,4,3));pred=gt[::2].copy();pred[0,0,0,0]=np.nan
    with pytest.raises(ValueError):
        source_shift_mse(pred,gt,np.arange(0,20,2),np.ones(10))
