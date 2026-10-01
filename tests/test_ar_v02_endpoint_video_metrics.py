"""Regression coverage for the requested whole-vector endpoint flow metrics."""

import numpy as np
import pytest
from types import SimpleNamespace

from cosmos3_joint_video_hand_pose.src.ar_v02_endpoint_video_metrics import (
    endpoint_flow, flow_sums, pool_flow,
)


def test_flat_cosine_and_mean_magnitude():
    g = np.array([[[1., 0.], [0., 2.]]])
    for scale, ratio, cosine in [(1, 1, 1), (2, 2, 1), (-1, 1, -1)]:
        r = pool_flow([flow_sums(scale*g, g)])
        assert r["magnitude_ratio"] == pytest.approx(ratio)
        assert r["direction_cosine"] == pytest.approx(cosine)


def test_pooling_is_concatenated_vector_cosine_not_mean_pixel_cosine():
    p = np.array([[[10., 0.], [-1., 0.]]])
    g = np.array([[[10., 0.], [1., 0.]]])
    full = pool_flow([flow_sums(p, g)])
    split = pool_flow([flow_sums(p[:, :1], g[:, :1]),
                       flow_sums(p[:, 1:], g[:, 1:])])
    assert full["direction_cosine"] == pytest.approx(99/101)
    assert split["direction_cosine"] == full["direction_cosine"]
    assert full["direction_cosine"] != 0  # Mean of pixel cosines would be 0.


def test_stationary_flow_is_not_fabricated():
    zero = np.zeros((3, 4, 2))
    assert pool_flow([flow_sums(zero, zero)])["direction_cosine"] is None
    assert pool_flow([flow_sums(np.ones_like(zero), zero)])["magnitude_ratio"] is None
    assert pool_flow([flow_sums(zero, np.ones_like(zero))])["magnitude_ratio"] == 0
    with pytest.raises(ValueError):
        flow_sums(np.full_like(zero, np.nan), zero)


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
    r = endpoint_flow(SimpleNamespace(boundaries=boundaries),
                      {"valid_image_rect": [0, 0, 640, 360]},
                      {"generated_offsets": [0, 3, 6], "generated_rgb": generated,
                       "gt_rgb": gt})
    assert seen == [(0, 32), (0, 32), (32, 64), (32, 64)]
    assert r["all"]["pairs"] == 2
    assert r["all"]["pixels"] == 2*320*180
    assert r["all"]["direction_cosine"] == pytest.approx(1)
