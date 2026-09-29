"""Raw annotation metrics include AE error; all fixtures are CPU algebra only."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from test_fixed_camera_runtime import clip, raw_clip, scene
from cosmos3_joint_video_hand_pose.src.ar_v02_evaluation import (
    evaluate_joint_actions,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, STATE


def evaluate(data, raw, **kwargs):
    layout, payload, actions, states, sn, fn, codecs = data
    return evaluate_joint_actions(layout,payload,actions,states,state_normalizer=sn,
        future_normalizer=fn,hand_codecs=codecs,raw_gt=raw,**kwargs)


def test_raw_primary_includes_codec_error_when_prediction_equals_gt_latent():
    data = clip()
    raw = raw_clip(data[0])
    # The synthetic codec retains only five points and repeats them. Change one
    # discarded annotation: E(q) stays identical but reconstruction has error.
    raw["keypoints"][:, :, 6, 0] += .021
    from cosmos3_joint_video_hand_pose.src.action_fixed_camera import encode_chunk_physical
    h,w,_ = scene(len(raw["keypoints"]))
    for b in data[0].boundaries:
        sl=slice(b.source_start,b.source_stop+1)
        encoded=encode_chunk_physical(h[sl].float(),w[sl].float(),raw["keypoints"][sl].float(),
                                      range(b.source_start,b.source_stop+1),data[-1])
        torch.testing.assert_close(encoded.actions, data[2][b.source_start:b.source_stop,:57])
    result = evaluate(data,raw)
    assert result["hand_metric_target"] == "raw_gt_keypoints"
    assert result["hand_metrics_include_gt_codec_reconstruction_error"]
    for record in result["chunks"]:
        for side in ("right", "left"):
            assert record[f"local_{side}_mpjpe_mm"] == pytest.approx(1., abs=.003)
            assert record[f"decoded_gt_aux_{side}_mpjpe_mm"] < .003
            assert record[f"local_{side}_wrist_local_shape_mpjpe_mm"] > 1.


@pytest.mark.parametrize("history", ["gt", "pred_history", "generated"])
def test_raw_source_alignment_across_chunks_and_nonzero_window(history):
    data = clip(3)
    raw = raw_clip(data[0],137)
    result = evaluate(data,raw,source_offset=137,history=history)
    assert result["episode_source_indexes"] == list(range(138,194))
    for record in result["chunks"]:
        assert record["local_right_mpjpe_mm"] < .003
    shifted = deepcopy(raw)
    shifted["source_indexes"] += 1
    with pytest.raises(ValueError,match="source frames"):
        evaluate(data,shifted,source_offset=137,history=history)


def test_generated_ignores_later_gt_payload_states_and_retains_drift():
    data = clip()
    layout,payload,actions,states,sn,fn,codecs = data
    roles,chunks,_ = layout.action_metadata()
    payload[torch.where((roles == ACTION) & (chunks == 1))[0][0],9] += .1
    result = evaluate(data,raw_clip(layout),history="generated")
    assert result["chunks"][1]["local_right_wrist_mean_mm"] > 99
    before = deepcopy(result)
    payload[(roles == STATE) & (chunks > 1),9:12] += 7
    after = evaluate(data,raw_clip(layout),history="generated")
    assert after == before
    assert after["boundary_state_reset_to_gt"] is False


def test_missing_raw_is_not_silently_primary():
    with pytest.raises(ValueError,match="raw GT"):
        evaluate(clip(),None)


def test_legacy_missing_raw_remains_explicit_diagnostic():
    from test_ar_v02_inference import fixture
    sampler,future,_ = fixture()
    class Codec:
        def decode(self,z):
            return z.new_zeros(len(z),20,3)
    result=evaluate_joint_actions(sampler.layout,sampler.gt_action,future,sampler.gt_states,
        state_normalizer=sampler.state_normalizer,future_normalizer=sampler.future_normalizer,
        hand_codecs=(Codec(),Codec()))
    assert result["hand_metric_status"] == "legacy_decoded_gt_diagnostic_only"
    assert result["hand_metric_target"] == "decoded_gt_action_latents"
    assert not result["hand_metrics_include_gt_codec_reconstruction_error"]


@pytest.mark.parametrize("field,value", [("units","millimetres"),("hand_order",["left","right"]),("coordinate_frame","camera")])
def test_raw_coordinate_contract_rejected(field,value):
    data = clip()
    raw = raw_clip(data[0])
    raw[field] = value
    with pytest.raises(ValueError,match="world coordinates"):
        evaluate(data,raw)


def test_raw_archive_payload_roundtrip_and_incomplete_rejected(tmp_path, monkeypatch):
    from cosmos3_joint_video_hand_pose.src.ar_v02_eval import save_rollout,load_rollout,_raw_gt_payload
    data=clip()
    layout,payload,actions,states,sn,fn,codecs=data
    raw=raw_clip(layout,113)
    meta=dict(sample_id="test",episode_id="fixture",history="gt",seed=42,source_offset=113,
              source_fps=30.,speed_factor=.5,action_representation=sn.representation,
              hand_codecs={s:dict(path=f"/{s}",sha256=c.checkpoint_sha256) for s,c in zip(("right","left"),codecs)},
              state_normalizer=dict(path="/state",sha256="a"*64),future_normalizer=dict(path="/future",sha256="b"*64))
    path=save_rollout(tmp_path/"raw.npz",layout=layout,predicted_action=payload,gt_future=actions,
                      boundary_states=states,metadata=meta,raw_gt=raw)
    _,loaded,arrays=load_rollout(path)
    restored=_raw_gt_payload(loaded,arrays)
    np.testing.assert_array_equal(restored["keypoints"],raw["keypoints"])
    result=evaluate(data,restored,source_offset=113)
    assert result["hand_metric_target"] == "raw_gt_keypoints"
    kwargs=dict(layout=layout,predicted_action=payload,gt_future=actions,boundary_states=states,metadata=meta)
    with pytest.raises(ValueError,match="raw GT required"):
        save_rollout(tmp_path/"missing.npz",**kwargs)
    diagnostic=save_rollout(tmp_path/"diagnostic.npz",raw_gt_disabled_diagnostic=True,**kwargs)
    _,diagnostic_meta,_=load_rollout(diagnostic)
    assert diagnostic_meta["raw_gt_disabled_diagnostic"] is True
    from types import SimpleNamespace
    from cosmos3_joint_video_hand_pose.src import ar_v02_eval as cli
    monkeypatch.setattr(cli,"_normalizers",lambda *_: (sn,fn))
    monkeypatch.setattr(cli,"_codecs",lambda *_: codecs)
    with pytest.raises(ValueError,match="raw GT keypoints required"):
        cli.evaluate_archive(diagnostic,SimpleNamespace(rigid_only=False))
    rigid=cli.evaluate_archive(diagnostic,SimpleNamespace(rigid_only=True))["metrics"]
    assert not rigid["hand_metrics_available"]
    assert rigid["hand_metric_status"] == "not_computed"
    assert all("local_right_mpjpe_mm" not in row for row in rigid["chunks"])
    with pytest.raises(ValueError,match="cannot accompany raw GT"):
        save_rollout(tmp_path/"conflict.npz",raw_gt=raw,raw_gt_disabled_diagnostic=True,**kwargs)
    del arrays["raw_gt_camera_poses"]
    with pytest.raises(ValueError,match="incomplete raw GT"):
        _raw_gt_payload(loaded,arrays)
