"""Fixed-camera runtime integration. Synthetic codec is algebra-only, not AE quality."""
from types import SimpleNamespace
import pytest
import torch
from test_action_fixed_camera import FakeCodec, scene
from test_ar_v02_streaming import PerfectModel, observation
from cosmos3_joint_video_hand_pose.src.action_fixed_camera import (
    FixedCameraState, encode_chunk_physical, encode_state_physical, pad_action,
)
from cosmos3_joint_video_hand_pose.src.action_representation import ActionRepresentationAdapter, FIXED_CAMERA
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_evaluation import (
    decode_joint_actions, evaluate_joint_actions, hand_keypoints, record_hand_keypoints,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_overlay import project_chunk_hands
from cosmos3_joint_video_hand_pose.src.ar_v02_streaming import StreamingJointSampler


def dependencies():
    codecs = (FakeCodec(), FakeCodec())
    hashes = ("a"*64, "b"*64)
    for codec, digest in zip(codecs, hashes):
        codec.representation = FIXED_CAMERA
        codec.checkpoint_sha256 = digest
    norms = tuple(SimpleNamespace(
        representation=FIXED_CAMERA, kind=kind, codec_sha256=hashes,
        normalize=lambda x: x, denormalize=lambda x: x,
    ) for kind in ("state", "future"))
    return *norms, codecs


def clip(tail=1):
    layout = JointChunkLayout(5+tail, 1, 4)
    h, w, k = (x.float() for x in scene(1+(4+tail)*8))
    sn, fn, codecs = dependencies()
    states = []
    for j in range(layout.num_frames-1):
        start = j*8
        enc = encode_chunk_physical(h[start:start+2], w[start:start+2], k[start:start+2], range(start,start+2), codecs)
        states.append(pad_action(encode_state_physical(enc.state)))
    actions = []
    for b in layout.boundaries:
        sl = slice(b.source_start, b.source_stop+1)
        enc = encode_chunk_physical(h[sl], w[sl], k[sl], range(b.source_start,b.source_stop+1), codecs)
        actions.append(pad_action(enc.actions))
    states, actions = torch.stack(states), torch.cat(actions)
    payload, _ = layout.assemble_action(actions, states, torch.ones(len(actions),2,dtype=torch.bool))
    return layout, payload, actions, states, sn, fn, codecs


def raw_clip(layout, source_offset=0):
    h, _, k = scene(1 + (layout.num_frames-1)*8)
    return dict(keypoints=k, camera_poses=h,
                source_indexes=torch.arange(source_offset, source_offset+len(h)),
                coordinate_frame="world", units="metres", hand_order=["right", "left"])


@pytest.mark.parametrize("tail", [1,2,3])
@pytest.mark.parametrize("history", ["gt","pred_history","generated"])
def test_fixed_camera_runtime_roundtrip_all_tail_groups(tail, history):
    layout,payload,actions,states,sn,fn,codecs = clip(tail)
    result = evaluate_joint_actions(layout,payload,actions,states,
        state_normalizer=sn,future_normalizer=fn,history=history,hand_codecs=codecs,raw_gt=raw_clip(layout))
    assert result["source_indexes"] == list(range(1,len(actions)+1))
    assert [r["action_count"] for r in result["chunks"]] == [32,tail*8]
    for row in result["chunks"]:
        assert row["local_right_mpjpe_mm"] < 0.002
        assert row["local_left_mpjpe_mm"] < 0.002


def test_generated_points_and_camera_share_alignment_no_double_wrist_rotation():
    layout,payload,actions,states,sn,fn,codecs = clip()
    # Perturb first chunk camera orientation and translation, producing accumulated gauge drift.
    roles, chunks, _ = layout.action_metadata()
    from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION
    row = torch.where((roles==ACTION) & (chunks==1))[0][0]
    payload[row,0] += 0.3
    payload[row,3:9] = torch.tensor([0.,1.,0.,-1.,0.,0.])
    records,_ = decode_joint_actions(layout,payload,actions,states,
        state_normalizer=sn,future_normalizer=fn,history="generated",hand_codecs=codecs)
    item = records[1]
    alignment = item["predicted_rigid"][0,0] @ torch.linalg.inv(item["predicted"].rigid_chunk[0,0])
    expected = torch.einsum("ij,thnj->thni", alignment[:3,:3], item["predicted"].keypoints_chunk)+alignment[:3,3]
    torch.testing.assert_close(record_hand_keypoints(item,"predicted",codecs),expected,atol=1e-5,rtol=1e-5)
    # Projection is unchanged if camera and skeleton receive the same rigid frame change.
    intrinsics = torch.tensor([[100.,0,50],[0,100,50],[0,0,1.]])
    uv1, valid1 = project_chunk_hands(expected, item["predicted_rigid"][:,0], intrinsics)
    uv2, valid2 = project_chunk_hands(item["predicted"].keypoints_chunk,item["predicted"].rigid_chunk[:,0],intrinsics)
    import numpy as np
    np.testing.assert_array_equal(valid1,valid2)
    np.testing.assert_allclose(uv1[valid1],uv2[valid2],atol=0.01,rtol=1e-4)
    anchored = hand_keypoints(item["gt_anchor"].rigid_camera[None],item["gt_anchor"].hand_latents[None],codecs,representation=FIXED_CAMERA)
    altered = item["gt_anchor"].rigid_camera.clone()
    altered[1:,:3,:3] = torch.eye(3)
    unrotated = hand_keypoints(altered[None],item["gt_anchor"].hand_latents[None],codecs,representation=FIXED_CAMERA)
    wrist = item["gt_anchor"].rigid_camera[1:]
    expected = torch.einsum("hij,thnj->thni", wrist[:, :3, :3], unrotated-wrist[None,:,None,:3,3])+wrist[None,:,None,:3,3]
    torch.testing.assert_close(anchored, expected)
    assert not torch.allclose(anchored, unrotated)


def test_adapter_rejects_mixed_hashes_and_old_state():
    sn,fn,codecs = dependencies()
    adapter = ActionRepresentationAdapter(sn,fn,codecs)
    with pytest.raises(ValueError,match="state type"):
        adapter.encode_state(observation())
    fn.codec_sha256 = ("c"*64,"b"*64)
    with pytest.raises(ValueError,match="hashes"):
        ActionRepresentationAdapter(sn,fn,codecs)


def test_fixed_stream_reanchors_keeps_new_state_type_and_evicts():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sn,fn,codecs = dependencies()
        model = PerfectModel()
        model.action_representation = FIXED_CAMERA
        sampler = StreamingJointSampler(model,text_ids=[3,4,5],latent_shape=(4,2,2),
            state_normalizer=sn,future_normalizer=fn,hand_codecs=codecs,history="generated",chunk_size=4)
        original = observation()
        state = FixedCameraState(0,original.rigid_camera,original.hand_latents)
        for index in range(18):
            frames = 1 if index==17 else 4
            result = sampler.step(torch.ones(1,4,1,2,2) if index==0 else None,
                                  state if index==0 else None,frames=frames)
            assert isinstance(sampler._terminal,FixedCameraState)
            assert result.report["forward_calls"] == 32
            assert result.action.shape == (frames*8,64)
            if index==15:
                assert (sampler.cache.chunks==1).any()
            if index==16:
                assert not (sampler.cache.chunks==1).any()
        assert sampler.source_index == 17*32+8
    finally:
        torch.set_num_threads(previous)

@pytest.mark.parametrize("observed,c", [(None,4),("legacy_local_delta_absolute_hand_v1",4),(FIXED_CAMERA,3)])
def test_model_rejects_wrong_representation_before_parent(observed,c,monkeypatch):
    from test_ar_v02_model import bare_model
    from cosmos3_joint_video_hand_pose.src.ar_model import EgoVerseARModel
    from cosmos3_joint_video_hand_pose.src.ar_v02_layout import LAYOUT_VERSION
    model = bare_model(c)
    model.action_representation = FIXED_CAMERA
    def forbidden(*args):
        raise AssertionError("incompatible batch reached parent")
    monkeypatch.setattr(EgoVerseARModel,"_prepare_training_data",forbidden)
    with pytest.raises(ValueError,match="C=4"):
        model._prepare_training_data(dict(ar_layout_version=LAYOUT_VERSION,
            ar_boundary_states=[],ar_action_representation=observed),0)


def test_fixed_archive_requires_representation_and_codec_hashes(tmp_path):
    from cosmos3_joint_video_hand_pose.src.ar_v02_eval import save_rollout,load_rollout,_normalizers
    from cosmos3_joint_video_hand_pose.src.action_fixed_normalization import fit_fixed_normalizer
    import json,hashlib
    layout,payload,actions,states,sn,fn,codecs = clip()
    meta=dict(sample_id="synthetic",episode_id="fixture",history="gt",seed=42,source_offset=0,
              source_fps=30.,speed_factor=.5,action_representation=FIXED_CAMERA)
    for name,kind,values in (("state_normalizer","state",states[:,:57]),("future_normalizer","future",actions[:,:57])):
        path=tmp_path/(name+".json")
        path.write_text(json.dumps(fit_fixed_normalizer(values.numpy(),kind=kind,
                        codec_sha256=sn.codec_sha256,manifest_sha256="d"*64)))
        meta[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    kwargs=dict(layout=layout,predicted_action=payload,gt_future=actions,boundary_states=states,metadata=meta)
    with pytest.raises(ValueError,match="codec identities"):
        save_rollout(tmp_path/"missing.npz",**kwargs)
    meta["hand_codecs"]={side:dict(path=f"/synthetic/{side}.pt",sha256=c.checkpoint_sha256) for side,c in zip(("right","left"),codecs)}
    path=save_rollout(tmp_path/"bound.npz",raw_gt=raw_clip(layout),**kwargs)
    _,restored,_=load_rollout(path)
    ns,nf=_normalizers(restored,SimpleNamespace())
    assert ns.representation==nf.representation==FIXED_CAMERA
    restored["hand_codecs"]["right"]["sha256"]="f"*64
    with pytest.raises(ValueError,match="codec identity"):
        _normalizers(restored,SimpleNamespace())


def test_registered_fixed_dataset_configuration_matches_actual_factory_signature():
    import inspect
    from cosmos3_joint_video_hand_pose.src.config import _ar_v02_fixed_camera_experiment
    from cosmos3_joint_video_hand_pose.src.ar_dataset import get_egoverse_ar_dataset
    cfg = _ar_v02_fixed_camera_experiment()
    ds = cfg["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
    inspect.signature(get_egoverse_ar_dataset).bind(**{k:v for k,v in ds.items() if not k.startswith("_")})
