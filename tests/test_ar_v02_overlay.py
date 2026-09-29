"""Wrist-local projection applies wrist rotation exactly once."""
import numpy as np
import torch
import pytest
from test_fixed_camera_runtime import clip, raw_clip
from cosmos3_joint_video_hand_pose.src.action_representation import FIXED_CAMERA
from cosmos3_joint_video_hand_pose.src.ar_v02_evaluation import hand_keypoints
from cosmos3_joint_video_hand_pose.src.ar_v02_overlay import project_chunk_hands, render_joint_overlay


def test_wrist_rotation_once_then_camera_projection():
    class Codec:
        def decode(self,z):
            return z.new_tensor([.1,0.,0.]).expand(len(z),20,3)
    rigid=torch.eye(4).repeat(1,3,1,1)
    rigid[:,1:,:3,:3]=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rigid[:,1:,2,3]=1.
    points=hand_keypoints(rigid,torch.zeros(1,2,15),(Codec(),Codec()),representation=FIXED_CAMERA)
    torch.testing.assert_close(points[0,0,1],torch.tensor([0.,.1,1.]))
    uv,valid=project_chunk_hands(points,rigid[:,0],np.diag([100.,100.,1.]))
    np.testing.assert_allclose(uv[0,0,1],[0.,10.],atol=1e-6)
    assert valid.all()


def test_overlay_uses_raw_gt_and_dense_sources(monkeypatch):
    from cosmos3_joint_video_hand_pose.src import ar_v02_overlay as overlay
    layout,payload,actions,states,sn,fn,codecs=clip()
    raw=raw_clip(layout,73)
    captured=[]
    original=overlay.project_chunk_hands
    def capture(points,camera,k):
        captured.append(np.asarray(points).copy())
        return original(points,camera,k)
    monkeypatch.setattr(overlay,"project_chunk_hands",capture)
    raw["keypoints"][:,:,6,0]+=.021
    _,timeline=render_joint_overlay(layout,payload,actions,states,
        gt_rgb=np.zeros((41,48,64,3),dtype=np.uint8),
        generated_rgb_chunks=[np.zeros((17,48,64,3),dtype=np.uint8),np.zeros((5,48,64,3),dtype=np.uint8)],
        intrinsics=np.eye(3),gt_pixel_transform=np.eye(3),generated_pixel_transform=np.eye(3),
        state_normalizer=sn,future_normalizer=fn,hand_codecs=codecs,raw_gt=raw,source_offset=73,history="generated")
    from cosmos3_joint_video_hand_pose.src.ar_v02_evaluation import raw_gt_in_chunk
    aligned=raw_gt_in_chunk(layout,raw,source_offset=73,reference=payload)
    np.testing.assert_allclose(captured[0],aligned[1][1:])
    np.testing.assert_allclose(captured[3],aligned[2][1:])
    np.testing.assert_allclose(captured[6],aligned[1][:1])
    assert timeline["gt_hand_source"]=="raw_gt_keypoints"
    assert timeline["source_indexes"].tolist()==list(range(73,114))
    assert timeline["generated_chunk_ids"][33]==1
    assert timeline["generated_source_indexes"][33]==105
    with pytest.raises(ValueError,match="raw GT keypoints required"):
        render_joint_overlay(layout,payload,actions,states,
            gt_rgb=np.zeros((41,48,64,3),dtype=np.uint8),
            generated_rgb_chunks=[np.zeros((17,48,64,3),dtype=np.uint8),np.zeros((5,48,64,3),dtype=np.uint8)],
            intrinsics=np.eye(3),gt_pixel_transform=np.eye(3),generated_pixel_transform=np.eye(3),
            state_normalizer=sn,future_normalizer=fn,hand_codecs=codecs)
