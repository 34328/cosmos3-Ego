"""Synthetic codecs verify geometry/algebra only, NOT learned AE quality."""
import pytest
import torch
from scipy.spatial.transform import Rotation
from cosmos3_joint_video_hand_pose.src import action_fixed_camera as m


class FakeCodec:
    coordinate_system = "fixed-camera"
    representation = m.REPRESENTATION
    input_frame = m.INPUT_FRAME

    def encode(self, q):
        # Rotation-closed synthetic subspace: five points repeated four times.
        return torch.asinh(q[..., :5, :].flatten(-2))

    def decode(self, z):
        p = torch.sinh(z).reshape(z.shape[:-1] + (5, 3))
        return p.repeat(*([1] * (p.ndim - 2)), 4, 1)


CODECS = (FakeCodec(), FakeCodec())


def test_fp32_rigid_inverse_keeps_exact_homogeneous_row():
    # Real valid camera pose that exposed LU bottom-row noise during statistics.
    h = torch.tensor([
        [-0.001339922659099102, -0.6907328367233276, 0.7231088280677795, 0.018149999901652336],
        [-0.9842380881309509, -0.12696625292301178, -0.12310533225536346, -0.03683700039982796],
        [0.1768433153629303, -0.7118762135505676, -0.6796754002571106, 1.5793770551681519],
        [0, 0, 0, 1],
    ])
    m._rigid(h)
    inverse = m._rigid_inverse(h)
    assert torch.equal(inverse[3], h[3])
    torch.testing.assert_close(inverse.double(), torch.linalg.inv(h.double()), atol=2e-7, rtol=2e-7)
    w = h.repeat(2, 2, 1, 1)
    w[:, :, :3, 3] += torch.tensor([.2, -.1, .3])
    k = w[:, :, None, :3, 3].repeat(1, 1, 21, 1)
    k[:, :, 1:] += .01
    encoded = m.encode_chunk_physical(h.repeat(2, 1, 1), w, k, [0, 1], CODECS)
    assert torch.equal(encoded.state.rigid_camera[:, 3], h[3].expand(3, 4))
    bad = h.clone()
    bad[3, 0] = .01
    with pytest.raises(ValueError, match="homogeneous"):
        m._rigid_inverse(bad)


def scene(n=34):
    torch.manual_seed(71)
    r = torch.tensor(Rotation.from_euler("xyz", [[.07*t, .03*t+.2, -.09*t] for t in range(n*3)]).as_matrix())
    rigid = torch.eye(4, dtype=torch.float64).repeat(n, 3, 1, 1)
    rigid[..., :3, :3] = r.reshape(n, 3, 3, 3)
    rigid[..., :3, 3] = torch.randn(n, 3, 3, dtype=torch.float64)*.1
    q = torch.randn(n, 2, 5, 3, dtype=torch.float64)*.02
    q = q.repeat(1, 1, 4, 1)
    p = rigid[:, 1:, :3, 3]
    points = torch.cat((p[:, :, None], p[:, :, None]+q), dim=2)
    return rigid[:, 0], rigid[:, 1:], points


def encode(h, w, k, start=100):
    return m.encode_chunk_physical(h, w, k, range(start, start+len(h)), CODECS)


@pytest.mark.parametrize("n", [2, 9, 17, 25, 33, 34])
def test_roundtrip_and_times(n):
    h, w, k = scene(n)
    e = encode(h, w, k)
    d = m.decode_future_physical(e.state, e.actions, CODECS)
    inv = torch.linalg.inv(h[0])
    expected = inv @ torch.cat((h[:, None], w), 1)
    points = torch.einsum("ij,thnj->thni", inv[:3, :3], k[1:])+inv[:3, 3]
    torch.testing.assert_close(d.rigid_chunk, expected[1:], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(d.keypoints_chunk, points, atol=1e-12, rtol=1e-12)
    assert d.source_indices.tolist() == list(range(101, 100+n))
    assert d.end_state.source_index == 100+n-1
    torch.testing.assert_close(d.wrist_camera, torch.linalg.inv(h[1:])[:, None] @ w[1:])
    assert d.wrist_camera is d.wrists_camera


def test_latent_deltas_not_raw_xyz_deltas_or_full_z():
    h, w, k = scene(5)
    e = encode(h, w, k)
    q = torch.einsum("thji,thnj->thni", w[..., :3, :3], k[:, :, 1:]-w[:, :, None, :3, 3])
    z = torch.stack([CODECS[i].encode(q[:, i]) for i in range(2)], 1)
    torch.testing.assert_close(e.actions[:, 18:33], z[1:, 0]-z[:-1, 0])
    assert not torch.allclose(e.actions[:, 18:33], CODECS[0].encode(q[1:, 0]-q[:-1, 0]))
    d = m.decode_future_physical(e.state, e.actions, CODECS)
    torch.testing.assert_close(d.hand_latents, z[1:])


def test_left_rotation_and_fixed_translation():
    h, w, k = scene(5)
    e = encode(h, w, k)
    rigid = torch.linalg.inv(h[0]) @ torch.cat((h[:, None], w), 1)
    p, _ = m._split(e.actions)
    delta = m._matrices(p)
    r = rigid[..., :3, :3]
    torch.testing.assert_close(delta[..., :3, :3], r[1:] @ r[:-1].transpose(-1, -2))
    assert not torch.allclose(r[1:] @ r[:-1].transpose(-1,-2), r[:-1].transpose(-1,-2) @ r[1:])
    torch.testing.assert_close(delta[..., :3, 3], rigid[1:, :, :3, 3]-rigid[:-1, :, :3, 3])


def test_chunk_boundary_clone_and_tail():
    h, w, k = scene(40)
    first = encode(h[:33], w[:33], k[:33])
    d1 = m.decode_future_physical(first.state, first.actions, CODECS)
    second = encode(h[32:], w[32:], k[32:], start=132)
    torch.testing.assert_close(d1.end_state.rigid_camera, second.state.rigid_camera, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(d1.end_state.hand_latents, second.state.hand_latents, atol=1e-12, rtol=1e-12)
    d2 = m.decode_future_physical(d1.end_state, second.actions, CODECS)
    expected = torch.einsum("ij,thnj->thni", torch.linalg.inv(h[32])[:3,:3], k[33:])+torch.linalg.inv(h[32])[:3,3]
    torch.testing.assert_close(d2.keypoints_chunk, expected, atol=1e-12, rtol=1e-12)
    assert d2.source_indices.tolist() == list(range(133, 140))


def test_hand_points_multiply_wrist_rotation_exactly_once():
    h, w, k = scene(3)
    e = encode(h, w, k)
    modified = e.actions.clone()
    modified[:, 12:18] = torch.tensor([1., 0, 0, 0, 1, 0])
    a = m.decode_future_physical(e.state, e.actions, CODECS)
    b = m.decode_future_physical(e.state, modified, CODECS)
    assert not torch.allclose(a.keypoints_chunk[:,0], b.keypoints_chunk[:,0])
    torch.testing.assert_close(a.keypoints_chunk[:,1], b.keypoints_chunk[:,1])
    for decoded in (a,b):
        local = CODECS[0].decode(decoded.hand_latents[:,0])
        wrist = decoded.rigid_chunk[:,1]
        expected = wrist[:,None,:3,3] + torch.einsum("tij,tnj->tni", wrist[:,:3,:3], local)
        torch.testing.assert_close(decoded.keypoints_chunk[:,0,1:], expected)
    assert not torch.allclose(a.rigid_chunk[:,1,:3,:3], b.rigid_chunk[:,1,:3,:3])


def test_state_padding_and_normalizers():
    h, w, k = scene(3)
    e = encode(h, w, k)
    v = m.encode_state_physical(e.state)
    assert not v[:9].any()
    s = m.decode_state_physical(m.pad_action(v), source_index=100)
    torch.testing.assert_close(s.rigid_camera, e.state.rigid_camera)
    torch.testing.assert_close(s.hand_latents, e.state.hand_latents)

    class Norm:
        def __init__(self, shift): self.shift = shift
        def normalize(self, x):
            assert x.shape[-1] == 57
            return (x-self.shift)/3
        def denormalize(self, x): return x*3+self.shift

    sn, fn = Norm(2), Norm(7)
    torch.testing.assert_close(m.denormalize_state(m.normalize_state(v, sn), sn), v)
    torch.testing.assert_close(m.denormalize_future(m.normalize_future(e.actions, fn), fn), e.actions)
    torch.testing.assert_close(m.decode_future_physical(s, m.pad_action(e.actions), CODECS).hand_latents,
                               m.decode_future_physical(s, e.actions, CODECS).hand_latents)
    bad = m.pad_action(v)
    bad[-1] = 1
    with pytest.raises(ValueError, match="padding"): m.decode_state_physical(bad, 100)


def test_codec_identity_and_time_validation():
    h, w, k = scene(3)
    class OldCodec(FakeCodec): coordinate_system = "wrist-local"
    with pytest.raises(ValueError, match="fixed-camera"):
        m.encode_chunk_physical(h, w, k, range(3), (OldCodec(), FakeCodec()))
    with pytest.raises(ValueError, match="consecutive"):
        m.encode_chunk_physical(h, w, k, [0,1,3], CODECS)
    e = encode(h,w,k)
    with pytest.raises(ValueError, match="boundary"):
        m.decode_future_physical(e.state, e.actions, CODECS, [102,103])
    with pytest.raises(ValueError, match="integer"):
        m.encode_chunk_physical(h,w,k,[0.,1.,2.],CODECS)
    bad = e.actions.clone()
    bad[:,3:9] = 0
    with pytest.raises(ValueError, match="degenerate"):
        m.decode_future_physical(e.state,bad,CODECS)


def test_fp32_and_quaternion_inputs():
    h,w,k = scene(3)
    def pose(x):
        q = torch.tensor(Rotation.from_matrix(x[:,:3,:3].numpy()).as_quat())
        return torch.cat((x[:,:3,3],q[:,[3,0,1,2]]),-1).float()
    e = encode(pose(h), torch.stack((pose(w[:,0]),pose(w[:,1])),1), k.float())
    d = m.decode_future_physical(e.state,e.actions,CODECS)
    assert d.keypoints_chunk.dtype == torch.float32
    expected = torch.linalg.inv(h[0]) @ torch.cat((h[:,None],w),1)
    torch.testing.assert_close(d.rigid_chunk.double(),expected[1:],atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize("frame", [0, 1, 2])
@pytest.mark.parametrize("kind", ["zero_quaternion", "nan_pose", "nan_keypoint"])
def test_missing_hand_frames_are_rejected_without_skipping(frame, kind):
    h, w, k = scene(3)
    if kind == "zero_quaternion":
        def pose(x):
            q = torch.tensor(Rotation.from_matrix(x[:, :3, :3].numpy()).as_quat())
            return torch.cat((x[:, :3, 3], q[:, [3, 0, 1, 2]]), -1)
        w = torch.stack((pose(w[:, 0]), pose(w[:, 1])), 1)
        w[frame, 0] = 0
    elif kind == "nan_pose":
        w[frame, 1, 0, 3] = float("nan")
    else:
        k[frame, 0, 4, 1] = float("nan")
    with pytest.raises(ValueError):
        encode(h, w, k)


def test_fp32_long_rollout_reanchors_without_rotation_drift():
    h, w, k = scene(33)
    # Repeat a closed local-shape cycle. Repeating an arbitrary nonzero net dz
    # with this sinh fixture grows "hands" to 24 metres after 76 blocks.
    local0 = torch.einsum("hji,hnj->hni",w[0,:,:3,:3],k[0,:,1:]-w[0,:,None,:3,3])
    k[-1,:,1:] = w[-1,:,None,:3,3] + torch.einsum("hij,hnj->hni",w[-1,:,:3,:3],local0)
    e32 = encode(h.float(), w.float(), k.float())
    # Match the exact FP32 inputs for a FP64 accumulation reference.
    initial = m.FixedCameraState(e32.state.source_index,
        e32.state.rigid_camera.double(), e32.state.hand_latents.double())
    s32, s64 = e32.state, initial
    for _ in range(100):
        d32 = m.decode_future_physical(s32, e32.actions, CODECS)
        d64 = m.decode_future_physical(s64, e32.actions.double(), CODECS)
        assert torch.isfinite(d32.rigid_chunk).all() and torch.isfinite(d32.keypoints_chunk).all()
        torch.testing.assert_close(d32.keypoints_chunk.double(),d64.keypoints_chunk,atol=1e-4,rtol=1e-4)
        all_r = d32.rigid_chunk[..., :3, :3]
        torch.testing.assert_close(all_r.transpose(-1,-2) @ all_r,
            torch.eye(3).expand_as(all_r),atol=5e-7,rtol=5e-7)
        torch.testing.assert_close(torch.linalg.det(all_r),torch.ones_like(all_r[...,0,0]),atol=5e-7,rtol=5e-7)
        s32, s64 = d32.end_state, d64.end_state
        r = s32.rigid_camera[:, :3, :3]
        torch.testing.assert_close(r.transpose(-1,-2) @ r,
            torch.eye(3).expand_as(r), atol=5e-7, rtol=5e-7)
        torch.testing.assert_close(torch.linalg.det(r), torch.ones(3), atol=5e-7, rtol=5e-7)
    assert s32.source_index == 3300
    torch.testing.assert_close(s32.rigid_camera.double(), s64.rigid_camera, atol=1e-4, rtol=1e-4)


def test_gross_invalid_rotation_still_rejected():
    h, w, k = scene(3)
    w[1, 0, :3, :3] *= 1.01
    with pytest.raises(ValueError, match="orthogonal"):
        encode(h, w, k)


@pytest.mark.parametrize("tail", [8, 16, 24])
def test_reanchor_100_blocks_and_tail_copies_z_without_codec_calls(tail):
    class NoCalls(FakeCodec):
        def encode(self, q): raise AssertionError("reanchor called encode")
        def decode(self, z): raise AssertionError("reanchor called decode")
    h, w, _ = scene(102)
    z = torch.linspace(-.08, .09, 30, dtype=torch.float64).reshape(2,15)
    local = torch.stack([c.decode(z[i]) for i,c in enumerate(CODECS)])
    previous = torch.cat((h[:1], w[0]))
    index = 0
    for block in range(101):
        # Rigid poses are expressed in the previous block's boundary axes.
        world = torch.cat((h[block+1:block+2], w[block+1]))
        rigid = torch.linalg.inv(previous[0]) @ world
        index += 32 if block < 100 else tail
        before = rigid[1:,None,:3,3] + torch.einsum("hij,hnj->hni", rigid[1:,:3,:3],local)
        state = m.reanchor_state(rigid,z,index,(NoCalls(),NoCalls()))
        assert torch.equal(state.hand_latents,z)
        assert state.hand_latents.data_ptr() != z.data_ptr()
        after = state.rigid_camera[1:,None,:3,3] + torch.einsum("hij,hnj->hni",state.rigid_camera[1:,:3,:3],local)
        inv = torch.linalg.inv(rigid[0])
        torch.testing.assert_close(after, torch.einsum("ij,hnj->hni",inv[:3,:3],before)+inv[:3,3],atol=1e-12,rtol=1e-12)
        previous = world
    assert index == 3200 + tail


def test_constant_wrist_local_shape_has_zero_dz_despite_rigid_motion():
    h,w,_ = scene(33)
    z=torch.linspace(-.02,.03,30,dtype=torch.float64).reshape(2,15)
    q=torch.stack([c.decode(z[i]) for i,c in enumerate(CODECS)])
    p=w[..., :3,3]
    points=p[:,:,None]+torch.einsum("thij,hnj->thni",w[...,:3,:3],q)
    k=torch.cat((p[:,:,None],points),dim=2)
    encoded=encode(h,w,k)
    _,dz=m._split(encoded.actions)
    torch.testing.assert_close(dz,torch.zeros_like(dz),atol=1e-14,rtol=0)
    torch.testing.assert_close(encoded.state.hand_latents,z,atol=1e-14,rtol=0)


def test_same_dimension_camera_axis_codec_is_rejected():
    class CameraAxis(FakeCodec):
        representation = "fixed_camera_delta_latent_v1"
        input_frame = "chunk_camera_axes_wrist_origin"
    h,w,k=scene(2)
    with pytest.raises(ValueError,match="old camera-axis"):
        m.encode_chunk_physical(h,w,k,range(2),(CameraAxis(),CameraAxis()))
