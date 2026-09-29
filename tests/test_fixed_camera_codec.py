import json
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import fixed_offsets, train_side, sha256, boundary_indices, ae_losses
from cosmos3_joint_video_hand_pose.scripts.validate_fixed_camera_codec import evaluate, shape_measurements
from cosmos3_joint_video_hand_pose.src.codec_fixed_camera import FrozenFixedCameraHandAE15, REPRESENTATION, INPUT_FRAME


@pytest.fixture
def artifact(tmp_path):
    torch.manual_seed(12)
    x=torch.randn(32,20,3)*.03
    fit=dict(split="train",episode_ids=["train"],manifest_sha256="a"*64,episodes_sha256="b"*64,
             source_hashes=dict(train_episodes="b"*64,heldout_episodes="b"*64,train_segments="c"*64,heldout_segments="d"*64),
             representation=REPRESENTATION,input_frame=INPUT_FRAME,
             source_windows=[dict(episode="train",window="train-window",start=0,span=32)])
    payload=train_side(x,fit,"right",0,16,5e-4,42,"cpu")
    path=tmp_path/"right_mlp15.pt"
    torch.save(payload,path)
    return path,x


def test_wrist_local_axes_independent_of_camera():
    head=np.array([[0,0,0,1,0,0,0.]],float)
    r=Rotation.from_euler("z",90,degrees=True)
    xyzw=r.as_quat()
    head[0,3:]=xyzw[[3,0,1,2]]
    wrist=np.array([[3,4,5,1,0,0,0]],float)
    points=np.tile(wrist[:,:3,None].transpose(0,2,1),(1,21,1))
    points[:,1:,0]+=1
    q=fixed_offsets(head,wrist,points)
    np.testing.assert_allclose(fixed_offsets(head,wrist,points.reshape(1,63)),q)
    np.testing.assert_allclose(q[0,0],[1,0,0],atol=1e-7)
    wrist[:,3:]=head[:,3:]
    np.testing.assert_allclose(fixed_offsets(head,wrist,points)[0,0],[0,-1,0],atol=1e-7)
    head[:,:3]+=20
    head[:,3:]=[1,0,0,0]
    np.testing.assert_allclose(fixed_offsets(head,wrist,points)[0,0],[0,-1,0],atol=1e-7)


def test_identity_dtype_hash_and_no_legacy(artifact):
    path,x=artifact
    c=FrozenFixedCameraHandAE15(path,expected_sha256=sha256(path),allow_unvalidated=True)
    assert c.representation==REPRESENTATION and c.coordinate_system=="fixed-camera"
    assert c.checkpoint_path==str(path.resolve())
    assert c.encode(x.double()).dtype==torch.float64
    assert c.decode(c.encode(x)).shape==x.shape
    assert not any(p.requires_grad for p in c.parameters())
    with pytest.raises(ValueError,match="hash"):
        FrozenFixedCameraHandAE15(path,expected_sha256="0"*64,allow_unvalidated=True)
    payload=torch.load(path,weights_only=True)
    payload.pop("representation")
    torch.save(payload,path)
    with pytest.raises(ValueError,match="representation"):
        FrozenFixedCameraHandAE15(path,allow_unvalidated=True)


def test_stats_tamper_and_sidecar_required(artifact):
    path,x=artifact
    with pytest.raises(ValueError,match="sidecar"):
        FrozenFixedCameraHandAE15(path)
    p=torch.load(path,weights_only=True)
    p["state_dict"]["mean"][0]+=1
    torch.save(p,path)
    with pytest.raises(ValueError,match="statistics hash"):
        FrozenFixedCameraHandAE15(path,allow_unvalidated=True)


def test_production_reanchor_with_real_network(artifact):
    path,q=artifact
    codec=FrozenFixedCameraHandAE15(path,allow_unvalidated=True)
    rotations=Rotation.from_euler("zyx",[[0,0,0],[20,10,5],[-5,30,15]],degrees=True).as_matrix().astype("float32")
    metrics=evaluate(codec,q,[(q[0].numpy(),rotations)])
    invariant=metrics["reanchor_invariance"]
    assert invariant["passed"] and invariant["latent_exact"] and invariant["no_reencode"]
    assert invariant["full_chunks"] == 100 and invariant["tail_frames"] == 8
    assert invariant["max_point_error_m"] <= 1e-5


def test_train_refuses_heldout():
    with pytest.raises(ValueError,match="train-only"):
        train_side(torch.randn(2,20,3),{"split":"heldout"},"right",1,2,.001,1,"cpu")


def test_c4_boundary_indices_and_tail():
    assert boundary_indices(257)==list(range(0,257,32))
    assert boundary_indices(33)==[0,32]
    assert boundary_indices(70)==[0,32,64,69]
    with pytest.raises(ValueError): boundary_indices(1)


def test_consistency_target_detached_and_weight_zero_compatibility():
    enc,dec=torch.nn.Linear(2,2,bias=False),torch.nn.Linear(2,2,bias=False)
    with torch.no_grad():
        enc.weight.copy_(torch.eye(2)*.8);dec.weight.copy_(torch.eye(2)*.9)
    x=torch.ones(3,2,requires_grad=True)
    total,rec,cons=ae_losses(enc,dec,x,1.)
    torch.testing.assert_close(total,rec+cons)
    cons.backward()
    assert x.grad is None
    assert enc.weight.grad.abs().sum()>0 and dec.weight.grad.abs().sum()>0
    total,rec,cons=ae_losses(enc,dec,x,0.)
    torch.testing.assert_close(total,(dec(enc(x))-x).square().mean())
    assert cons.item()==0
    with pytest.raises(ValueError): ae_losses(enc,dec,x,-1)


def test_shape_diagnostic_detects_collapsed_hand():
    torch.manual_seed(8)
    q=torch.randn(3,20,3)*.04
    exact=shape_measurements(q,q)
    torch.testing.assert_close(exact[:,0],torch.ones(3))
    assert exact[:,1].max()==0
    collapsed=shape_measurements(torch.zeros_like(q),q)
    assert collapsed[:,0].max()==0 and collapsed[:,1].min()>10



def test_sidecar_checks_actual_metrics(artifact):
    path,x=artifact
    fit=torch.load(path,weights_only=True)["fit"]
    report=dict(schema_version=2,input_frame=INPUT_FRAME,representation=REPRESENTATION,checkpoint_sha256=sha256(path),passed=True,
                heldout_episode_ids=["heldout"],heldout_sample_count=4,
                heldout_provenance={**fit,"split":"heldout","episode_ids":["heldout"],
                    "source_windows":[dict(episode="heldout",window="heldout-window",start=0,span=32)]},
                thresholds=dict(reconstruction_mean_mm=5,reconstruction_p95_mm=15),
                metrics=dict(reconstruction=dict(mean_mm=20,p95_mm=30),trajectory_count=1,
                    reanchor_invariance=dict(passed=True,latent_exact=True,no_reencode=True,full_chunks=100,tail_frames=8,
                        max_point_error_m=1e-6,nonzero_wrist_rotation=True,nonzero_camera_rotation=True)))
    path.with_suffix(".validation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="metrics"):
        FrozenFixedCameraHandAE15(path)
    report['metrics']['reconstruction'] = dict(mean_mm=1,p95_mm=2)
    path.with_suffix('.validation.json').write_text(json.dumps(report))
    assert FrozenFixedCameraHandAE15(path).checkpoint_sha256 == sha256(path)
    for field,value in (("latent_exact",False),("no_reencode",False),("full_chunks",99),
                        ("nonzero_wrist_rotation",False),("nonzero_camera_rotation",False),
                        ("max_point_error_m",float("nan")),("tail_frames",0)):
        original=report["metrics"]["reanchor_invariance"][field]
        report["metrics"]["reanchor_invariance"][field]=value
        path.with_suffix('.validation.json').write_text(json.dumps(report))
        with pytest.raises(ValueError,match="metrics"):
            FrozenFixedCameraHandAE15(path)
        report["metrics"]["reanchor_invariance"][field]=original
    report["thresholds"]["reconstruction_mean_mm"]=6
    path.with_suffix('.validation.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match="metrics"):
        FrozenFixedCameraHandAE15(path)
    report["thresholds"]["reconstruction_mean_mm"]=5
    report['heldout_episode_ids']=['train']
    path.with_suffix('.validation.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='leakage'):
        FrozenFixedCameraHandAE15(path)


@pytest.mark.parametrize("mutation", ["old_representation", "metadata_only", "data_binding", "source_episode"])
def test_old_camera_weights_cannot_be_relabelled(artifact, mutation):
    path,_=artifact
    p=torch.load(path,weights_only=True)
    if mutation == "old_representation":
        p["representation"]="fixed_camera_delta_latent_v1"
    elif mutation == "metadata_only":
        # Camera-axis checkpoints have identical learned layer shapes but no
        # wrist-local data binding tensor; changing metadata cannot add it.
        del p["state_dict"]["wrist_local_data_binding"]
    elif mutation == "data_binding":
        p["fit"]["data_sha256"]="0"*64
    else:
        p["fit"]["source_windows"][0]["episode"]="heldout"
    torch.save(p,path)
    with pytest.raises(ValueError):
        FrozenFixedCameraHandAE15(path,allow_unvalidated=True)


def test_collector_requires_valid_current_wrist_rotation():
    head=np.array([[0,0,0,1,0,0,0.]],float)
    wrist=head.copy()
    points=np.zeros((1,21,3))
    wrist[:,3:]=0
    with pytest.raises(ValueError,match="quaternion"):
        fixed_offsets(head,wrist,points)


def test_fit_requires_current_source_provenance_before_optimization():
    with pytest.raises(ValueError,match="wrist-local"):
        train_side(torch.randn(2,20,3),dict(split="train"),"right",0,2,.001,1,"cpu")


@pytest.fixture
def source_audit(tmp_path):
    import csv
    import zarr
    paths = {}
    windows = {}
    for side, split in enumerate(("train", "heldout")):
        eid = split + "-episode"
        group_path = tmp_path / (split + ".zarr")
        group = zarr.open_group(str(group_path), mode="w")
        group.attrs["total_frames"] = 65
        pose = np.zeros((65,7),dtype=np.float32)
        pose[:,3] = 1
        points = np.full((65,21,3), .03, dtype=np.float32)
        for name in ("obs_head_pose","right.obs_wrist_pose","left.obs_wrist_pose"):
            group.create_array(name, data=pose)
        for name in ("right.obs_keypoints","left.obs_keypoints"):
            group.create_array(name, data=points)
        path = tmp_path / (split + ".csv")
        with path.open("w") as stream:
            writer = csv.DictWriter(stream,fieldnames=["episode_hash","split","abs_zarr_path"])
            writer.writeheader()
            writer.writerow(dict(episode_hash=eid,split=split,abs_zarr_path=str(group_path)))
        paths[split+"_episodes"] = path
        path = tmp_path / (split + "_segments.csv")
        path.write_text("episode_hash,split,start_idx,end_idx\n"+eid+","+split+",0,65\n")
        paths[split+"_segments"] = path
        windows[eid+":0:0:65"] = dict(episode=eid,split=split,span=65,frames=33,starts=[0])
    manifest = dict(schema="ar_v02_codec_source_windows_v1",representation=REPRESENTATION,
                    frame_stride=2,chunk_size=4,tokens_per_latent=8,
                    source_hashes={name:sha256(path) for name,path in paths.items()},
                    tracking_validation=dict(version="fixed_camera_float32_v1",quaternion_norm_atol=1e-4,
                        so3_atol=1e-5,so3_rtol=1e-5,missing_hand="exclude_entire_window",
                        finite_dtype="float32",scope="all_source_frames"),windows=windows)
    path = tmp_path/"valid_windows.json"
    path.write_text(json.dumps(manifest))
    return paths,path,manifest


def test_bootstrap_collect_needs_no_codec_and_split_csvs_are_bound(source_audit):
    from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import collect
    from cosmos3_joint_video_hand_pose.src.codec_fixed_camera import validate_source_pair
    paths,path,manifest=source_audit
    provenance={}
    for split in ("train","heldout"):
        data,_,provenance[split]=collect(paths[split+"_episodes"],path,split,1,1,1)
        assert data["right"].shape==(66,20,3)
        assert provenance[split]["source_hashes"]==manifest["source_hashes"]
        assert len(provenance[split]["source_windows"][0]["input_sha256"])==64
    assert provenance["train"]["episodes_sha256"] != provenance["heldout"]["episodes_sha256"]
    validate_source_pair(provenance["train"],provenance["heldout"])
    provenance["heldout"]["source_hashes"] = dict(provenance["heldout"]["source_hashes"],train_segments="0"*64)
    with pytest.raises(ValueError,match="provenance"):
        validate_source_pair(provenance["train"],provenance["heldout"])


@pytest.mark.parametrize("mutation",["no_schema","old_schema","old_representation","stride","chunk","tokens",
                                     "no_sources","missing_hash","bad_hash","episode_hash","tracking","missing_tracking",
                                     "wrong_episode","split_overlap"])
def test_collect_rejects_unversioned_stale_or_unbound_audit(source_audit,mutation):
    from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import collect
    paths,path,m=source_audit
    if mutation=="no_schema": m.pop("schema")
    elif mutation=="old_schema": m["schema"]="ar_v02_valid_windows_v1"
    elif mutation=="old_representation": m["representation"]="fixed_camera_delta_latent_v1"
    elif mutation=="stride": m["frame_stride"]=1
    elif mutation=="chunk": m["chunk_size"]=1
    elif mutation=="tokens": m["tokens_per_latent"]=4
    elif mutation=="no_sources": m.pop("source_hashes")
    elif mutation=="missing_hash": m["source_hashes"].pop("heldout_segments")
    elif mutation=="bad_hash": m["source_hashes"]["train_segments"]="z"*64
    elif mutation=="episode_hash": m["source_hashes"]["train_episodes"]="0"*64
    elif mutation=="tracking": m["tracking_validation"]["finite_dtype"]="float64"
    elif mutation=="missing_tracking": m.pop("tracking_validation")
    elif mutation=="wrong_episode": m["windows"]["train-episode:0:0:65"]["episode"]="unknown"
    else: m["windows"]["heldout-episode:0:0:65"]["episode"]="train-episode"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        collect(paths["train_episodes"],path,"train",1,1,1)


def test_collect_accepts_current_formal_manifest_only_with_codec_binding(source_audit):
    from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import collect
    from cosmos3_joint_video_hand_pose.src.action_fixed_normalization import VALID_WINDOWS_SCHEMA
    paths,path,m=source_audit
    m["schema"]=VALID_WINDOWS_SCHEMA
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match="codec hashes"):
        collect(paths["train_episodes"],path,"train",1,1,1)
    m["codec_sha256"]=["1"*64,"2"*64]
    path.write_text(json.dumps(m))
    assert len(collect(paths["train_episodes"],path,"train",1,1,1)[0]["left"])==66


def test_collect_rejects_source_csv_changed_after_audit(source_audit):
    from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import collect
    paths,path,_=source_audit
    with paths["train_episodes"].open("a") as stream: stream.write("\n")
    with pytest.raises(ValueError,match="episodes hash mismatch"):
        collect(paths["train_episodes"],path,"train",1,1,1)


@pytest.mark.parametrize("mutation",["head_quaternion","head_nonfinite","wrist_quaternion","wrist_overflow",
                                     "left_allzero","right_nonfinite","short_stream"])
def test_collect_revalidates_all_actual_source_frames(source_audit,mutation):
    import zarr
    from cosmos3_joint_video_hand_pose.scripts.prepare_fixed_camera_codec import collect
    paths,path,_=source_audit
    group=zarr.open_group(str(path.parent/"train.zarr"),mode="a")
    if mutation=="head_quaternion": group["obs_head_pose"][17,3:]=0
    elif mutation=="head_nonfinite": group["obs_head_pose"][17,0]=float("nan")
    elif mutation=="wrist_quaternion": group["left.obs_wrist_pose"][17,3]=1.01
    elif mutation=="wrist_overflow": group["right.obs_wrist_pose"][17,0]=float("inf")
    elif mutation=="left_allzero": group["left.obs_keypoints"][17]=0
    elif mutation=="right_nonfinite": group["right.obs_keypoints"][17,3,0]=float("nan")
    else: group["left.obs_keypoints"].resize((64,21,3))
    with pytest.raises(ValueError,match="source"):
        collect(paths["train_episodes"],path,"train",1,1,1)


@pytest.mark.parametrize("target",["episodes","windows"])
def test_collect_detects_manifest_changes_during_sampling(source_audit,monkeypatch,target):
    from cosmos3_joint_video_hand_pose.scripts import prepare_fixed_camera_codec as module
    paths,path,_=source_audit
    original=module.fixed_offsets
    changed=False
    def offsets(*args,**kwargs):
        nonlocal changed
        if not changed:
            destination=paths["train_episodes"] if target=="episodes" else path
            with destination.open("a") as stream: stream.write("\n")
            changed=True
        return original(*args,**kwargs)
    monkeypatch.setattr(module,"fixed_offsets",offsets)
    with pytest.raises(ValueError,match="changed while collecting"):
        module.collect(paths["train_episodes"],path,"train",1,1,1)
