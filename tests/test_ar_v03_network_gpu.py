"""Real Cosmos pack/encode/attention/decode with V0.3's single-pass adapter."""

from types import SimpleNamespace

import pytest
import torch

from test_ar_v02_network_gpu import make_network
from test_ar_v03_model import model_fixture,data_fixture
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION,VIDEO,JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def test_official_training_step_runs_one_forward_and_real_backward():
    layouts=[JointChunkLayout(10,1,4),JointChunkLayout(6,1,4)]
    model=model_fixture(layouts,"cuda")
    model.net=make_network()
    data=data_fixture(layouts,"cuda")
    plans=[SimpleNamespace(has_action=True,has_sound=False) for _ in layouts]
    info={"skip_text":False,"initial_temporal_offset":0,"dual_kv_cache":None,"frame_idx":0}
    model._get_training_inputs=lambda batch,iteration: ([[3],[4,5]],plans,data,info,["480"]*2,None)
    count=[]
    handle=model.net.register_forward_pre_hook(lambda *args: count.append(1))
    def forbidden(*args,**kwargs):
        raise AssertionError("clean history forward must not execute")
    model._build_clean_tf_cache=forbidden
    output,loss=model.training_step({},1)
    assert len(count)==1
    assert torch.isfinite(loss)
    output["_backward_loss"].backward()
    grads=[p.grad for p in model.net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0
    raw_fields=[k for k in output if k.startswith("egoverse_loss_action_") and k.endswith("_raw")
                and k != "egoverse_loss_action_raw"]
    assert len(raw_fields)==8
    assert "_tf_memory_state" not in info
    handle.remove()


def test_noisy_chunk_receives_self_and_later_loss_gradients_through_full_network():
    torch.manual_seed(33)
    layout=JointChunkLayout(10,1,4)
    model=model_fixture([layout],"cuda")
    model.net=make_network()
    data=data_fixture([layout],"cuda")
    video=data.x0_tokens_vision[0].requires_grad_()
    action=data.x0_tokens_action[0].requires_grad_()
    packed=pack_joint_sequence(layout=layout,gen_data_clean=data,text_ids=[3,4],
        special_tokens=model.llm_special_tokens,timesteps=torch.linspace(0,900,layout.num_video_frames),latent_patch_size=2)
    packed.to_cuda()
    vr,vc,_=layout.video_metadata(device="cuda")
    ar,ac,_=layout.action_metadata(device="cuda")
    def gradients(chunks):
        # Official compiled FlexAttention donates backward buffers. Recompute
        # the same deterministic forward instead of changing production flags.
        memory=model.build_memory_state(packed,{})
        out=model.denoise(data_batch_packed=packed,memory=memory)
        loss=sum(out["preds_vision"][0][:,:,(vr==VIDEO)&(vc==chunk)].square().mean()
                 +out["preds_action"][0][(ar==ACTION)&(ac==chunk),:57].square().mean() for chunk in chunks)
        result=torch.autograd.grad(loss,(video,action))
        assert all(x is None for x in memory._clean_gen_kv)
        assert all(x is None for x in memory._clean_und_kv)
        return result
    grads_own=gradients([1])
    grads_later=gradients([2])
    grads_total=gradients([1,2])
    for name,a,b,total in zip(("video","action"),grads_own,grads_later,grads_total):
        region=((vr==VIDEO)&(vc==1)) if name=="video" else ((ar==ACTION)&(ac==1))
        aa=a[:,:,region] if name=="video" else a[region,:57]
        bb=b[:,:,region] if name=="video" else b[region,:57]
        assert torch.isfinite(aa).all() and aa.norm()>0
        assert torch.isfinite(bb).all() and bb.norm()>0
        future=(vc>1) if name=="video" else (ac>1)
        assert torch.count_nonzero(a[:,:,future] if name=="video" else a[future])==0
        torch.testing.assert_close(total,a+b,atol=1e-5,rtol=1e-4)
        print(f"V03_NETWORK_GRAD {name} own={float(aa.norm())} later={float(bb.norm())}")
