"""Real Cosmos pack/encode/attention/decode with V0.3's single-pass adapter."""

from types import SimpleNamespace

import pytest
import torch

from test_ar_v02_network_gpu import make_network
from test_ar_v03_model import model_fixture,data_fixture
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION,VIDEO,JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize('prefix_low_noise_enabled', [False, True])
def test_official_training_step_runs_one_forward_and_real_backward(prefix_low_noise_enabled):
    layouts=[JointChunkLayout(10,1,4),JointChunkLayout(6,1,4)]
    model=model_fixture(layouts,"cuda")
    model.config.prefix_low_noise_enabled=prefix_low_noise_enabled
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


@pytest.mark.parametrize('prefix_low_noise_enabled', [False, True])
def test_noisy_chunk_receives_self_and_later_loss_gradients_through_full_network(prefix_low_noise_enabled):
    from cosmos3_joint_video_hand_pose.src.ar_v02_layout import CONDITION_VIDEO, STATE

    torch.manual_seed(33)
    layout=JointChunkLayout(10,1,4)
    model=model_fixture([layout],"cuda")
    model.config.prefix_low_noise_enabled=prefix_low_noise_enabled
    model.net=make_network()
    data=data_fixture([layout],"cuda")
    timesteps,sigmas=model._get_train_noise_level_vision(
        1,False,[layout.num_video_frames],["480"],iteration=3)
    model._get_train_noise_level_action(1,iteration=3)
    packed=pack_joint_sequence(layout=layout,gen_data_clean=data,text_ids=[3,4],
        special_tokens=model.llm_special_tokens,timesteps=timesteps[0],latent_patch_size=2)
    noised=model._add_noise_to_input(data,packed,sigmas,iteration=3)
    vr,vc,_=layout.video_metadata(device="cuda")
    ar,ac,_=layout.action_metadata(device="cuda")

    # Use the actual routed/noised V/A as differentiable inputs, not clean x0.
    # Targets remain fixed epsilon-x0; otherwise a target path could masquerade
    # as an attention-history gradient or leak a future block into the loss.
    video=noised.xt_tokens_vision[0].detach().requires_grad_()
    action=noised.xt_tokens_action[0].detach().requires_grad_()
    noised.xt_tokens_vision[0]=video
    noised.xt_tokens_action[0]=action
    target_video=noised.vt_target_vision[0].detach()
    target_action=noised.vt_target_action[0].detach()
    torch.testing.assert_close(target_video,
        noised.epsilon_vision[0]-data.x0_tokens_vision[0],atol=0,rtol=0)
    torch.testing.assert_close(target_action,
        noised.epsilon_action[0]-data.x0_tokens_action[0],atol=0,rtol=0)
    torch.testing.assert_close(video[:,:,vr==CONDITION_VIDEO],
        data.x0_tokens_vision[0][:,:,vr==CONDITION_VIDEO],atol=0,rtol=0)
    torch.testing.assert_close(action[ar==STATE],data.x0_tokens_action[0][ar==STATE],atol=0,rtol=0)
    assert torch.count_nonzero(sigmas[0,vr==CONDITION_VIDEO])==0
    assert torch.count_nonzero(noised.sigmas_action[0][ar==STATE])==0
    assert torch.count_nonzero(action[:,57:])==0
    assert not torch.equal(video[:,:,vr==VIDEO],data.x0_tokens_vision[0][:,:,vr==VIDEO])
    assert not torch.equal(action[ar==ACTION,:57],data.x0_tokens_action[0][ar==ACTION,:57])
    for chunk in vc.unique():
        video_rows=(vr==VIDEO)&(vc==chunk)
        action_rows=(ar==ACTION)&(ac==chunk)
        assert sigmas[0,video_rows].unique().numel()==1
        torch.testing.assert_close(noised.sigmas_action[0][action_rows].flatten(),
            model._ar_step.action_sigmas[0,chunk].expand(int(action_rows.sum())),atol=0,rtol=0)
    torch.testing.assert_close(packed.vision.timesteps.to("cuda"),timesteps[0,vr==VIDEO],atol=0,rtol=0)
    torch.testing.assert_close(packed.action.timesteps.to("cuda"),
        model._ar_step.action_timesteps[0][ar==ACTION],atol=0,rtol=0)
    model._replace_clean_with_noised(packed,noised)
    packed.to_cuda()
    assert packed.vision.tokens[0] is video and packed.action.tokens[0] is action
    weights=action.new_ones(57)
    weights[9:18]=weights[33:42]=3
    torch.testing.assert_close(weights,action.new_tensor(model.config.action_channel_weights),atol=0,rtol=0)
    count=[]
    handle=model.net.register_forward_pre_hook(lambda *args: count.append(1))

    def gradients(chunks):
        # Official compiled FlexAttention donates backward buffers. Recompute
        # the same deterministic forward instead of changing production flags.
        memory=model.build_memory_state(packed,{})
        out=model.denoise(data_batch_packed=packed,memory=memory)
        losses=[]
        for chunk in chunks:
            video_rows=(vr==VIDEO)&(vc==chunk)
            action_rows=(ar==ACTION)&(ac==chunk)
            video_error=(out["preds_vision"][0][:,:,video_rows]-target_video[:,:,video_rows]).square()
            action_error=(out["preds_action"][0][action_rows,:57]-target_action[action_rows,:57]).square()
            # Independent formula for V0.3's weighted 57D flow objective;
            # every selected future row is kept, U/S and padding are excluded.
            action_loss=(action_error*weights).sum()/(action_error.shape[0]*weights.sum())
            losses.append(video_error.mean()+action_loss)
        loss=sum(losses)
        assert torch.isfinite(loss)
        result=torch.autograd.grad(loss,(video,action))
        assert all(x is None for x in memory._clean_gen_kv)
        assert all(x is None for x in memory._clean_und_kv)
        return result
    grads_own=gradients([1])
    grads_later=gradients([2])
    grads_total=gradients([1,2])
    handle.remove()
    assert len(count)==3
    for name,a,b,total in zip(("video","action"),grads_own,grads_later,grads_total):
        region=((vr==VIDEO)&(vc==1)) if name=="video" else ((ar==ACTION)&(ac==1))
        aa=a[:,:,region] if name=="video" else a[region,:57]
        bb=b[:,:,region] if name=="video" else b[region,:57]
        assert torch.isfinite(aa).all() and aa.norm()>0
        assert torch.isfinite(bb).all() and bb.norm()>0
        assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(total).all()
        future=(vc>1) if name=="video" else (ac>1)
        assert torch.count_nonzero(a[:,:,future] if name=="video" else a[future])==0
        future_later=(vc>2) if name=="video" else (ac>2)
        assert torch.count_nonzero(b[:,:,future_later] if name=="video" else b[future_later])==0
        torch.testing.assert_close(total,a+b,atol=1e-5,rtol=1e-4)
        print(f"V03_NETWORK_NOISED_FLOW_GRAD prefix={prefix_low_noise_enabled} {name} own={float(aa.norm())} later={float(bb.norm())}")
