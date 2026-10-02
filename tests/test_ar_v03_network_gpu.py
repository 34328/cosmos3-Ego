"""Real Cosmos pack/encode/attention/decode with V0.3's single-pass adapter."""

from types import SimpleNamespace

import pytest
import torch

from test_ar_v02_network_gpu import make_network
from test_ar_v03_model import model_fixture,data_fixture
from test_ar_v031_model import prefix_iteration
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION,VIDEO,JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence
from cosmos3_joint_video_hand_pose.src.ar_v03_model import EgoVerseARV03Model
from cosmos3_joint_video_hand_pose.src.ar_v031_model import EgoVerseARV031Model

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


@pytest.mark.parametrize("model_class,prefix_low_noise_enabled,mask_prefix_loss", [
    pytest.param(EgoVerseARV03Model, False, False, id="v03-original-sigmas"),
    pytest.param(EgoVerseARV03Model, True, False, id="v03-prefix-sigmas"),
    pytest.param(EgoVerseARV031Model, True, False, id="v031-mask-off"),
    pytest.param(EgoVerseARV031Model, True, True, id="v031-numerator-only"),
])
def test_noisy_chunk_receives_self_and_later_loss_gradients_through_full_network(
    model_class, prefix_low_noise_enabled, mask_prefix_loss,
):
    from cosmos3_joint_video_hand_pose.src.ar_v02_layout import CONDITION_VIDEO, STATE

    torch.manual_seed(33)
    layout=JointChunkLayout(10,1,4)
    model=model_fixture([layout],"cuda",cls=model_class)
    model.config.causal_training_strategy="diffusion_forcing"
    model.config.prefix_low_noise_enabled=prefix_low_noise_enabled
    model.config.mask_prefix_loss=mask_prefix_loss
    model.net=make_network()
    data=data_fixture([layout],"cuda")
    # Choose an actual stateless L=2 draw: block 1 is prefix, block 2 is
    # supervised suffix. The mask-on assertion cannot pass vacuously at L=1.
    iteration=prefix_iteration([layout])
    before_draw=torch.get_rng_state()
    control=model_fixture([layout],"cuda")
    old_times,old_sigmas=control._get_train_noise_level_vision(
        1,False,[layout.num_video_frames],["480"],iteration=iteration)
    control._get_train_noise_level_action(1,iteration=iteration)
    old_action_sigmas=control._ar_step.action_sigmas.clone()
    after_original_draw=torch.get_rng_state()
    torch.set_rng_state(before_draw)
    timesteps,sigmas=model._get_train_noise_level_vision(
        1,False,[layout.num_video_frames],["480"],iteration=iteration)
    model._get_train_noise_level_action(1,iteration=iteration)
    assert torch.equal(torch.get_rng_state(),after_original_draw)
    if prefix_low_noise_enabled:
        plan=model._ar_step.prefix_low_noise_plan
        assert plan.metadata[0]["prefix_length"]==2 and plan.mask[0,1] and not plan.mask[0,2]
        assert 0 <= model._ar_step.video_chunk_sigmas[0,1] < .1
        assert torch.equal(model._ar_step.video_chunk_sigmas[plan.mask],
                           model._ar_step.action_sigmas[plan.mask])
        assert torch.equal(model._ar_step.action_sigmas[~plan.mask],old_action_sigmas[~plan.mask])
        vr,vc,_=layout.video_metadata()
        suffix=(vr==VIDEO)&(vc>=2)
        assert torch.equal(sigmas[0,suffix],old_sigmas[0,suffix])
        assert torch.equal(timesteps[0,suffix],old_times[0,suffix])
    else:
        assert model._ar_step.prefix_low_noise_plan is None
        assert torch.equal(sigmas,old_sigmas) and torch.equal(timesteps,old_times)
        assert torch.equal(model._ar_step.action_sigmas,old_action_sigmas)
    packed=pack_joint_sequence(layout=layout,gen_data_clean=data,text_ids=[3,4],
        special_tokens=model.llm_special_tokens,timesteps=timesteps[0],latent_patch_size=2)
    noised=model._add_noise_to_input(data,packed,sigmas,iteration=iteration)
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
    noised.vt_target_vision[0]=target_video
    noised.vt_target_action[0]=target_action
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
        # Select ONLY contribution numerators. Keeping the original pack
        # preserves full-clip coordinate denominators and active sample counts.
        selected=dict(out)
        vselect=(vr==VIDEO)&torch.isin(vc,torch.tensor(chunks,device=vc.device))
        aselect=(ar==ACTION)&torch.isin(ac,torch.tensor(chunks,device=ac.device))
        prediction=out["preds_vision"][0]
        selected["preds_vision"]=[torch.where(
            vselect.reshape(*([1]*(prediction.ndim-3)),-1,1,1),prediction,target_video)]
        selected["preds_action"]=[torch.where(aselect[:,None],out["preds_action"][0],target_action)]
        loss,stats=model._compute_whole_losses(selected,packed,noised,timesteps,False)
        assert torch.isfinite(loss)
        assert stats["egoverse_global_video_samples"]==stats["egoverse_global_action_samples"]==1
        assert len(model._last_visibility_loss_metrics)==8
        result=torch.autograd.grad(loss,(video,action,out["preds_vision"][0],out["preds_action"][0]))
        prefix_v=(vr==VIDEO)&(vc==1)
        prefix_a=(ar==ACTION)&(ac==1)
        output_v=result[2][:,:,prefix_v]
        output_a=result[3][prefix_a,:57]
        if mask_prefix_loss or 1 not in chunks:
            assert torch.count_nonzero(output_v)==torch.count_nonzero(output_a)==0
        else:
            assert output_v.norm()>0 and output_a.norm()>0
        assert all(x is None for x in memory._clean_gen_kv)
        assert all(x is None for x in memory._clean_und_kv)
        return result[:2]
    grads_own=gradients([1])
    grads_later=gradients([2])
    grads_total=gradients([1,2])
    handle.remove()
    assert len(count)==3
    for name,a,b,total in zip(("video","action"),grads_own,grads_later,grads_total):
        region=((vr==VIDEO)&(vc==1)) if name=="video" else ((ar==ACTION)&(ac==1))
        aa=a[:,:,region] if name=="video" else a[region,:57]
        bb=b[:,:,region] if name=="video" else b[region,:57]
        assert torch.isfinite(aa).all()
        if mask_prefix_loss:
            assert torch.count_nonzero(aa)==0
        else:
            assert aa.norm()>0
        assert torch.isfinite(bb).all() and bb.norm()>0
        assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(total).all()
        future=(vc>1) if name=="video" else (ac>1)
        assert torch.count_nonzero(a[:,:,future] if name=="video" else a[future])==0
        future_later=(vc>2) if name=="video" else (ac>2)
        assert torch.count_nonzero(b[:,:,future_later] if name=="video" else b[future_later])==0
        torch.testing.assert_close(total,a+b,atol=1e-5,rtol=1e-4)
        print(f"AR_NETWORK_NATIVE_FLOW_GRAD model={model_class.__name__} "
              f"prefix={prefix_low_noise_enabled} mask={mask_prefix_loss} "
              f"{name} own={float(aa.norm())} later={float(bb.norm())}")
