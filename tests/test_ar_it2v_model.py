"""Exercise real Cosmos noising/loss extension hooks with tiny CPU tensors."""
from types import SimpleNamespace as NS
import pytest
import torch
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos3_ar_it2v.model import ARIT2VModel, sample_chunk_sigmas


class Flow:
    noise_scheduler=NS(config=NS(num_train_timesteps=1000))
    def get_interpolation(self,eps,clean,sigmas):
        return ([(1-s)*x+s*e for x,e,s in zip(clean,eps,sigmas)],[e-x for e,x in zip(eps,clean)])
    def train_time_weight(self,ts,kwargs):
        return torch.ones_like(ts).to(**kwargs)


def model():
    m=object.__new__(ARIT2VModel)
    torch.nn.Module.__init__(m)
    m.config=NS(action_gen=False,sound_gen=False,lidar_gen=False,lidar_state_ch=None,vision_gen=True,causal_training_strategy='diffusion_forcing',frames_per_chunk=4,local_attention_frames=16,
                sigma_min=.02,sigma_max=.98,sigma_shift=5.,noise_level_type='sigma',
                rectified_flow_training_config=NS(independent_sound_schedule=False,normalize_loss_by_active=True,loss_scale=1.,image_loss_scale=None,sample_level_loss_averaging=True))
    m.tensor_kwargs_fp32=dict(device='cpu',dtype=torch.float32)
    m.tensor_kwargs=m.tensor_kwargs_fp32
    m.parallel_dims=None
    m.rectified_flow_video=Flow()
    return m


def test_sigma_chunks_independent_and_padding_zero():
    a=sample_chunk_sigmas([5,13],generator=torch.Generator().manual_seed(42))
    b=sample_chunk_sigmas([5,13],generator=torch.Generator().manual_seed(42))
    assert torch.equal(a,b) and (a[:,0]==0).all() and (a[0,5:]==0).all()
    for row,t in zip(a,[5,13]):
        for start in range(1,t,4):
            assert (row[start:start+4]==row[start]).all()
        assert (row[1:t]>=.02).all() and (row[1:t]<=.98).all()
    draws=torch.rand(4,generator=torch.Generator().manual_seed(42))
    expected=(5*draws/(1+4*draws)).clamp(.02,.98)
    assert torch.equal(torch.stack([a[0,1],a[1,1],a[1,5],a[1,9]]),expected)


def test_real_timestep_routing():
    m=model()
    ts,s=m._get_train_noise_level_vision(2,False,[5,9])
    assert torch.equal(ts,s*1000) and (ts[:,0]==0).all()
    assert (ts[1,1:5]==ts[1,1]).all()
    assert ts[1,1]!=ts[1,5]


def test_real_official_noising_preserves_first_frame():
    m=model()
    clean=torch.randn(1,2,9,2,2)
    data=GenerationDataClean(batch_size=1,is_image_batch=False,x0_tokens_vision=[clean],
                             fps_vision=torch.tensor([15.]))
    mask=torch.zeros(9,1,1);mask[0]=1
    packed=NS(vision=NS(condition_mask=[mask]),action=None,sound=None,lidar=None)
    s=sample_chunk_sigmas([9])
    out=m._add_noise_to_input(data,packed,s)
    assert torch.equal(out.xt_tokens_vision[0][:,:,0],clean[:,:,0])
    assert torch.equal(out.sigmas_vision[0].flatten(),s[0])
    assert not torch.equal(out.xt_tokens_vision[0][:,:,1:],clean[:,:,1:])
    assert out.xt_tokens_action is None


def test_real_flow_loss_supervises_all_future_and_not_first():
    m=model()
    p=torch.ones(2,9,2,2,requires_grad=True)
    target=torch.zeros_like(p)
    cond=torch.zeros(9,1,1);cond[0]=1
    pack=NS(vision=NS(tokens=[p],mse_loss_indexes=torch.arange(1,9),condition_mask=[cond],noisy_frame_indexes=[torch.arange(1,9)]),action=None,sound=None,lidar=None)
    noised=NS(vt_target_vision=[target])
    loss,_=m._compute_losses(out_net={'preds_vision':[p]},data_batch_packed=pack,gen_data_noised=noised,
        timesteps=sample_chunk_sigmas([9])*1000,is_image_batch=False)
    assert loss.item()==1
    loss.backward()
    assert p.grad[:,0].abs().sum()==0
    assert (p.grad[:,1:]!=0).all()


def test_no_extra_condition_frames_or_action_allowed():
    m=model()
    mask=torch.zeros(5,1,1);mask[0]=1
    pack=NS(vision=NS(condition_mask=[mask]),action=None,sound=None,lidar=None)
    assert m.pre_noise_memory_hook(pack,None,{})=={}
    mask[1]=1
    with pytest.raises(ValueError,match='exactly latent frame zero'):
        m.pre_noise_memory_hook(pack,None,{})
    pack.action=object()
    with pytest.raises(ValueError,match='non-video'):
        m.pre_noise_memory_hook(pack,None,{})
