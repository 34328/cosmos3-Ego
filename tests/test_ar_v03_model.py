"""V0.3 routing uses the actual official noise/packing path on CPU."""

from types import SimpleNamespace

import torch

from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos3_joint_video_hand_pose.src.ar_model import ARStepContext
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, CONDITION_VIDEO, STATE, VIDEO, JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_model import EgoVerseARV02Model
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence
from cosmos3_joint_video_hand_pose.src.ar_v03_model import EgoVerseARV03Model, wrist_channel_weights


class FlowFixture:
    """Linear flow fixture, while invoking the real official input-noise code."""

    def __init__(self):
        self.noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.draws = []

    def sample_train_time(self, count, *, iteration, shifts):
        self.draws.append((count, shifts.clone()))
        return torch.rand(count)

    def get_interpolation(self, epsilon, clean, sigmas):
        return ([(1-s)*x+s*e for x,e,s in zip(clean,epsilon,sigmas)],
                [e-x for x,e in zip(clean,epsilon)])

    def train_time_weight(self, ts, kwargs):
        return torch.ones_like(ts).to(**kwargs)


def model_fixture(layouts, device="cpu", cls=EgoVerseARV03Model):
    model = object.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        causal_training_strategy="diffusion_forcing" if cls is EgoVerseARV03Model else "teacher_forcing",
        action_gen=True, vision_gen=True, sound_gen=False, lidar_gen=False, lbl=None, enable_moba=False,
        clamp_empty_varlen_kv=False, action_channel_weights=wrist_channel_weights(), correct_cp_gradients=True,
        prefix_low_noise_enabled=False, sigma_hist_max=0.1,
        rectified_flow_training_config=SimpleNamespace(
            use_discrete_rf=False, shift={"480":5}, shift_action=5,
            independent_action_schedule=True, sample_level_loss_averaging=True,
            loss_scale=1.0, action_loss_weight=1.0, image_loss_scale=None,
        ),
        diffusion_expert_config=SimpleNamespace(
            patch_spatial=2, base_fps=24, unified_3d_mrope_reset_spatial_ids=True,
            unified_3d_mrope_temporal_modality_margin=0,
        ),
    )
    model._joint_layouts = layouts
    model._joint_layout = layouts[0] if len(layouts)==1 else None
    model._ar_step = ARStepContext(4,15)
    model.ar_seed = 42
    model.action_representation = "fixed_camera_wrist_local_delta_latent_v1"
    model.rectified_flow_video = FlowFixture()
    model.rectified_flow_action = FlowFixture()
    model.tensor_kwargs_fp32 = dict(device=device,dtype=torch.float32)
    model.tensor_kwargs = dict(device=device,dtype=torch.float32)
    model.precision = torch.float32
    model.parallel_dims = None
    model.llm_special_tokens = {"eos_token_id":1,"start_of_generation":2}
    model.whole_action_loss = True
    model._ar_loss_window = None
    model._ar_gradient_accumulation = 1
    model._ar_backward_loss = None
    model._current_hand_visibility = [torch.ones(x.num_action_rows,2,dtype=torch.bool) for x in layouts]
    model._last_visibility_loss_metrics = {}
    model._bidirectional_step_active = False
    return model


def data_fixture(layouts, device="cpu"):
    return GenerationDataClean(
        batch_size=len(layouts), is_image_batch=False,
        x0_tokens_vision=[torch.randn(1,4,x.num_video_frames,2,2,device=device) for x in layouts],
        x0_tokens_action=[torch.cat((torch.randn(x.num_action_rows,57,device=device),
                                    torch.zeros(x.num_action_rows,7,device=device)),1) for x in layouts],
        fps_vision=torch.tensor([7.5]*len(layouts)), fps_action=torch.tensor([15.]*len(layouts)),
        raw_action_dim=[torch.tensor(57) for _ in layouts],
    )


def test_chunk_video_draws_match_v02_rng_distribution_and_conditions():
    layouts = [JointChunkLayout(10,1,4),JointChunkLayout(6,1,4)]
    counts = [x.num_video_frames for x in layouts]
    outputs=[]
    for cls in (EgoVerseARV02Model,EgoVerseARV03Model):
        model=model_fixture(layouts,cls=cls)
        torch.manual_seed(19)
        ts,sg=model._get_train_noise_level_vision(2,False,counts,["480"]*2,[160,96],iteration=1)
        outputs.append((ts,sg))
        assert model.rectified_flow_video.draws[0][0] == 8
        for i,layout in enumerate(layouts):
            roles,chunks,_=layout.video_metadata()
            assert torch.count_nonzero(sg[i,:len(roles)][roles==CONDITION_VIDEO])==0
            assert torch.count_nonzero(ts[i,:len(roles)][roles==CONDITION_VIDEO])==0
            for chunk in chunks.unique():
                values=sg[i,:len(roles)][(roles==VIDEO)&(chunks==chunk)]
                assert values.unique().numel()==1
            assert torch.count_nonzero(sg[i,len(roles):])==0
    for old,new in zip(*outputs):
        torch.testing.assert_close(old,new,atol=0,rtol=0)


def test_real_noising_keeps_us_clean_and_routes_both_modalities_per_chunk():
    layouts=[JointChunkLayout(10,1,4),JointChunkLayout(6,1,4)]
    model=model_fixture(layouts)
    data=data_fixture(layouts)
    ts,sg=model._get_train_noise_level_vision(2,False,[x.num_video_frames for x in layouts],["480"]*2,iteration=2)
    model._get_train_noise_level_action(2,iteration=2)
    packed=pack_joint_sequence(layout=layouts,gen_data_clean=data,text_ids=[[3],[4,5]],
        special_tokens=model.llm_special_tokens,timesteps=ts,latent_patch_size=2,condition_frames=[(),()])
    noised=model._add_noise_to_input(data,packed,sg,iteration=2)
    expected_action_times=[]
    for i,layout in enumerate(layouts):
        vr,vc,_=layout.video_metadata()
        ar,ac,_=layout.action_metadata()
        torch.testing.assert_close(noised.xt_tokens_vision[i][:,:,vr==CONDITION_VIDEO],
                                   data.x0_tokens_vision[i][:,:,vr==CONDITION_VIDEO],atol=0,rtol=0)
        torch.testing.assert_close(noised.xt_tokens_action[i][ar==STATE],
                                   data.x0_tokens_action[i][ar==STATE],atol=0,rtol=0)
        assert torch.count_nonzero(noised.xt_tokens_action[i][:,57:])==0
        assert torch.count_nonzero(noised.sigmas_action[i][ar==STATE])==0
        for chunk in ac.unique():
            torch.testing.assert_close(noised.sigmas_action[i][(ar==ACTION)&(ac==chunk)].flatten(),
                model._ar_step.action_sigmas[i,chunk].expand(int(((ar==ACTION)&(ac==chunk)).sum())),atol=0,rtol=0)
        expected_action_times.append(model._ar_step.action_timesteps[i][ar==ACTION])
        vtime=packed.vision.timesteps[sum(int((x.video_metadata()[0]==VIDEO).sum()) for x in layouts[:i]):]
        torch.testing.assert_close(vtime[:int((vr==VIDEO).sum())],ts[i,:len(vr)][vr==VIDEO],atol=0,rtol=0)
    torch.testing.assert_close(packed.action.timesteps,torch.cat(expected_action_times),atol=0,rtol=0)
    assert not packed.uses_single_timestep


def test_pre_noise_hook_has_no_forward_or_clean_replay():
    layout=JointChunkLayout(9,1,4)
    model=model_fixture([layout])
    data=data_fixture([layout])
    packed=pack_joint_sequence(layout=layout,gen_data_clean=data,text_ids=[3],
        special_tokens=model.llm_special_tokens,timesteps=500,latent_patch_size=2)
    def forbidden(*args,**kwargs):
        raise AssertionError("a clean denoiser pass ran")
    model.denoise=forbidden
    model._build_clean_tf_cache=forbidden
    info={"skip_text":False,"initial_temporal_offset":0}
    assert model.pre_noise_memory_hook(packed,data,info) is info
