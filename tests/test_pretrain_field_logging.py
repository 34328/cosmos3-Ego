"""Field diagnostics must not change the objective, gradients, masks or RNG."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from cosmos3_joint_video_hand_pose.src.loss import whole_action_flow_loss, ACTION_SUBBLOCKS
from cosmos3_joint_video_hand_pose.src import wandb_metrics as wm
from cosmos3_joint_video_hand_pose.src.pretrain_probe import payload_digest

def test_field_diagnostics_leave_loss_and_gradient_unchanged():
    torch.manual_seed(42)
    p = [torch.randn(n,64,requires_grad=True) for n in (34,10)]
    kwargs = dict(pred=p,target=[torch.randn_like(x) for x in p],
                  condition_mask=[torch.arange(len(x))%9==0 for x in p],
                  visibility=[torch.zeros(len(x),2) for x in p],mask_out_of_fov=False)
    a,_ = whole_action_flow_loss(**kwargs)
    rng = torch.get_rng_state().clone()
    b,stats = whole_action_flow_loss(**kwargs,collect_field_metrics=True)
    assert torch.equal(a,b)
    assert torch.equal(rng,torch.get_rng_state())
    ga=torch.autograd.grad(a,p); gb=torch.autograd.grad(b,p)
    for x,y in zip(ga,gb): assert torch.equal(x,y)
    weighted=0
    for name,(sl,_) in ACTION_SUBBLOCKS.items():
        values=stats["field_per_sample_losses"][name]
        assert not values.requires_grad
        expected=[]
        for pred,target,condition in zip(p,kwargs["target"],kwargs["condition_mask"]):
            expected.append((pred.detach()[~condition,sl]-target[~condition,sl]).square().mean())
        torch.testing.assert_close(values,torch.stack(expected))
        weighted=weighted+values*(sl.stop-sl.start)/57
    torch.testing.assert_close(weighted,stats["per_sample_losses"])

def test_all_field_metrics_reach_jsonl_and_wandb(tmp_path,monkeypatch):
    cb=wm.EgoVerseLossWandbCallback()
    cb.config=SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)),trainer=SimpleNamespace(logging_iter=1))
    monkeypatch.setattr(wm.distributed,"is_rank0",lambda:True)
    monkeypatch.setattr(wm.wandb,"run",object())
    log=Mock(); monkeypatch.setattr(wm.wandb,"log",log)
    outputs={source:torch.tensor(float(i+1)) for i,source in enumerate(wm.LOSS_METRIC_SOURCES.values())}
    cb.on_training_step_end(None,{},outputs,torch.tensor(0.),1)
    row=json.loads((tmp_path/"loss_metrics.jsonl").read_text())
    for name,source in wm.LOSS_METRIC_SOURCES.items():
        assert row[name]==outputs[source].item()==log.call_args.args[0][name]

def test_data_digest_detects_changed_source_text_or_action():
    value={"source":torch.arange(3),"text":[torch.tensor([4,5])],"action":torch.zeros(2,64)}
    rng=torch.get_rng_state().clone()
    before=payload_digest(value)
    assert before==payload_digest(value)
    assert torch.equal(rng,torch.get_rng_state())
    for key in value:
        changed={k:v for k,v in value.items()}
        changed[key]=torch.tensor([100])
        assert before!=payload_digest(changed)

def test_v02_c4_validation_preserves_joint_latents_and_tail(monkeypatch):
    import pytest
    from cosmos3_joint_video_hand_pose.src.ar_model import EgoVerseARModel
    from cosmos3_joint_video_hand_pose.src.ar_v02_model import EgoVerseARV02Model
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
    for cls, required in ((EgoVerseARModel,1),(EgoVerseARV02Model,4)):
        model=object.__new__(cls);torch.nn.Module.__init__(model)
        model.config=SimpleNamespace(video_temporal_causal=True,causal_training_strategy='teacher_forcing',
            teacher_forcing_frames_per_chunk=required,supervise_temporal_causal_actions=True,
            action_tokens_per_latent=8,enable_moba=False,teacher_forcing_target_only_no_text_pass2=False,
            parallelism=SimpleNamespace(context_parallel_shard_degree=1))
        monkeypatch.setattr(model,'_get_teacher_forcing_kv_implementation',lambda:'singleview_threeway_kv')
        model._validate_ar_config()
        model.config.teacher_forcing_frames_per_chunk=5-required
        with pytest.raises(ValueError,match='teacher_forcing_frames_per_chunk'):model._validate_ar_config()
        model.config.teacher_forcing_frames_per_chunk=required
        if required==4:
            for latent_frames,actions in ((40,256),(80,512),(85,544),(7,40)):
                video=torch.zeros(1,4,latent_frames,2,2); action=torch.zeros(actions,64)
                data=GenerationDataClean(batch_size=1,is_image_batch=False,
                    x0_tokens_vision=[video],x0_tokens_action=[action])
                assert model._truncate_for_chunkwise_tf(data) is data
                model._assert_chunkwise_tf_shape(data)
                assert data.x0_tokens_vision[0] is video and data.x0_tokens_action[0] is action
