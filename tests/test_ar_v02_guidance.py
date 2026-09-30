import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_v02_guidance import video_guidance_output
from cosmos3_joint_video_hand_pose.src.ar_inference import flow_sigmas
from cosmos3_joint_video_hand_pose.src.ar_v02_eval import parser


def test_video_only_guidance_preserves_conditional_action_object():
    action = [torch.tensor([3., 7.])]
    cond = dict(preds_vision=[torch.tensor([2., 4.])], preds_action=action)
    uncond = dict(preds_vision=[torch.tensor([1., 2.])], preds_action=[torch.tensor([-100., -100.])])
    for scale in (3., 6.):
        result = video_guidance_output(cond, uncond, scale)
        torch.testing.assert_close(result['preds_vision'][0], torch.tensor([1., 2.]) * (scale + 1))
        assert result['preds_action'] is action
    assert video_guidance_output(cond, None, 1.) is cond
    torch.testing.assert_close(cond['preds_vision'][0], torch.tensor([2., 4.]))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0., -1.])
def test_invalid_guidance_rejected(value):
    with pytest.raises(ValueError):
        video_guidance_output({}, {}, value)


def test_video_shift_does_not_change_default_action_schedule():
    original = flow_sigmas(30, 5)
    args = parser().parse_args(['sample', '--ckpt', 'model', '--episodes-manifest', 'e',
        '--segments-manifest', 's', '--eval-windows', 'w', '--output', 'o', '--split', 'heldout'])
    assert args.video_shift == 5 and args.video_guidance == 1
    assert len(flow_sigmas(30, 10)) == 31
    assert not torch.equal(original, flow_sigmas(30, 10))
    torch.testing.assert_close(original, flow_sigmas(30, 5), atol=0, rtol=0)


def test_cfg_uses_isolated_text_and_history_caches(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from cosmos3_joint_video_hand_pose.src import ar_v02_guidance as mod
    sampler = object.__new__(mod.VideoGuidedJointARSampler)
    sampler.layout = SimpleNamespace(num_video_frames=5)
    sampler.chunk_size = 4
    sampler.text = [[10, 11, 12]]
    sampler.negative_text_ids = [99]
    sampler.video_guidance = 3.
    sampler.memory_info = {'initial_temporal_offset': 0}
    sampler.plans, sampler.gen = [], object()
    sampler.cache = SimpleNamespace(events=[])
    sampler._cache_template = object()
    sampler._negative = None
    sampler.model = SimpleNamespace(net=SimpleNamespace(num_hidden_layers=1,num_kv_heads=1,head_dim=1),
        tensor_kwargs={'dtype':torch.float32},ar_context=lambda *a:nullcontext(),
        _pack_input_sequence=lambda plans,text,*a,**k:tuple(text[0]))
    monkeypatch.setattr(mod,'JointKVCache',lambda *a,**k:SimpleNamespace(events=[]))
    calls=[]
    def forward(self,video,action,indexes,*,chunk,phase,**kw):
        self.cache.events.append((chunk,phase))
        calls.append((id(self.cache),tuple(self.text[0]),id(video),id(action),phase))
        v=1. if self.text[0]==[99] else 2.
        return {'preds_vision':[torch.tensor([v])],'preds_action':[action]}
    monkeypatch.setattr(mod.JointARSampler,'_cache_forward',forward)
    video,action=torch.zeros(1),torch.ones(1)
    for phase in ('text','condition','noisy','refresh'):
        result=sampler._cache_forward(video,action,[],chunk=1,phase=phase)
        if phase=='noisy':
            assert result['preds_vision'][0].item()==4.
            assert result['preds_action'][0] is action
    assert sampler.cache is not sampler._negative.cache
    assert sampler.cache.events==sampler._negative.cache.events
    assert sampler.text==[[10,11,12]] and sampler._negative.text==[[99]]
    assert len(calls)==8 and all(c[2:4]==(id(video),id(action)) for c in calls)
