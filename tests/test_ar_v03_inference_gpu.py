"""Real official small network/cache integration for V0.3 joint refresh."""

from types import SimpleNamespace

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_v02_cache import JointKVCache
from cosmos3_joint_video_hand_pose.src.ar_v02_inference import JointARSampler
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, VIDEO, JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v03_inference import DiffusionForcingJointARSampler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocated CUDA GPU required")


@pytest.mark.parametrize("c", [1, 4])
@torch.no_grad()
def test_official_network_zero_equivalence_and_refresh_affects_only_later_blocks(c):
    from test_ar_v02_network_gpu import make_network, make_pack

    torch.manual_seed(42)
    net = make_network().eval()
    layout = JointChunkLayout(c + 2, 1, c)
    full = make_pack([layout])
    video, action = full.vision.tokens[0], full.action.tokens[0]
    original_video, original_action = video.clone(), action.clone()
    roles, chunks, _ = layout.metadata()

    def run(sampler_type, sigma):
        sampler = object.__new__(sampler_type)
        sampler.layout, sampler._cache_template = layout, full
        sampler.sigma_small, sampler.history_video_sigma = sigma, 0.
        sampler._history_noise_seed, sampler._cache_phase = 12345, None
        sampler.cache = JointKVCache(layout, num_layers=2, num_kv_heads=4, head_dim=16,
                                     device="cuda", dtype=torch.float32)
        video_flow = SimpleNamespace(noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)))
        action_flow = SimpleNamespace(noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=2000)))
        sampler.model = SimpleNamespace(rectified_flow_video=video_flow, rectified_flow_action=action_flow,
                                        _cast_generated_tokens_to_precision=lambda packed: None,
                                        denoise=lambda *, data_batch_packed, memory: net(data_batch_packed, memory=memory))
        sampler._cache_forward(video, action, [], chunk=0, phase="text")
        outputs, refreshed, conditions = [], [], []
        for b in layout.boundaries:
            sampler._cache_forward(video, action, layout.condition_prefill_indexes(b.chunk_id),
                                   chunk=b.chunk_id, phase="condition")
            conditions.append([(k.clone(), v.clone()) for k, v in sampler.cache.kv])
            target = torch.where((chunks == b.chunk_id) & ((roles == ACTION) | (roles == VIDEO)))[0]
            saved = [(k.clone(), v.clone()) for k, v in sampler.cache.kv]
            outputs.append(sampler._cache_forward(video, action, target, chunk=b.chunk_id, phase="noisy",
                                                   video_sigma=.4, action_sigma=.6))
            assert all(torch.equal(x, y) for a, b in zip(saved, sampler.cache.kv) for x, y in zip(a, b))
            sampler._cache_forward(video, action, target, chunk=b.chunk_id, phase="refresh")
            refreshed.append([(k.clone(), v.clone()) for k, v in sampler.cache.kv])
        return outputs, refreshed, conditions

    old, old_kv, old_condition = run(JointARSampler, 0)
    zero, zero_kv, zero_condition = run(DiffusionForcingJointARSampler, 0)
    positive, positive_kv, positive_condition = run(DiffusionForcingJointARSampler, .02)
    for left, right in zip(old, zero):
        for name in ("preds_vision", "preds_action"):
            assert torch.equal(left[name][0], right[name][0])
    assert all(torch.equal(x, y) for a, b in zip(old_kv, zero_kv) for p, q in zip(a, b) for x, y in zip(p, q))
    assert all(torch.equal(x, y) for a, b in zip(old_condition, zero_condition) for p, q in zip(a, b) for x, y in zip(p, q))
    for name in ("preds_vision", "preds_action"):
        assert torch.equal(zero[0][name][0], positive[0][name][0])
        assert not torch.equal(zero[1][name][0], positive[1][name][0])
        assert torch.isfinite(positive[1][name][0]).all()
    assert any(not torch.equal(x, y) for a, b in zip(zero_kv[0], positive_kv[0]) for x, y in zip(a, b))
    assert all(torch.equal(x, y) for a, b in zip(zero_condition[0], positive_condition[0]) for x, y in zip(a, b))
    assert torch.equal(video, original_video) and torch.equal(action, original_action)
    assert not action[:, 57:].count_nonzero()
