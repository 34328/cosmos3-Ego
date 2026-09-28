import copy
import pytest
import torch
from test_ar_v02_network_gpu import make_network, make_pack, adapter_memory
from cosmos_framework.model.generator.teacher_forcing import make_teacher_forcing_clean_pack
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_compact import (
    compact_joint_targets,
    restore_joint_predictions,
    CompactJointMemory,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def native_network():
    from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
    from cosmos_framework.model.generator.mot import causal_attention

    cfg = Qwen3VLMoTConfig(
        config_dict={
            "text_config": dict(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=16,
                rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
            )
        },
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
    )
    cfg.use_und_k_norm_for_gen = True
    lm = Qwen3VLTextForCausalLM(cfg)
    config = Cosmos3VFMNetworkConfig(
        vlm_config=cfg.text_config,
        vision_gen=True,
        action_gen=True,
        latent_channel_size=4,
        latent_patch_size=2,
        action_dim=64,
        joint_attn_implementation="three_way",
        video_temporal_causal=True,
        enable_action_state_embedding=True,
        enable_vision_condition_embedding=True,
    )
    net = Cosmos3VFMNetwork(lm, config).cuda()
    # Match the training FP32 master parameters with bf16 autocast below.
    with torch.no_grad():
        for p in net.parameters():
            if p.ndim >= 2:
                p.normal_(std=0.03)
    for layer in lm.model.layers:
        layer.self_attn.dispatch_attention_fn = causal_attention.dispatch_attention_with_memory
    return net


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_compact_matches_full_predictions_and_gradients(monkeypatch, c, native):
    torch.manual_seed(190 + c)
    net = native_network() if native else make_network()
    layouts = [JointChunkLayout(16 * c + 2, 1, c), JointChunkLayout(c + 2, 1, c)]
    full = make_pack(layouts)
    if native:
        for mod in (full.vision, full.action):
            mod.tokens = [t.bfloat16() for t in mod.tokens]
    params = [p for p in net.parameters() if p.requires_grad]

    def forward(pack, memory):
        if not native:
            return net(pack, memory=memory)
        # Materialize bf16 compute parameters while accumulating into FP32 masters,
        # matching FSDP rather than converting the master gradient buffers to bf16.
        compute = {name: p.to(torch.bfloat16) for name, p in net.named_parameters()}
        return torch.func.functional_call(net, compute, (pack,), {"memory": memory})

    def run(compact):
        pack = copy.deepcopy(full)
        memory = adapter_memory(monkeypatch, pack)
        clean = make_teacher_forcing_clean_pack(pack)
        clean.to_cuda()
        forward(clean, memory)
        memory.pass_number = 2
        if compact:
            targets, maps = compact_joint_targets(pack)
            targets.to_cuda()
            view = CompactJointMemory(memory, layouts, pack.joint_text_lengths, torch.device("cuda"))
            out = restore_joint_predictions(forward(targets, view), pack, maps)
        else:
            pack.to_cuda()
            out = forward(pack, memory)
        loss = sum(p.float().square().sum() for key in ("preds_vision", "preds_action") for p in out[key])
        grad = torch.autograd.grad(loss, params, allow_unused=True)
        return out, grad

    expected, expected_grad = run(False)
    actual, actual_grad = run(True)
    for key in ("preds_vision", "preds_action"):
        for a, b in zip(actual[key], expected[key]):
            torch.testing.assert_close(a, b, atol=0.01 if native else 1e-5, rtol=0.03 if native else 1e-4)
            assert (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6) <= (0.01 if native else 0.0001)
    names = [name for name, p in net.named_parameters() if p.requires_grad]
    for name, p, a, b in zip(names, params, actual_grad, expected_grad):
        if a is None:
            a = torch.zeros_like(p)
        if b is None:
            b = torch.zeros_like(p)
        assert torch.isfinite(a).all()
        if native:
            relative = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)
            assert relative <= 0.01, (name, float(relative))
        else:
            torch.testing.assert_close(a, b, atol=2e-4, rtol=1e-4)
    # The condition-only state/image type embeddings must still receive gradients.
    for name, p in net.named_parameters():
        if "state_embed" in name or "condition_embed" in name:
            i = next(i for i, q in enumerate(params) if q is p)
            assert actual_grad[i] is not None and actual_grad[i].count_nonzero() > 0
