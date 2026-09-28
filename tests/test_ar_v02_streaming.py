"""CPU streaming correctness; native small MoT runs with dense CPU kernels.

No GPU is selected here. Full-prefix tensors exist only in these reference tests.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_chunk_state import (
    ChunkCameraState,
    ChunkCameraStateNormalizer,
    encode_chunk_camera_state,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_cache import BoundedJointKVCache
from cosmos3_joint_video_hand_pose.src.ar_v02_inference import assert_numerically_close
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import (
    ACTION,
    CONDITION_VIDEO,
    STATE,
    VIDEO,
    JointChunkLayout,
    joint_mask_mod,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence, select_joint_pack
from cosmos3_joint_video_hand_pose.src.ar_v02_streaming import ChunkPackWorkspace, StreamingJointSampler
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

SPECIAL = {"eos_token_id": 1, "start_of_generation": 2}


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def normalizers():
    state = object.__new__(ChunkCameraStateNormalizer)
    state.center, state.scale, state.beta = torch.zeros(18), torch.ones(18), 1.0
    return state, SimpleNamespace(normalize=lambda x: x, denormalize=lambda x: x)


def observation(source=0):
    rigid = torch.eye(4).repeat(3, 1, 1)
    rigid[1, 0, 3], rigid[2, 0, 3] = 0.3, -0.3
    rigid[1:, 2, 3] = 1.0
    return ChunkCameraState(source, rigid, torch.full((2, 15), 0.02))


def future(frames):
    action = torch.zeros(frames * 8, 64)
    for start in (0, 9, 33):
        action[:, start + 3] = action[:, start + 7] = 1
        action[:, start] = 0.001
    return action


def pack(layout, video, action, sigma=0.0, conditions=(), margin=0.0, reset_spatial=True):
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[video],
        x0_tokens_action=[action],
        fps_vision=torch.tensor([7.5]),
        fps_action=torch.tensor([15.0]),
        raw_action_dim=[torch.tensor(57)],
        action_domain_id=[torch.tensor([0])],
    )
    return pack_joint_sequence(
        layout=layout,
        gen_data_clean=data,
        text_ids=[3, 4, 5],
        special_tokens=SPECIAL,
        timesteps=sigma,
        latent_patch_size=2,
        condition_frames=conditions,
        modality_margin=margin,
        reset_spatial=reset_spatial,
    )


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("reset_spatial", [True, False])
@torch.no_grad()
def test_direct_packs_match_training_selection_and_absolute_rope(c, reset_spatial):
    # Includes first eviction and >2 ring revolutions. Reference allocation only.
    layout = JointChunkLayout(1 + 49 * c, 1, c)
    video = torch.randn(1, 4, layout.num_video_frames, 2, 2)
    action = torch.randn(layout.num_action_rows, 64)
    reference = pack(layout, video, action, sigma=123.0, margin=15000.0, reset_spatial=reset_spatial)
    workspace = ChunkPackWorkspace(
        text_ids=[3, 4, 5],
        special_tokens=SPECIAL,
        latent_shape=(4, 2, 2),
        chunk_size=c,
        patch_size=2,
        device="cpu",
        dtype=torch.float32,
        modality_margin=15000.0,
        reset_spatial=reset_spatial,
    )
    torch.testing.assert_close(
        workspace.text.position_ids, select_joint_pack(reference, layout, [], include_text=True).position_ids
    )
    r, chunks, _ = layout.metadata()
    ar, ac, _ = layout.action_metadata()
    addresses = {key: value.position_ids.data_ptr() for key, value in workspace.packs.items()}
    for chunk in (1, 16, 17, 33, 49):
        b = layout.boundaries[chunk - 1]
        workspace.set_boundary(b.source_start, c)
        for phase in ("condition", "noisy"):
            cond = phase == "condition"
            indexes = (
                layout.condition_prefill_indexes(chunk)
                if cond
                else torch.where((chunks == chunk) & ((r == VIDEO) | (r == ACTION)))[0]
            )
            expected = select_joint_pack(reference, layout, indexes, include_text=False)
            vi = layout.video_indexes(chunk, condition=cond)
            ai = torch.where((ac == chunk) & (ar == (STATE if cond else ACTION)))[0]
            got = workspace.load(phase, video[:, :, vi], action[ai], video_t=123.0, action_t=123.0)
            for name in ("position_ids", "action_state_mask", "vision_condition_type_mask"):
                torch.testing.assert_close(getattr(got, name), getattr(expected, name))
            for name in ("vision", "action"):
                gm, em = getattr(got, name), getattr(expected, name)
                for attr in ("sequence_indexes", "mse_loss_indexes", "timesteps"):
                    torch.testing.assert_close(getattr(gm, attr), getattr(em, attr))
                torch.testing.assert_close(gm.tokens[0], em.tokens[0])
                torch.testing.assert_close(gm.condition_mask[0], em.condition_mask[0])
    assert addresses == {key: value.position_ids.data_ptr() for key, value in workspace.packs.items()}


class PerfectModel:
    """Runs the real cache protocol; exact finite flow makes history policy observable."""

    def __init__(self):
        self.tensor_kwargs = dict(device="cpu", dtype=torch.float32)
        self.config = SimpleNamespace(
            diffusion_expert_config=SimpleNamespace(
                patch_spatial=2,
                base_fps=24.0,
                unified_3d_mrope_reset_spatial_ids=True,
                unified_3d_mrope_temporal_modality_margin=15000.0,
            )
        )
        self.net = SimpleNamespace(num_hidden_layers=2, num_kv_heads=1, head_dim=4)
        self.llm_special_tokens = SPECIAL
        flow = SimpleNamespace(noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)))
        self.rectified_flow_video = self.rectified_flow_action = flow
        self.records = []
        self.decoded_blocks = []
        self.observer = None

    def encode(self, rgb):
        return rgb * 0.1

    def decode(self, block):
        self.decoded_blocks.append(block.clone())
        return block.sum(2, keepdim=True).expand(-1, -1, 1 + 4 * (block.shape[2] - 1), -1, -1).clone()

    def denoise(self, *, data_batch_packed, memory):
        p = data_batch_packed
        n = p.sequence_length - p.text_ids.numel()
        memory.init(
            {
                "_num_full_tokens": n,
                "_num_causal_tokens": p.text_ids.numel(),
                "causal_seq": torch.zeros(max(1, p.text_ids.numel()), 4),
            },
            torch.device("cpu"),
        )
        x = torch.zeros(n, 4)
        if n:
            # Preserve payload identity in every layer's stored K/V for policy checks.
            x[p.vision.sequence_indexes] = p.vision.tokens[0].mean((0, 1, 3, 4))[:, None]
            x[p.action.sequence_indexes] = p.action.tokens[0][:, :4]
        self.records.append((memory.current_chunk, memory._phase))
        if self.observer is not None:
            self.observer(p, memory)
        for layer in range(2):
            kv = (x + layer)[None, :, None]
            txt = torch.ones(1, max(1, memory.text_len), 1, 4)
            memory.write_for_layer(layer, (kv, kv, txt, txt))
        if not n:
            return {}
        v, a = p.vision.tokens[0], p.action.tokens[0]
        if memory._phase != "noisy":
            return {"preds_vision": [torch.zeros_like(v)], "preds_action": [torch.zeros_like(a)]}
        sv, sa = float(p.vision.timesteps[0]) / 1000, float(p.action.timesteps[0]) / 1000
        desired = future(v.shape[2])
        desired[:, 9] += 0.003
        return {"preds_vision": [(v - 0.75) / max(sv, 1e-6)], "preds_action": [(a - desired) / sa]}


def stream(model=None, c=4, history="gt", **kwargs):
    sn, an = normalizers()
    return StreamingJointSampler(
        model or PerfectModel(),
        text_ids=[3, 4, 5],
        latent_shape=(4, 2, 2),
        state_normalizer=sn,
        future_normalizer=an,
        chunk_size=c,
        history=history,
        **kwargs,
    )


@pytest.mark.parametrize("c,tail", [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("history", ["gt", "oracle", "pred_history", "generated"])
@torch.no_grad()
def test_48_chunks_bounded_storage_protocol_history_and_tail(c, tail, history, monkeypatch):
    sampler = stream(
        c=c, history=history, video_schedule=torch.linspace(1, 0, 31), action_schedule=torch.linspace(1, 0, 31).square()
    )
    # Production must never fall back to either whole-clip or remapping paths.
    import cosmos3_joint_video_hand_pose.src.ar_v02_packing as packing

    def forbidden(*args, **kwargs):
        raise AssertionError("full-clip/remap path called")

    monkeypatch.setattr(packing, "pack_joint_sequence", forbidden)
    monkeypatch.setattr(packing, "select_joint_pack", forbidden)
    monkeypatch.setattr(JointChunkLayout, "metadata", forbidden)
    cache = sampler.cache
    ptrs = [t.data_ptr() for kv in cache.kv for t in kv]
    assert not hasattr(cache, "all_roles") and not hasattr(cache, "layout")
    saved = None
    for k in range(1, 50):
        frames = c if k <= 48 else tail
        u = torch.full((1, 4, 1, 2, 2), 2.0 + k / 10)
        state = observation(sampler.source_index)
        source = sampler.source_index
        gt_v, gt_a = torch.full((1, 4, frames, 2, 2), 3.0), future(frames)
        args = dict(gt_video=gt_v, gt_action=gt_a) if history in ("gt", "oracle") else {}
        if history == "generated" and k > 1:
            u = state = None

        def observer(p, memory):
            if memory._phase == "noisy":
                allowed = memory.attention.predicate(memory.text_len)(
                    0,
                    0,
                    torch.arange(len(memory.query_meta[0]))[:, None],
                    torch.arange(memory.text_len + memory.attention.key_len)[None],
                )
                visible = memory.chunks[(memory.roles >= 0) & (memory.chunks < k)].unique().tolist()
                assert visible == list(range(max(1, k - 15), k))
                real_keys = memory.attention.key_meta[0] >= 0
                assert allowed[:, memory.text_len :][:, real_keys].all()
                assert not allowed[:, memory.text_len :][:, ~real_keys].any()
                assert memory.attention.key_len <= memory.capacity + (sampler.workspace.vision_tokens + 8) * c + 3 * 128
            if memory._phase == "refresh":
                expected_a = gt_a if history in ("gt", "oracle") else future(frames)
                if history not in ("gt", "oracle"):
                    expected_a[:, 9] += 0.003
                torch.testing.assert_close(p.action.tokens[0], expected_a, atol=1e-6, rtol=1e-5)

        sampler.model.observer = observer
        result = sampler.step(u, state, frames=frames, **args)
        expected = future(frames)
        expected[:, 9] += 0.003
        torch.testing.assert_close(result.action, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result.video, gt_v if history == "oracle" else torch.full_like(gt_v, 0.75))
        assert result.report["forward_calls"] == 32 and cache.forward_calls == 1 + 32 * k
        assert result.report["boundary_source_index"] == source
        assert result.report["condition_source"] == ("prediction" if history == "generated" and k > 1 else "gt")
        assert ptrs == [t.data_ptr() for kv in cache.kv for t in kv]
        assert len(cache._role_templates) == c + 1
        assert len(sampler.workspace.packs) == 1 + 2 * c
        assert result.report["cache_tokens"] <= cache.capacity
        assert not result.action[:, 57:].count_nonzero()
        if k == 16:
            assert (cache.chunks == 1).any()
        if k == 17:
            assert not (cache.chunks == 1).any()
        if saved is None:
            saved = (result, result.action.clone(), result.video.clone())
        torch.testing.assert_close(saved[0].action, saved[1], rtol=0, atol=0)
        torch.testing.assert_close(saved[0].video, saved[2], rtol=0, atol=0)
        # No production output log is retained; the diagnostic model's log is caller-owned.
        sampler.model.records.clear()
        sampler.model.decoded_blocks.clear()
    if tail < c:
        with pytest.raises(RuntimeError, match="reset"):
            sampler.step()
    sampler.reset(seed=123)
    assert not cache.text_ready and cache.forward_calls == 0 and (cache.roles < 0).all()
    assert all(not t.count_nonzero() for kv in cache.kv for t in kv)
    assert ptrs == [t.data_ptr() for kv in cache.kv for t in kv]


@torch.no_grad()
def test_generated_next_condition_uses_complete_block_and_owned_terminal():
    sampler = stream(history="generated")
    first = sampler.step(torch.ones(1, 4, 1, 2, 2) * 2, observation())
    expected_u = torch.cat((first.condition_video, first.video), 2).sum(2, keepdim=True) * 0.1
    expected_s = encode_chunk_camera_state(first.decoded.end_state, sampler.state_normalizer)
    # Caller changes its returned buffers. Private continuation must remain unaffected.
    first.video.fill_(99)
    first.decoded.end_state.hand_latents.fill_(99)
    second = sampler.step()
    torch.testing.assert_close(second.condition_video, expected_u)
    torch.testing.assert_close(second.condition_state[:57], expected_s)
    assert sampler.model.decoded_blocks[0].shape[2] == 5
    with pytest.raises(ValueError, match="external U/S"):
        sampler.step(torch.ones(1, 4, 1, 2, 2), observation(64))
    with pytest.raises(RuntimeError, match="reset"):
        sampler.step()


@torch.no_grad()
def test_invalid_inputs_failure_recovery_and_fixed_steps():
    with pytest.raises(ValueError, match="exactly 30"):
        stream(steps=20)
    with pytest.raises(ValueError, match="31"):
        stream(video_schedule=torch.linspace(1, 0, 21))
    sampler = stream(history="pred_history")
    with pytest.raises(ValueError, match="future GT"):
        sampler.step(torch.ones(1, 4, 1, 2, 2), observation(), gt_action=future(4))
    with pytest.raises(RuntimeError, match="reset"):
        sampler.step()
    sampler.reset(source_start=64)
    first = sampler.step(torch.ones(1, 4, 1, 2, 2), observation(64), condition_source="observation")
    assert first.report["boundary_source_index"] == 64
    assert first.report["condition_source"] == "observation"
    sampler.reset(source_start=64)
    second = sampler.step(torch.ones(1, 4, 1, 2, 2), observation(64))
    torch.testing.assert_close(first.action, second.action, rtol=0, atol=0)


def native_model(monkeypatch):
    from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
    from cosmos_framework.model.generator.mot import causal_attention, flex_attention

    def dense(q, k, v, *, block_mask, enable_gqa=False):
        allowed = block_mask.mask_mod(0, 0, torch.arange(q.shape[-2])[:, None], torch.arange(k.shape[-2])[None])
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=enable_gqa)

    def text_attention(q, k, v, offsets):
        tq, _ = causal_attention.get_causal_seq(q)
        tk, _ = causal_attention.get_causal_seq(k)
        tv, _ = causal_attention.get_causal_seq(v)
        length = int(offsets[-1])
        out = tq.new_zeros(tq.shape[0], tq.shape[1] * tq.shape[2])
        result = torch.nn.functional.scaled_dot_product_attention(
            tq[:length].transpose(0, 1)[None],
            tk[:length].transpose(0, 1)[None],
            tv[:length].transpose(0, 1)[None],
            is_causal=True,
            enable_gqa=True,
        )
        out[:length] = result[0].transpose(0, 1).flatten(1)
        return out

    monkeypatch.setattr(flex_attention, "_COMPILED_FLEX_ATTENTION", dense)
    monkeypatch.setattr(causal_attention, "_three_way_text_self_attention", text_attention)
    torch.manual_seed(27)
    cfg = Qwen3VLMoTConfig(
        config_dict={
            "text_config": dict(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
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
    net = Cosmos3VFMNetwork(lm, config).float().eval()
    with torch.no_grad():
        for p in net.parameters():
            if p.ndim >= 2:
                p.normal_(std=0.03)
    for layer in lm.model.layers:
        layer.self_attn.dispatch_attention_fn = causal_attention.dispatch_attention_with_memory
    model = PerfectModel()
    model.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin = 0.0
    model.net = net
    model.denoise = lambda *, data_batch_packed, memory: net(data_batch_packed, memory=memory)
    return model


def replay_clean(net, packed, layout):
    from cosmos_framework.model.generator.utils.kv_cache import TeacherForcingMemoryState, DualKVCache
    from cosmos_framework.model.generator.teacher_forcing import make_teacher_forcing_clean_pack
    from cosmos3_joint_video_hand_pose.src.ar_v02_attention import JointTeacherForcingAttention

    replay = TeacherForcingMemoryState(
        vision_token_shapes=packed.vision.token_shapes,
        num_action_tokens_per_supertoken=0,
        null_action_supertokens=False,
        segment_idx=0,
        dual_kv_cache=[DualKVCache(gen_cache_size=2) for _ in range(2)],
        num_kv_heads=2,
        head_dim=16,
    )
    attention = JointTeacherForcingAttention(layout, device=torch.device("cpu"))
    base_read = replay.read_for_layer

    def read(layer):
        value = base_read(layer)
        value.gen_attention_override = attention
        return value

    replay.read_for_layer = read
    net(make_teacher_forcing_clean_pack(packed), memory=replay)
    return replay


@pytest.mark.parametrize("c,tail", [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)])
@torch.no_grad()
def test_native_mot_every_euler_step_and_kv_vs_complete_prefix(c, tail, monkeypatch):
    model = native_model(monkeypatch)
    sampler = stream(
        model,
        c=c,
        history="gt",
        video_schedule=torch.linspace(1, 0, 31),
        action_schedule=torch.linspace(1, 0, 31).square(),
    )
    videos, actions = [], []
    original = model.denoise
    for chunk, frames in enumerate((c, c, tail), start=1):
        source = (chunk - 1) * c * 8
        u = torch.full((1, 4, 1, 2, 2), 1.0 + chunk / 10)
        state = observation(source)
        encoded = torch.nn.functional.pad(encode_chunk_camera_state(state, sampler.state_normalizer), (0, 7))[None]
        gv = torch.randn(1, 4, frames, 2, 2) * 0.1
        ga = future(frames)
        videos.append(torch.cat((u, gv), 2))
        actions.append(torch.cat((encoded, ga)))
        video, action = torch.cat(videos, 2), torch.cat(actions)
        layout = JointChunkLayout(1 + (chunk - 1) * c + frames, 1, c)
        prefix = pack(layout, video, action)
        replay = replay_clean(model.net, prefix, layout)
        ar, ac, _ = layout.action_metadata()
        rows = torch.where((ar == ACTION) & (ac == chunk))[0]
        vi = layout.video_indexes(chunk, False)
        _, vchunks, _ = layout.video_metadata()
        history = torch.where(vchunks < chunk)[0].tolist()
        step = 0

        def checked(*, data_batch_packed, memory):
            nonlocal step
            p = data_batch_packed
            before = [(k.clone(), v.clone()) for k, v in memory.kv]
            got = original(data_batch_packed=p, memory=memory)
            if memory._phase == "noisy":
                noisy_v, noisy_a = video.clone(), action.clone()
                noisy_v[:, :, vi] = p.vision.tokens[0]
                noisy_a[rows] = p.action.tokens[0]
                noisy = pack(layout, noisy_v, noisy_a, conditions=history, sigma=float(p.vision.timesteps[0]))
                noisy.action.timesteps.fill_(float(p.action.timesteps[0]))
                replay.pass_number = 2
                expected = model.net(noisy, memory=replay)
                for name, indexes in (("vision", vi), ("action", rows)):
                    ref = expected["preds_" + name][0]
                    ref = ref[:, :, indexes] if name == "vision" else ref[indexes]
                    result = got["preds_" + name][0]
                    assert_numerically_close(result, ref, fp32=True, context=f"{name} {chunk}/{step}")
                    current = p.vision.tokens[0] if name == "vision" else p.action.tokens[0]
                    schedule = sampler.sv if name == "vision" else sampler.sa
                    assert_numerically_close(
                        current + (schedule[step + 1] - schedule[step]) * result,
                        current + (schedule[step + 1] - schedule[step]) * ref,
                        fp32=True,
                        context="Euler",
                    )
                assert all(torch.equal(a, b) for old, new in zip(before, memory.kv) for a, b in zip(old, new))
                step += 1
            elif memory._phase in ("condition", "refresh"):
                live = memory.ids >= 0
                for pair, reference in zip(memory.kv, replay._clean_gen_kv):
                    for got_kv, ref_kv in zip(pair, reference):
                        assert_numerically_close(
                            got_kv[:, live], ref_kv[:, memory.ids[live]], fp32=True, context=memory._phase + " KV"
                        )
            return got

        model.denoise = checked
        result = sampler.step(u, state, gt_video=gv, gt_action=ga, frames=frames)
        assert step == 30
        assert torch.isfinite(result.decoded.rigid_chunk).all()
