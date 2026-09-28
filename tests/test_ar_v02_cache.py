"""Multi-layer cache/reference equivalence: every step, first eviction and partial tail."""

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_v02_cache import JointKVCache
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import (
    ACTION,
    CONDITION_VIDEO,
    STATE,
    VIDEO,
    JointChunkLayout,
    joint_mask_mod,
)


def attend(q, k, v, allowed):
    return ((q @ k.T / q.shape[-1] ** 0.5).masked_fill(~allowed, -torch.inf).softmax(-1)) @ v


@pytest.mark.parametrize("c,tail", [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)])
@torch.no_grad()
def test_every_step_matches_causal_prefix(c, tail):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(3)
        layout = JointChunkLayout(1 + c * 16 + tail, 2, c)
        roles, chunks, _ = layout.metadata()
        n, dim, layers, nt = layout.num_tokens, 4, 2, 2
        clean_x, text = torch.randn(n, dim) * 0.1, torch.randn(nt, dim) * 0.1
        weights = [torch.randn(3, dim, dim) * 0.1 for _ in range(layers)]
        cache = JointKVCache(layout, num_layers=layers, num_kv_heads=1, head_dim=dim, device="cpu", dtype=torch.float32)

        def reference(x, clean_kv, noisy):
            length = x.shape[0]
            mask = joint_mask_mod(roles[:length], chunks[:length], text_pad_len=nt, text_len=nt, noisy=noisy)
            allowed = mask(0, 0, torch.arange(length)[:, None], torch.arange(nt + length * (2 if noisy else 1))[None])
            saved = []
            for layer, w in enumerate(weights):
                q, k, v = (x @ p for p in w)
                saved.append((k, v))
                keys, values = [text, k], [text, v]
                if noisy:
                    keys.append(clean_kv[layer][0])
                    values.append(clean_kv[layer][1])
                x = x + attend(q, torch.cat(keys), torch.cat(values), allowed)
            return x, saved

        def streamed(indexes, x, chunk, capture, initial=False):
            cache.begin(indexes, chunk=chunk, capture=capture, include_text=initial)
            cache.init(
                {
                    "_num_full_tokens": len(indexes),
                    "_num_causal_tokens": nt if initial else 0,
                    "causal_seq": torch.zeros(nt if initial else 1, dim),
                },
                torch.device("cpu"),
            )
            for layer, w in enumerate(weights):
                q, k, v = (x @ p for p in w)
                memory = cache.read_for_layer(layer)
                keys = torch.cat((text, cache.attention.assemble_gen(memory.cached_gen_k, k).reshape(-1, dim)))
                values = torch.cat((text, cache.attention.assemble_gen(memory.cached_gen_v, v).reshape(-1, dim)))
                allowed = cache.attention.predicate(nt)(
                    0, 0, torch.arange(len(indexes))[:, None], torch.arange(len(keys))[None]
                )
                x = x + attend(q, keys, values, allowed)
                cache.write_for_layer(
                    layer, (k[None, :, None], v[None, :, None], text[None, :, None], text[None, :, None])
                )
            return x

        initial = torch.empty(0, dtype=torch.long)
        streamed(initial, clean_x[initial], 0, True, initial=True)
        assert cache.text_ready and not (cache.roles >= 0).any()
        for boundary in layout.boundaries:
            chunk = boundary.chunk_id
            conditions = layout.condition_prefill_indexes(chunk)
            streamed(conditions, clean_x[conditions], chunk, True)
            current = torch.where((chunks == chunk) & ((roles == ACTION) | (roles == VIDEO)))[0]
            end = int(torch.where(chunks <= chunk)[0][-1]) + 1
            _, clean_kv = reference(clean_x[:end], None, False)
            for step in range(30):
                noisy_x = clean_x[:end].clone()
                noisy_x[current] += (30 - step) / 30 * torch.randn(len(current), dim) * 0.1
                expected, _ = reference(noisy_x, clean_kv, True)
                before = cache.ids.clone()
                before_kv = [(k.clone(), v.clone()) for k, v in cache.kv]
                before_text = [(k.clone(), v.clone()) for k, v in cache.text_kv]
                got = streamed(current, noisy_x[current], chunk, False)
                torch.testing.assert_close(got, expected[current], atol=1e-5, rtol=1e-4)
                assert (
                    torch.linalg.vector_norm(got - expected[current]) / expected[current].norm().clamp_min(1e-12)
                    <= 1e-4
                )
                # begin() may evict at a boundary; noisy forwards never append anything.
                assert torch.equal(cache.ids, before)
                for old, new in zip(before_kv + before_text, cache.kv + cache.text_kv):
                    assert all(torch.equal(a, b) for a, b in zip(old, new))
            streamed(current, clean_x[current], chunk, True)
            live = cache.ids >= 0
            assert (cache.chunks[live] >= chunk - 15).all()
            for layer, (ck, cv) in enumerate(clean_kv):
                torch.testing.assert_close(cache.kv[layer][0][0, live, 0], ck[cache.ids[live]], atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(cache.kv[layer][1][0, live, 0], cv[cache.ids[live]], atol=1e-5, rtol=1e-4)
            expected_live = torch.where((chunks >= max(1, chunk - 15)) & (chunks <= chunk))[0]
            assert torch.equal(cache.ids[live].sort().values, expected_live)
            if chunk == 16:
                assert ((cache.roles == CONDITION_VIDEO) & (cache.chunks == 1)).sum() == 2
                assert ((cache.roles == STATE) & (cache.chunks == 1)).any()
            if chunk == 17:
                assert not (cache.chunks == 1).any()
    finally:
        torch.set_num_threads(old_threads)


@torch.no_grad()
def test_condition_mask_no_future_or_answer_leakage():
    from cosmos3_joint_video_hand_pose.src.ar_v02_cache import StreamingJointAttention

    layout = JointChunkLayout(5, 3, 1)
    roles, chunks, _ = layout.metadata()
    indexes = layout.condition_prefill_indexes(2)
    attention = StreamingJointAttention(
        (roles[indexes], chunks[indexes], indexes),
        (roles, chunks, torch.arange(len(roles))),
        cached_text_only=True,
        text_len=2,
    )
    allowed = attention.predicate(4)(
        0, 0, torch.arange(len(indexes))[:, None], torch.arange(4 + attention.key_len)[None]
    )
    assert allowed[:, :2].all() and not allowed[:, 2:4].any()
    kr, kc, _ = attention.key_meta
    expected = (kc == 2) & ((kr == STATE) | (kr == CONDITION_VIDEO))
    assert torch.equal(allowed[:, 4:], expected.expand(len(indexes), -1))
    assert allowed[:, 4 + attention.query_positions].all()  # U patches and S mutually visible.


@torch.no_grad()
def test_capture_requires_complete_layers_and_text_only_start():
    layout = JointChunkLayout(4, 1, 1)
    cache = JointKVCache(layout, num_layers=2, num_kv_heads=1, head_dim=2, device="cpu", dtype=torch.float32)
    conditions = layout.condition_prefill_indexes(1)
    with pytest.raises(ValueError, match="text-only"):
        cache.begin(conditions, chunk=1, capture=True)
    with pytest.raises(ValueError, match="text-only"):
        cache.begin(conditions, chunk=1, capture=True, include_text=True)
    cache.begin([], chunk=0, capture=True, include_text=True)
    hidden = {"_num_full_tokens": 0, "_num_causal_tokens": 2, "causal_seq": torch.zeros(2, 2)}
    cache.init(hidden, torch.device("cpu"))
    empty, text = torch.zeros(1, 0, 1, 2), torch.ones(1, 2, 1, 2)
    cache.write_for_layer(0, (empty, empty, text, text))
    with pytest.raises(RuntimeError, match="incomplete"):
        cache.begin(conditions, chunk=1, capture=True)
    cache.write_for_layer(1, (empty, empty, text, text))
    cache.ensure_complete()
    with pytest.raises(ValueError, match="text-only"):
        cache.begin([], chunk=0, capture=True, include_text=True)
    cache.begin(conditions, chunk=1, capture=True)
    cache.init(
        {"_num_full_tokens": len(conditions), "_num_causal_tokens": 0, "causal_seq": torch.zeros(1, 2)},
        torch.device("cpu"),
    )
    kv = torch.ones(1, len(conditions), 1, 2)
    for layer in range(2):
        cache.write_for_layer(layer, (kv, kv, text, text))
    with pytest.raises(ValueError, match="only once"):
        cache.begin(conditions, chunk=1, capture=True)
    with pytest.raises(ValueError, match="clean refresh"):
        cache.begin(layout.condition_prefill_indexes(2), chunk=2, capture=True)


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@torch.no_grad()
def test_native_cosmos_network_cache_vs_replay(c, monkeypatch):
    """Real 2-layer MoT + VFM encoders/heads/type embeddings/RoPE/replay.

    CPU uses dense attention kernels only; GPU mode uses the native kernels.
    Set AR_V02_NATIVE_DEVICE=cuda after GPU coordination. This is a small random
    model integration test, not pretrained Nano quality/performance acceptance.
    """
    import os
    from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
    from cosmos_framework.model.generator.mot import causal_attention, flex_attention as flex_module
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
    from cosmos_framework.model.generator.utils.kv_cache import TeacherForcingMemoryState, DualKVCache
    from cosmos_framework.model.generator.teacher_forcing import (
        make_teacher_forcing_clean_pack,
        mark_modality_as_clean_condition,
    )
    from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence, select_joint_pack
    from cosmos3_joint_video_hand_pose.src.ar_v02_attention import JointTeacherForcingAttention
    from cosmos3_joint_video_hand_pose.src.ar_v02_inference import assert_numerically_close

    device = torch.device(os.environ.get("AR_V02_NATIVE_DEVICE", "cpu"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if device.type == "cpu":

            def dense(q, k, v, *, block_mask, enable_gqa=False):
                qi = torch.arange(q.shape[-2])[:, None]
                ki = torch.arange(k.shape[-2])[None]
                allowed = block_mask.mask_mod(0, 0, qi, ki)
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=allowed, enable_gqa=enable_gqa
                )

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

            monkeypatch.setattr(flex_module, "_COMPILED_FLEX_ATTENTION", dense)
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
        net = Cosmos3VFMNetwork(lm, config).to(device).float().eval()
        for parameter in net.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(std=0.03)
        for layer in lm.model.layers:
            layer.self_attn.dispatch_attention_fn = causal_attention.dispatch_attention_with_memory
        complete_chunks = int(os.environ.get("AR_V02_NATIVE_CHUNKS", "2"))
        layout = JointChunkLayout(1 + c * complete_chunks + 1, 1, c)
        video = torch.randn(1, 4, layout.num_video_frames, 2, 2, device=device)
        actions = torch.randn(layout.num_action_rows, 64, device=device) * 0.1
        roles, chunks, _ = layout.metadata()
        ar, ac, _ = layout.action_metadata(device=device)
        actions[:, 57:] = 0
        actions[ar == STATE, :9] = 0
        data = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=[video],
            x0_tokens_action=[actions],
            fps_vision=torch.tensor([7.5]),
            fps_action=torch.tensor([15.0]),
            raw_action_dim=[torch.tensor(57)],
        )

        def pack(lay, vid, act, conditions=(), sigma=0.0):
            import dataclasses

            part = dataclasses.replace(data, x0_tokens_vision=[vid], x0_tokens_action=[act])
            result = pack_joint_sequence(
                layout=lay,
                gen_data_clean=part,
                text_ids=[3, 4, 5],
                special_tokens={"eos_token_id": 1, "start_of_generation": 2},
                timesteps=sigma,
                latent_patch_size=2,
                condition_frames=conditions,
            )
            if device.type == "cuda":
                result.to_cuda()
            return result

        full = pack(layout, video, actions, sigma=500.0)
        cache = JointKVCache(layout, num_layers=2, num_kv_heads=2, head_dim=16, device=device, dtype=torch.float32)

        def streamed(indexes, chunk, capture, initial=False, sigma=0.0):
            selected = select_joint_pack(full, layout, indexes, include_text=initial)
            for mod in (selected.vision, selected.action):
                if mod is not None:
                    if capture:
                        mark_modality_as_clean_condition(mod)
                    else:
                        mod.timesteps.fill_(sigma)
            cache.begin(indexes, chunk=chunk, capture=capture, include_text=initial)
            if device.type == "cuda":
                selected.to_cuda()
            result = net(selected, memory=cache)
            cache.ensure_complete()
            return result

        streamed([], 0, True, True)
        assert cache.text_ready
        for b in layout.boundaries:
            lay = JointChunkLayout(b.latent_stop, 1, c)
            nv, na = lay.num_video_frames, lay.num_action_rows
            prefix = pack(lay, video[:, :, :nv], actions[:na])
            replay = TeacherForcingMemoryState(
                vision_token_shapes=prefix.vision.token_shapes,
                num_action_tokens_per_supertoken=0,
                null_action_supertokens=False,
                segment_idx=0,
                dual_kv_cache=[DualKVCache(gen_cache_size=2) for _ in range(2)],
                num_kv_heads=2,
                head_dim=16,
            )
            attention = JointTeacherForcingAttention(lay, device=device)
            base_read = replay.read_for_layer

            def read(layer, base_read=base_read, attention=attention):
                value = base_read(layer)
                value.gen_attention_override = attention
                return value

            replay.read_for_layer = read
            net(make_teacher_forcing_clean_pack(prefix), memory=replay)
            conditions = layout.condition_prefill_indexes(b.chunk_id)
            streamed(conditions, b.chunk_id, True)
            for layer in range(2):
                live = cache.ids >= 0
                for got, expected in zip(cache.kv[layer], replay._clean_gen_kv[layer]):
                    assert_numerically_close(
                        got[:, live], expected[:, cache.ids[live]], fp32=True, context="prefill KV"
                    )
                base_value = replay._read_teacher_forcing_base_value(layer)
                for got, expected in zip(cache.text_kv[layer], (base_value.cached_und_k, base_value.cached_und_v)):
                    assert_numerically_close(got, expected[:, : cache.text_len], fp32=True, context="text KV")
            vi = layout.video_indexes(b.chunk_id, False).to(device)
            rows = torch.where((ar == ACTION) & (ac == b.chunk_id))[0]
            target = torch.where((chunks == b.chunk_id) & ((roles == VIDEO) | (roles == ACTION)))[0]
            _, vchunks, _ = lay.video_metadata()
            history = torch.where(vchunks < b.chunk_id)[0].tolist()
            replay.pass_number = 2
            original_v, original_a = video[:, :, vi].clone(), actions[rows].clone()
            for step in range(30):
                sigma = (30 - step) / 30
                video[:, :, vi] = original_v + sigma * 0.1
                actions[rows, :57] = original_a[:, :57] + sigma * 0.05
                noisy = pack(lay, video[:, :, :nv], actions[:na], conditions=history, sigma=sigma * 1000)
                before = [(k.clone(), v.clone()) for k, v in cache.kv]
                got = streamed(target, b.chunk_id, False, sigma=sigma * 1000)
                expected = net(noisy, memory=replay)
                assert_numerically_close(
                    got["preds_vision"][0], expected["preds_vision"][0][:, :, vi], fp32=True, context="video flow"
                )
                assert_numerically_close(
                    got["preds_action"][0], expected["preds_action"][0][rows], fp32=True, context="action flow"
                )
                for old, new in zip(before, cache.kv):
                    assert all(torch.equal(x, y) for x, y in zip(old, new))
            video[:, :, vi], actions[rows] = original_v, original_a
            streamed(target, b.chunk_id, True)
            for layer in range(2):
                live = cache.ids >= 0
                for got, expected in zip(cache.kv[layer], replay._clean_gen_kv[layer]):
                    assert_numerically_close(
                        got[:, live], expected[:, cache.ids[live]], fp32=True, context="refresh KV"
                    )
        assert cache.forward_calls == 1 + 32 * len(layout.boundaries)
    finally:
        torch.set_num_threads(old_threads)


@torch.no_grad()
def test_text_only_framework_branch_has_no_gen_output(monkeypatch):
    from cosmos_framework.model.generator.mot.attention import build_packed_sequence
    from cosmos_framework.model.generator.mot import causal_attention

    nt, heads, dim = 3, 2, 4
    tokens = torch.arange(nt * heads * dim).float().reshape(nt, heads * dim) / 20
    pack, meta, _ = build_packed_sequence(
        "three_way",
        packed_sequence=tokens,
        attn_modes=["causal", "full"],
        split_lens=[nt, 0],
        sample_lens=[nt],
        packed_und_token_indexes=torch.arange(nt),
        packed_gen_token_indexes=torch.empty(0, dtype=torch.long),
        num_heads=heads,
        head_dim=dim,
        num_layers=1,
        skip_natten_metadata=True,
    )
    cache = JointKVCache(
        JointChunkLayout(2, 1, 1), num_layers=1, num_kv_heads=heads, head_dim=dim, device="cpu", dtype=torch.float32
    )
    cache.begin([], chunk=0, capture=True, include_text=True)
    cache.init(pack, torch.device("cpu"))
    qkv = dict(pack)
    for key in ("causal_seq", "full_only_seq"):
        qkv[key] = pack[key].reshape(-1, heads, dim)

    def cpu_attention(q, k, v, **kwargs):
        assert kwargs["is_causal"]
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2)

    monkeypatch.setattr(causal_attention, "attention", cpu_attention)
    out = causal_attention.three_way_attention_with_kv_cache(
        qkv, qkv, qkv, cache.read_for_layer(0), attention_meta=meta
    )
    real, _ = causal_attention.get_causal_seq(out)
    gen, _ = causal_attention.get_full_only_seq(out)
    values = tokens.reshape(nt, heads, dim).transpose(0, 1)
    scores = values @ values.transpose(-1, -2) / dim**0.5
    scores = scores.masked_fill(~torch.ones(nt, nt, dtype=torch.bool).tril(), -torch.inf)
    expected = (scores.softmax(-1) @ values).transpose(0, 1).flatten(1)
    torch.testing.assert_close(real[:nt], expected)
    assert not gen.count_nonzero()
    assert out["_num_full_tokens"] == 0
