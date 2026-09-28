"""GPU-only real Cosmos pack/encode/attention/decode integration (small backbone)."""

from types import SimpleNamespace
import copy
import pytest
import torch

from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.causal_attention import three_way_attention_with_kv_cache
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.utils.kv_cache import DualKVCache, TeacherForcingMemoryState
from cosmos_framework.model.generator.teacher_forcing import (
    mark_modality_as_clean_condition,
    make_teacher_forcing_clean_pack,
)
from cosmos3_joint_video_hand_pose.src.ar_model import OmniMoTCausalModel
from cosmos3_joint_video_hand_pose.src.ar_v02_model import EgoVerseARV02Model
from cosmos3_joint_video_hand_pose.src.ar_v02_cache import JointKVCache
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, CONDITION_VIDEO, STATE, VIDEO, JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence, select_joint_pack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; main agent schedules GPU runs")


class TinyReasoner(torch.nn.Module):
    """Two layers expose indirect text leakage; this is not the pretrained Nano backbone."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=4, head_dim=16, num_hidden_layers=2
        )
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(32, 64)
        self.projections = torch.nn.ModuleList([torch.nn.Linear(64, 192) for _ in range(2)])

    def forward(self, input_pack, attention_mask=None, position_ids=None, natten_metadata_list=None, memory=None):
        memory.init(input_pack, input_pack["full_only_seq"].device)
        hidden = input_pack
        for layer, projection in enumerate(self.projections):
            q, k, v = dict(hidden), dict(hidden), dict(hidden)
            for name in ("causal_seq", "full_only_seq"):
                values = projection(hidden[name]).chunk(3, -1)
                q[name], k[name], v[name] = (x.reshape(-1, 4, 16) for x in values)
            attended = three_way_attention_with_kv_cache(
                q, k, v, memory.read_for_layer(layer), attention_meta=attention_mask
            )
            g, t = hidden["_num_full_tokens"], hidden["_num_causal_tokens"]
            memory.write_for_layer(
                layer,
                (
                    k["full_only_seq"][:g][None],
                    v["full_only_seq"][:g][None],
                    k["causal_seq"][:t][None],
                    v["causal_seq"][:t][None],
                ),
            )
            output = dict(hidden)
            for name in ("causal_seq", "full_only_seq"):
                output[name] = hidden[name] + 0.2 * attended[name]
            hidden = output
        return hidden, {}


def make_network():
    lm = TinyReasoner()
    config = Cosmos3VFMNetworkConfig(
        vlm_config=lm.config,
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
    return Cosmos3VFMNetwork(lm, config).cuda().float()


def make_pack(layouts, texts=None):
    texts = texts or [[3, 4], [7, 8, 9]][: len(layouts)]
    data = GenerationDataClean(
        batch_size=len(layouts),
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 4, x.num_video_frames, 2, 2, device="cuda") for x in layouts],
        x0_tokens_action=[
            torch.cat(
                (torch.randn(x.num_action_rows, 57, device="cuda"), torch.zeros(x.num_action_rows, 7, device="cuda")), 1
            )
            for x in layouts
        ],
        fps_vision=torch.tensor([7.5] * len(layouts)),
        fps_action=torch.tensor([15.0] * len(layouts)),
    )
    return pack_joint_sequence(
        layout=layouts,
        gen_data_clean=data,
        text_ids=texts,
        special_tokens={"eos_token_id": 1, "start_of_generation": 2},
        timesteps=[500] * len(layouts),
        condition_frames=[() for _ in layouts],
        latent_patch_size=2,
    )


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@torch.no_grad()
def test_text_then_each_chunk_condition_prefill_noisy_and_clean_refresh(c):
    torch.manual_seed(42)
    net = make_network()
    layout = JointChunkLayout(c + 2, 1, c)
    full = make_pack([layout])
    cache = JointKVCache(layout, num_layers=2, num_kv_heads=4, head_dim=16, device="cuda", dtype=torch.float32)
    roles, chunks, _ = layout.metadata()

    def run(indexes, chunk, capture, include_text=False):
        pack = select_joint_pack(full, layout, indexes, include_text=include_text)
        if capture:
            for mod in (pack.vision, pack.action):
                if mod is not None:
                    mark_modality_as_clean_condition(mod)
        cache.begin(indexes, chunk=chunk, capture=capture, include_text=include_text)
        pack.to_cuda()
        return net(pack, memory=cache)

    run([], 0, True, True)
    assert cache.text_ready and not (cache.roles >= 0).any()
    for b in layout.boundaries:
        run(layout.condition_prefill_indexes(b.chunk_id), b.chunk_id, True)
        assert int(((cache.roles == STATE) & (cache.chunks == b.chunk_id)).sum()) == 1
        assert int(((cache.roles == CONDITION_VIDEO) & (cache.chunks == b.chunk_id)).sum()) == 1
        current = torch.where(((roles == VIDEO) | (roles == ACTION)) & (chunks == b.chunk_id))[0]
        saved = [(k.clone(), v.clone()) for k, v in cache.kv]
        out = run(current, b.chunk_id, False)
        assert out["preds_vision"][0].shape[-3:] == (b.latent_stop - b.latent_start, 2, 2)
        assert out["preds_action"][0].shape == (b.action_count, 64)
        assert torch.isfinite(out["preds_action"][0]).all()
        for (k, v), (sk, sv) in zip(cache.kv, saved):
            torch.testing.assert_close(k, sk, atol=0, rtol=0)
            torch.testing.assert_close(v, sv, atol=0, rtol=0)
        run(current, b.chunk_id, True)


def adapter_memory(monkeypatch, packed):
    # Exercise the real model adapter, including its init override for text offsets.
    def parent(self, packed_sequence, memory_info, **kwargs):
        return TeacherForcingMemoryState(
            packed_sequence.vision.token_shapes,
            0,
            False,
            0,
            [DualKVCache(gen_cache_size=2) for _ in range(2)],
            4,
            16,
            detach_clean_kv=False,
        )

    monkeypatch.setattr(OmniMoTCausalModel, "_build_tf_memory_state", parent)
    model = object.__new__(EgoVerseARV02Model)
    torch.nn.Module.__init__(model)
    model._joint_layouts = packed.joint_layouts
    return model._build_tf_memory_state(packed, {})


def replay(net, packed, monkeypatch, clean_source=None):
    memory = adapter_memory(monkeypatch, packed)
    clean = make_teacher_forcing_clean_pack(clean_source if clean_source is not None else packed)
    clean.to_cuda()
    net(clean, memory=memory)
    memory.pass_number = 2
    packed.to_cuda()
    return net(packed, memory=memory), memory


def objective(out):
    return sum(p.square().mean() for name in ("preds_vision", "preds_action") for p in out[name])


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_packed_replay_matches_separate_values_and_parameter_gradients(monkeypatch, c):
    torch.manual_seed(40 + c)
    net = make_network()
    layouts = [JointChunkLayout(c + 2, 1, c), JointChunkLayout(2 * c + 2, 1, c)]
    packed = make_pack(layouts)
    combined, _ = replay(net, copy.deepcopy(packed), monkeypatch)
    parameters = [p for p in net.parameters() if p.requires_grad]
    joint_grads = torch.autograd.grad(objective(combined) / 2, parameters, allow_unused=True)
    separate = []
    for i, layout in enumerate(layouts):
        data = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=[packed.vision.tokens[i]],
            x0_tokens_action=[packed.action.tokens[i]],
            fps_vision=torch.tensor([7.5]),
            fps_action=torch.tensor([15.0]),
        )
        single = pack_joint_sequence(
            layout=layout,
            gen_data_clean=data,
            text_ids=([3, 4] if i == 0 else [7, 8, 9]),
            special_tokens={"eos_token_id": 1, "start_of_generation": 2},
            timesteps=500,
            latent_patch_size=2,
        )
        out, _ = replay(net, single, monkeypatch)
        separate.append(out)
        for name in ("preds_vision", "preds_action"):
            torch.testing.assert_close(combined[name][i], out[name][0], atol=1e-5, rtol=1e-4)
    separate_grads = torch.autograd.grad(sum(objective(x) for x in separate) / 2, parameters, allow_unused=True)
    for a, b in zip(joint_grads, separate_grads):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_replay_text_and_payload_perturbation_cannot_cross_samples(monkeypatch):
    torch.manual_seed(51)
    net = make_network()
    layouts = [JointChunkLayout(4, 1, 2), JointChunkLayout(6, 1, 2)]
    packed = make_pack(layouts)
    baseline, _ = replay(net, copy.deepcopy(packed), monkeypatch)
    changed = copy.deepcopy(packed)
    # Perturb the EARLIER sample: merged causal text offsets would leak into sample 2.
    changed.text_ids[:2] = torch.tensor([20, 21])
    changed.vision.tokens[0] += 2
    changed.action.tokens[0][:, :57] -= 3
    actual, _ = replay(net, changed, monkeypatch)
    for name in ("preds_vision", "preds_action"):
        torch.testing.assert_close(actual[name][1], baseline[name][1], atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_multilayer_replay_clean_answer_and_future_condition_leakage(monkeypatch):
    torch.manual_seed(61)
    net = make_network()
    layout = JointChunkLayout(8, 1, 2)
    packed = make_pack([layout])
    baseline, memory = replay(net, copy.deepcopy(packed), monkeypatch)
    clean_changed = copy.deepcopy(packed)
    vr, vc, _ = layout.video_metadata()
    ar, ac, _ = layout.action_metadata()
    clean_changed.vision.tokens[0][:, :, (vc > 2) | ((vc == 2) & (vr == VIDEO))] += 3
    clean_changed.action.tokens[0][(ac > 2) | ((ac == 2) & (ar == ACTION)), :57] -= 2
    actual, changed_memory = replay(net, copy.deepcopy(packed), monkeypatch, clean_source=clean_changed)
    torch.testing.assert_close(
        actual["preds_vision"][0][:, :, (vc == 2) & (vr == VIDEO)],
        baseline["preds_vision"][0][:, :, (vc == 2) & (vr == VIDEO)],
        atol=1e-5,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        actual["preds_action"][0][(ac == 2) & (ar == ACTION)],
        baseline["preds_action"][0][(ac == 2) & (ar == ACTION)],
        atol=1e-5,
        rtol=1e-4,
    )
    conditions = layout.condition_prefill_indexes(2).cuda()
    for original, changed in zip(memory._clean_gen_kv, changed_memory._clean_gen_kv):
        for a, b in zip(original, changed):
            torch.testing.assert_close(a[:, conditions], b[:, conditions], atol=1e-5, rtol=1e-4)
