"""CPU model-hook tests; no pretrained weights or GPU initialization."""

from types import SimpleNamespace
import pytest
import torch

from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos3_joint_video_hand_pose.src.ar_model import ARStepContext, EgoVerseARModel, OmniMoTCausalModel
from cosmos3_joint_video_hand_pose.src.ar_v02_model import EgoVerseARV02Model
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import (
    ACTION,
    CONDITION_VIDEO,
    STATE,
    VIDEO,
    LAYOUT_VERSION,
    JointChunkLayout,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence


def bare_model(c=3, frames=(9, 17)):
    model = object.__new__(EgoVerseARV02Model)
    torch.nn.Module.__init__(model)
    model._ar_step = ARStepContext(chunk_size=c, window=15)
    model._joint_original_frames = list(frames)
    model._joint_layouts = [JointChunkLayout(t, 2, c) for t in frames]
    model._joint_layout = model._joint_layouts[0] if len(frames) == 1 else None
    model._current_hand_visibility = None
    model.whole_action_loss = True
    model.chunk_state_conditioning = True
    model.config = SimpleNamespace(action_tokens_per_latent=8, diffusion_expert_config=SimpleNamespace(patch_spatial=2))
    return model


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_model_prepares_multiple_samples_without_dropping_partial_tail(monkeypatch, c):
    model = bare_model(c)
    layouts = model._joint_layouts
    futures, states, visibility = [], [], []
    for i, lay in enumerate(layouts):
        a = torch.zeros((lay.num_frames - 1) * 8, 64)
        a[:, 0] = torch.arange(1, len(a) + 1) + i * 1000
        s = torch.zeros(lay.num_frames - 1, 64)
        s[:, 9] = torch.arange(len(s)) * 8 + i * 1000
        futures.append(a)
        states.append(s)
        visibility.append(torch.ones(1, len(a), 2, dtype=torch.bool))
    data = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 4, x.num_video_frames, 2, 4) for x in layouts],
        x0_tokens_action=[x.clone() for x in futures],
    )
    result = ([], [SimpleNamespace(), SimpleNamespace()], data, {"skip_text": False}, [], [])
    monkeypatch.setattr(EgoVerseARModel, "_prepare_training_data", lambda *args: result)
    model._prepare_training_data(
        {
            "ar_layout_version": [LAYOUT_VERSION] * 2,
            "ar_boundary_states": [x[None] for x in states],
            "hand_visibility": visibility,
        },
        0,
    )
    for i, lay in enumerate(layouts):
        ar, _, src = lay.action_metadata()
        torch.testing.assert_close(data.x0_tokens_action[i][ar == ACTION], futures[i])
        torch.testing.assert_close(data.x0_tokens_action[i][ar == STATE, 9], src[ar == STATE].float() + i * 1000)
        assert not model._current_hand_visibility[i][ar == STATE].any()
    assert model._joint_layout is None and len(model._joint_layouts) == 2


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("balance", [False, True])
def test_online_vae_receives_independent_overlapping_rgb_chunks(monkeypatch, c, balance):
    model = bare_model(c)
    calls = []

    def encode(raw, *, num_views):
        assert num_views == 1 and not torch.is_grad_enabled()
        calls.append(raw.clone())
        # Shape-only causal codec: a deterministic first frame plus each four-frame endpoint.
        return raw[:, :1, ::4].clone()

    monkeypatch.setattr(model, "_encode_vision_item", encode)
    raw = [
        torch.arange(t).reshape(1, 1, t, 1, 1).expand(1, 3, t, 1, 1).float() + i * 1000 for i, t in enumerate((33, 65))
    ]
    result = model._encode_vision_x0_tokens(raw, None, None, balance_vae_encode=balance)
    assert model._joint_original_frames == [9, 17]
    cursor = 0
    for i, rgb in enumerate(raw):
        expected_parts = []
        for begin in range(0, rgb.shape[2] - 1, 4 * c):
            end = min(begin + 4 * c, rgb.shape[2] - 1)
            torch.testing.assert_close(calls[cursor], rgb[:, :, begin : end + 1])
            expected_parts.append(rgb[:, :1, begin : end + 1 : 4])
            cursor += 1
        torch.testing.assert_close(result[i], torch.cat(expected_parts, dim=2))
    assert cursor == len(calls)


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_video_and_action_sigma_ownership_in_mixed_length_pack(monkeypatch, c):
    model = bare_model(c)
    n = max(len(x.boundaries) for x in model._joint_layouts) + 1

    def sample_video(self, batch_size, **kwargs):
        assert batch_size == 2 * n
        sigma = torch.arange(1, batch_size + 1).float().reshape(-1, 1) / (batch_size + 2)
        return sigma * 1000, sigma

    def sample_action(self, batch_size, **kwargs):
        assert batch_size == 2 * n
        sigma = 1 - torch.arange(1, batch_size + 1).float().reshape(-1, 1) / (batch_size + 3)
        return sigma * 1000, sigma

    monkeypatch.setattr(OmniMoTCausalModel, "_get_train_noise_level_vision", sample_video)
    monkeypatch.setattr(OmniMoTCausalModel, "_get_train_noise_level_action", sample_action)
    lengths = [x.num_video_frames for x in model._joint_layouts]
    ts, sg = model._get_train_noise_level_vision(2, False, lengths, resolutions=["480"] * 2)
    model._get_train_noise_level_action(2, iteration=0)
    expected_rows = []
    for i, lay in enumerate(model._joint_layouts):
        vr, vc, _ = lay.video_metadata()
        ar, ac, _ = lay.action_metadata()
        assert (sg[i, : len(vr)][vr == CONDITION_VIDEO] == 0).all()
        expected = (i * n + vc[vr == VIDEO] + 1).float() / (2 * n + 2)
        torch.testing.assert_close(sg[i, : len(vr)][vr == VIDEO], expected)
        torch.testing.assert_close(ts[i], sg[i] * 1000)
        assert not sg[i, len(vr) :].any()
        action_sigma = 1 - (i * n + ac + 1).float() / (2 * n + 3)
        expected_rows.append(torch.where(ar == STATE, 0, action_sigma))
    captured = {}

    def add_noise(self, data, pack, sigmas, **kwargs):
        captured.update(kwargs)
        return "noised"

    monkeypatch.setattr(OmniMoTCausalModel, "_add_noise_to_input", add_noise)
    model.rectified_flow_action = SimpleNamespace(
        noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    )
    indexes = [torch.where(x.action_metadata()[0] == ACTION)[0] for x in model._joint_layouts]
    packed = SimpleNamespace(action=SimpleNamespace(noisy_frame_indexes=indexes))
    assert model._add_noise_to_input(None, packed, sg) == "noised"
    for actual, expected in zip(captured["sigmas_action"], expected_rows):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        packed.action.timesteps, torch.cat([x[idx] * 1000 for x, idx in zip(expected_rows, indexes)])
    )
    assert packed.uses_single_timestep is False


def test_model_rejects_legacy_dataset_before_encoding():
    model = bare_model()
    with pytest.raises(ValueError, match="joint_chunk_cond_v1"):
        model._prepare_training_data({"ar_layout_version": "joint_state_single_v1"}, 0)


def test_model_action_loss_masks_conditions_and_padding():
    model = bare_model(frames=(9,))
    model._current_hand_visibility = [torch.ones(3, 2, dtype=torch.bool)]
    prediction = torch.zeros(3, 64, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0] = 999
    target[1:, 18:33] = 2
    target[:, 57:] = 999
    loss, _ = model._compute_flow_matching_loss(
        [prediction],
        [target],
        [torch.tensor([[1.0], [0.0], [0.0]])],
        torch.zeros(1, 3),
        True,
        SimpleNamespace(),
        raw_action_dim=[57],
    )
    torch.testing.assert_close(loss, torch.tensor(60 / 57))
    loss.backward()
    assert prediction.grad[0].count_nonzero() == prediction.grad[:, 57:].count_nonzero() == 0


def test_state_type_embedding_has_gradient_only_on_state_rows():
    layout = JointChunkLayout(3, 1, 1)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, layout.num_video_frames, 2, 2)],
        x0_tokens_action=[torch.zeros(layout.num_action_rows, 64)],
        fps_vision=torch.tensor([7.5]),
        fps_action=torch.tensor([15.0]),
    )
    pack = pack_joint_sequence(
        layout=layout,
        gen_data_clean=data,
        text_ids=[1],
        special_tokens={"eos_token_id": 2, "start_of_generation": 3},
        timesteps=0,
        latent_patch_size=2,
        condition_frames=range(layout.num_video_frames),
    )
    net = object.__new__(Cosmos3VFMNetwork)
    torch.nn.Module.__init__(net)
    net.config = SimpleNamespace(enable_action_modality_embedding=False, enable_action_state_embedding=True)
    net.action_state_embed = torch.nn.Parameter(torch.arange(4).float())
    net.pack_action = lambda *args: (torch.zeros(layout.num_action_rows, 64), None)
    net.action2llm = lambda x, _: x[:, :4]
    buffer = torch.zeros(pack.sequence_length, 4)
    net._encode_action(pack, buffer, torch.float32)
    encoded = buffer[pack.action.sequence_indexes]
    torch.testing.assert_close(encoded[pack.action_state_mask], net.action_state_embed[None].expand(2, -1))
    assert encoded[~pack.action_state_mask].count_nonzero() == 0
    encoded.sum().backward()
    torch.testing.assert_close(net.action_state_embed.grad, torch.full((4,), 2.0))


@pytest.mark.parametrize("sample_count", [1, 2, 3])
def test_replay_adapter_preserves_text_boundaries_and_clean_gen_gradient(monkeypatch, sample_count):
    from cosmos_framework.model.generator.utils.kv_cache import DualKVCache, TeacherForcingMemoryState

    model = bare_model(c=2, frames=tuple(3 + i for i in range(sample_count)))
    layouts = model._joint_layouts
    text_lengths = [3 + i for i in range(sample_count)]
    packed = SimpleNamespace(
        joint_layouts=layouts,
        joint_text_lengths=text_lengths,
        vision=SimpleNamespace(tokens=[torch.zeros(1)]),
    )

    def parent(self, packed_sequence, memory_info, **kwargs):
        return TeacherForcingMemoryState(
            [(x.num_video_frames, 1, 2) for x in layouts],
            0,
            False,
            0,
            [DualKVCache(gen_cache_size=2)],
            1,
            2,
            detach_clean_kv=False,
        )

    monkeypatch.setattr(OmniMoTCausalModel, "_build_tf_memory_state", parent)
    memory = model._build_tf_memory_state(packed, {})
    nt, ng = sum(text_lengths), sum(x.num_tokens for x in layouts)
    hidden = {"causal_seq": torch.zeros(nt + 5, 2), "_num_causal_tokens": nt, "_num_full_tokens": ng}
    memory.init(hidden, torch.device("cpu"))
    expected_offsets = torch.tensor([0] + list(torch.tensor(text_lengths).cumsum(0).tolist()), dtype=torch.int32)
    torch.testing.assert_close(memory.und_kv_offsets, expected_offsets)
    text = torch.zeros(1, nt, 1, 2)
    # Unique GEN tags detect any mistaken global-sequence/text-offset indexing.
    keys = torch.arange(ng * 2).float().reshape(1, ng, 1, 2).requires_grad_()
    values = (keys.detach() + 0.5).requires_grad_()
    memory.write_for_layer(0, (keys, values, text, text))
    memory.pass_number = 2
    memory.init(hidden, torch.device("cpu"))
    read = memory.read_for_layer(0)
    torch.testing.assert_close(read.und_kv_offsets, expected_offsets)
    torch.testing.assert_close(read.cached_clean_gen_k, keys)
    torch.testing.assert_close(read.cached_clean_gen_v, values)
    assert read.gen_attention_override.gen_len == ng
    memory.write_for_layer(0, (torch.zeros_like(keys), torch.zeros_like(values), text, text))
    torch.testing.assert_close(memory.read_for_layer(0).cached_clean_gen_k, keys)
    (read.cached_clean_gen_k.sum() + read.cached_clean_gen_v.sum()).backward()
    torch.testing.assert_close(keys.grad, torch.ones_like(keys))
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_vision_condition_embedding_affects_only_U_and_receives_gradient(monkeypatch):
    layout = JointChunkLayout(4, 1, 2)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, layout.num_video_frames, 2, 2)],
        x0_tokens_action=[torch.zeros(layout.num_action_rows, 64)],
        fps_vision=torch.tensor([7.5]),
        fps_action=torch.tensor([15.0]),
    )
    pack = pack_joint_sequence(
        layout=layout,
        gen_data_clean=data,
        text_ids=[1],
        special_tokens={"eos_token_id": 2, "start_of_generation": 3},
        timesteps=0,
        latent_patch_size=2,
    )
    net = object.__new__(Cosmos3VFMNetwork)
    torch.nn.Module.__init__(net)
    net.config = SimpleNamespace(
        enable_vision_condition_embedding=True,
        enable_vision_modality_embeddings=False,
        enable_media_modality_embedding=False,
    )
    net.vision_condition_embed = torch.nn.Parameter(torch.arange(4).float())
    net.vae2llm = torch.nn.Identity()
    net.latent_channel = 1
    monkeypatch.setattr(net, "_encode_grid_stream", lambda *args, **kwargs: pack.vision.token_shapes)
    buffer = torch.zeros(pack.sequence_length, 4)
    net._encode_vision(pack, buffer, torch.float32)
    rows = buffer[pack.vision.sequence_indexes]
    mask = pack.vision_condition_type_mask
    torch.testing.assert_close(rows[mask], net.vision_condition_embed[None].expand(int(mask.sum()), -1))
    assert rows[~mask].count_nonzero() == 0
    rows.sum().backward()
    torch.testing.assert_close(net.vision_condition_embed.grad, torch.full((4,), float(mask.sum())))
