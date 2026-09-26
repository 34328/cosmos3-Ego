"""CPU tests for decoupling action tokens per latent (K) from the VAE tcf."""

import dataclasses

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.autoregressive import (
    pack_input_sequence_autoregressive,
    pack_input_sequence_autoregressive_batch,
)
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder
from cosmos_framework.data.generator.sequence_packing.temporal_causal import pack_supertokens_temporal_causal
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

SPECIAL_TOKENS = {
    "eos_token_id": 1,
    "start_of_generation": 2,
    "end_of_generation": 3,
    "start_of_video": 4,
    "end_of_video": 5,
}
TCF = 4
LATENT_C, LATENT_H, LATENT_W, PATCH = 4, 4, 4, 2
PATCHES_PER_FRAME = (LATENT_H // PATCH) * (LATENT_W // PATCH)
ACTION_DIM = 6
BASE_FPS = 24.0


def _assert_same(a, b, path="root"):
    """Element-wise equality over (nested) dataclasses, tensors, lists and dicts."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), path
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b), path
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), path
        for field in dataclasses.fields(a):
            _assert_same(getattr(a, field.name), getattr(b, field.name), f"{path}.{field.name}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            _assert_same(a[key], b[key], f"{path}[{key!r}]")
    else:
        assert a == b, (path, a, b)


def _vision(latent_t):
    return torch.randn(1, LATENT_C, latent_t, LATENT_H, LATENT_W)  # [1,C,T,H,W]


def _pack_tc(vision, action, vision_fps=None, action_fps=None, fps_mod=False, offset=0.0, **kwargs):
    builder = PackedSequenceBuilder(uses_single_timestep=True)
    builder._mrope_reset_spatial = True
    builder.begin_sample(offset)
    result = pack_supertokens_temporal_causal(
        seq_builder=builder,
        input_vision_tokens=vision,
        input_action_tokens=action,
        condition_frame_indexes_vision=[0],
        input_timestep=0.5,
        latent_patch_size=PATCH,
        temporal_compression_factor=TCF,
        action_dim=ACTION_DIM,
        vision_fps=vision_fps,
        action_fps=action_fps,
        enable_fps_modulation=fps_mod,
        base_fps=BASE_FPS,
        **kwargs,
    )
    return builder, result


def _temporal_layout(builder, latent_t, k):
    """Per-frame temporal positions: actions [T,K] and vision [T] (first vision token)."""
    positions = torch.cat(builder.position_ids, dim=1)[0]  # [N]
    action_idx = torch.tensor(builder.action.sequence_indexes)
    vision_idx = torch.tensor(builder.vision.sequence_indexes)
    action_t = positions[action_idx].reshape(latent_t, k)  # [T,K]
    vision_t = positions[vision_idx].reshape(latent_t, PATCHES_PER_FRAME)  # [T,HW]
    assert torch.all(vision_t == vision_t[:, :1])
    return action_t, vision_t[:, 0]


def _gen_data(vision, action, fps_vision, fps_action):
    return GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[vision],
        x0_tokens_action=[action] if action is not None else None,
        fps_vision=torch.tensor([fps_vision]),
        fps_action=torch.tensor([fps_action]) if fps_action is not None else None,
    )


def _pack_full(vision, action, fps_vision=30.0, fps_action=30.0, **kwargs):
    plan = SequencePlan(has_text=True, has_vision=True, has_action=True, condition_frame_indexes_vision=[0])
    return pack_input_sequence(
        sequence_plans=[plan],
        input_text_indexes=[[30] * 5],
        gen_data_clean=_gen_data(vision, action, fps_vision, fps_action),
        input_timesteps=torch.tensor([0.5]),
        special_tokens=SPECIAL_TOKENS,
        latent_patch_size=PATCH,
        enable_fps_modulation=True,
        base_fps=BASE_FPS,
        temporal_compression_factor=TCF,
        video_temporal_causal=True,
        action_dim=ACTION_DIM,
        **kwargs,
    )


# --------------------------------------------------------------------- (a)
@pytest.mark.parametrize("fps_mod", [False, True])
@pytest.mark.parametrize("layout", ["whole_clip", "ar_chunk", "null_only"])
def test_default_k_matches_legacy_supertoken_pack(fps_mod, layout):
    torch.manual_seed(0)
    latent_t = 3
    vision = _vision(latent_t)
    rows = {"whole_clip": (latent_t - 1) * TCF, "ar_chunk": latent_t * TCF, "null_only": None}[layout]
    action = torch.randn(rows, ACTION_DIM) if rows is not None else None
    fps = dict(vision_fps=30.0, action_fps=30.0, fps_mod=fps_mod)
    legacy_builder, legacy_result = _pack_tc(vision, action, **fps)
    new_builder, new_result = _pack_tc(vision, action, action_tokens_per_latent=None, **fps)
    assert legacy_result == new_result
    _assert_same(legacy_builder, new_builder)
    assert legacy_builder.action.token_shapes == [(latent_t * TCF,)]


def test_default_k_matches_legacy_pack_input_sequence():
    torch.manual_seed(0)
    latent_t = 3
    vision = _vision(latent_t)
    action = torch.randn((latent_t - 1) * TCF, ACTION_DIM)
    legacy = _pack_full(vision, action)
    new = _pack_full(vision, action, action_tokens_per_latent=None)
    _assert_same(legacy, new)
    assert legacy.num_action_tokens_per_supertoken == TCF


def test_explicit_k_equal_to_tcf_matches_legacy_pack_input_sequence():
    torch.manual_seed(0)
    vision = _vision(3)
    action = torch.randn(2 * TCF, ACTION_DIM)
    _assert_same(_pack_full(vision, action), _pack_full(vision, action, action_tokens_per_latent=TCF))


# --------------------------------------------------------------------- (b)
def test_k8_supertoken_layout_and_null_prefix():
    k, latent_t = 8, 3
    vision = _vision(latent_t)
    action = torch.randn((latent_t - 1) * k, ACTION_DIM)
    packed = _pack_full(vision, action, fps_vision=15.0, fps_action=30.0, action_tokens_per_latent=k)

    assert packed.num_action_tokens_per_supertoken == k
    assert packed.null_action_supertokens is True
    tokens = packed.action.tokens[0]  # [T*K,D]
    assert tuple(tokens.shape) == (latent_t * k, ACTION_DIM)
    assert torch.equal(tokens[:k], torch.zeros(k, ACTION_DIM))  # null group for frame 0
    assert torch.equal(tokens[k:], action)
    assert packed.action.token_shapes == [(latent_t * k,)]
    assert tuple(packed.action.condition_mask[0].shape) == (latent_t * k, 1)
    assert len(packed.action.sequence_indexes) == latent_t * k
    # Every supertoken is [K action, H*W vision].
    vision_indexes = torch.as_tensor(packed.vision.sequence_indexes).tolist()
    action_indexes = torch.as_tensor(packed.action.sequence_indexes).tolist()
    for frame in range(latent_t):
        group = action_indexes[frame * k : (frame + 1) * k]
        first_vision = vision_indexes[frame * PATCHES_PER_FRAME]
        assert group == list(range(first_vision - k, first_vision))


def test_k8_ar_chunk_has_no_null_prefix():
    k, latent_t = 8, 2
    builder, (split_len, null_flag) = _pack_tc(
        _vision(latent_t), torch.randn(latent_t * k, ACTION_DIM), action_tokens_per_latent=k
    )
    assert null_flag is False
    assert split_len == latent_t * (k + PATCHES_PER_FRAME)
    assert builder.action.token_shapes == [(latent_t * k,)]


def test_k8_all_null_layout():
    k, latent_t = 8, 3
    builder, (split_len, null_flag) = _pack_tc(_vision(latent_t), None, action_tokens_per_latent=k)
    assert null_flag is True
    assert split_len == latent_t * (k + PATCHES_PER_FRAME)
    assert torch.equal(builder.action.tokens[0], torch.zeros(latent_t * k, ACTION_DIM))


@pytest.mark.parametrize("rows", [2 * TCF, 3 * TCF, 2 * 8 + 1])
def test_k8_rejects_wrong_action_length(rows):
    with pytest.raises(ValueError, match=r"latent_t\*K rows.*K\(action_tokens_per_latent\)=8"):
        _pack_tc(_vision(3), torch.randn(rows, ACTION_DIM), action_tokens_per_latent=8)


def test_invalid_k_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        _pack_tc(_vision(2), None, action_tokens_per_latent=0)


# --------------------------------------------------------------------- (c)
@pytest.mark.parametrize("layout", ["whole_clip", "ar_chunk"])
def test_k8_mrope_last_action_aligns_with_video_latent(layout):
    k, latent_t = 8, 4
    real_frames = latent_t - 1 if layout == "whole_clip" else latent_t
    vision = _vision(latent_t)
    action = torch.randn(real_frames * k, ACTION_DIM)

    def layout_for(scale):
        builder, _ = _pack_tc(
            vision,
            action,
            vision_fps=7.5 * scale,
            action_fps=15.0 * scale,
            fps_mod=True,
            action_tokens_per_latent=k,
        )
        return _temporal_layout(builder, latent_t, k)

    action_t, vision_t = layout_for(1.0)
    real_groups = range(1, latent_t) if layout == "whole_clip" else range(latent_t)
    for frame in real_groups:
        group = action_t[frame]
        torch.testing.assert_close(group[-1], vision_t[frame], rtol=0, atol=1e-5)
        assert torch.all(group[1:] > group[:-1])
    if layout == "whole_clip":
        # Null conditioning group sits at the sample's temporal offset (0 here).
        assert torch.all(action_t[0] == 0)
    # Latent stride is base_fps / fps_video; the whole-clip vision frame 0 starts at 0.
    stride = BASE_FPS / 7.5
    vision_start = 0 if layout == "whole_clip" else 1
    torch.testing.assert_close(vision_t, (torch.arange(latent_t) + vision_start) * stride, rtol=0, atol=1e-5)

    action_half, vision_half = layout_for(0.5)
    torch.testing.assert_close(action_half, 2 * action_t, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(vision_half, 2 * vision_t, rtol=1e-6, atol=1e-5)


# --------------------------------------------------------------------- (d)
def test_k8_rejects_inconsistent_fps():
    with pytest.raises(ValueError, match=r"action_fps == vision_fps \* K / tcf"):
        _pack_tc(
            _vision(3),
            torch.randn(2 * 8, ACTION_DIM),
            vision_fps=7.5,
            action_fps=7.5,
            fps_mod=True,
            action_tokens_per_latent=8,
        )


def test_fps_constraint_not_applied_when_k_equals_tcf():
    # Legacy behavior: K == tcf never validates the FPS pair.
    _pack_tc(_vision(2), torch.randn(TCF, ACTION_DIM), vision_fps=30.0, action_fps=10.0, fps_mod=True)


# --------------------------------------------------------------------- (e)
def _pack_ar(vision, action, k, **kwargs):
    return pack_input_sequence_autoregressive(
        vision_latent=vision,
        action_latent=action,
        text_tokens=None,
        timestep=0.0,
        fps_vision=[7.5],
        fps_action=[15.0] if action is not None else None,
        special_tokens=SPECIAL_TOKENS,
        latent_patch_size=PATCH,
        temporal_compression_factor=TCF,
        video_temporal_causal=True,
        action_dim=ACTION_DIM,
        base_fps=BASE_FPS,
        cached_text_offset=0,
        action_tokens_per_latent=k,
        **kwargs,
    )


def test_autoregressive_pack_k8_chunk_matches_whole_clip_positions():
    k, latent_t, chunk_start, chunk_len = 8, 5, 2, 2
    vision = _vision(latent_t)
    actions = torch.randn((latent_t - 1) * k, ACTION_DIM)  # a_0..a_{T-2}
    train_builder, _ = _pack_tc(
        vision, actions, vision_fps=7.5, action_fps=15.0, fps_mod=True, action_tokens_per_latent=k
    )
    train_action_t, train_vision_t = _temporal_layout(train_builder, latent_t, k)

    chunk_actions = actions[(chunk_start - 1) * k : (chunk_start + chunk_len - 1) * k]  # [chunk_len*K,D]
    packed = _pack_ar(
        vision[:, :, chunk_start : chunk_start + chunk_len],
        chunk_actions,
        k,
        frame_idx=chunk_start,
        action_domain_id=torch.arange(chunk_len * k),
    )
    assert packed.num_action_tokens_per_supertoken == k
    assert packed.null_action_supertokens is False
    assert packed.action.token_shapes == [(chunk_len * k,)]
    positions = packed.position_ids[0]
    ar_action_t = positions[torch.as_tensor(packed.action.sequence_indexes)].reshape(chunk_len, k)
    ar_vision_t = positions[torch.as_tensor(packed.vision.sequence_indexes)].reshape(chunk_len, -1)[:, 0]
    frames = slice(chunk_start, chunk_start + chunk_len)
    torch.testing.assert_close(ar_action_t, train_action_t[frames], rtol=0, atol=1e-4)
    torch.testing.assert_close(ar_vision_t, train_vision_t[frames], rtol=0, atol=1e-4)


def test_autoregressive_pack_k8_null_frame_and_domain_ids():
    k = 8
    packed = _pack_ar(_vision(1), None, k, force_action_tokens=True, action_domain_id=torch.tensor([3]))
    assert packed.num_action_tokens_per_supertoken == k
    assert packed.null_action_supertokens is True
    assert packed.action.token_shapes == [(k,)]
    assert torch.equal(packed.action.tokens[0], torch.zeros(k, ACTION_DIM))
    # Per-token domain IDs are counted in K tokens per latent, not tcf.
    with pytest.raises(ValueError, match="one ID per packed action token"):
        _pack_ar(_vision(1), None, k, force_action_tokens=True, action_domain_id=torch.zeros(TCF))


def test_autoregressive_pack_default_k_matches_legacy():
    torch.manual_seed(0)
    vision = _vision(2)
    action = torch.randn(2 * TCF, ACTION_DIM)
    kwargs = dict(
        vision_latent=vision,
        action_latent=action,
        text_tokens=None,
        timestep=0.0,
        fps_vision=[30.0],
        fps_action=[30.0],
        special_tokens=SPECIAL_TOKENS,
        latent_patch_size=PATCH,
        frame_idx=3,
        temporal_compression_factor=TCF,
        video_temporal_causal=True,
        action_dim=ACTION_DIM,
        cached_text_offset=7,
    )
    _assert_same(
        pack_input_sequence_autoregressive(**kwargs),
        pack_input_sequence_autoregressive(**kwargs, action_tokens_per_latent=None),
    )


def test_autoregressive_batch_pack_accepts_k():
    kwargs = dict(
        vision_latent=torch.randn(2, LATENT_C, 1, LATENT_H, LATENT_W),
        text_tokens=None,
        timestep=0.0,
        fps_vision=[7.5, 7.5],
        special_tokens=SPECIAL_TOKENS,
        latent_patch_size=PATCH,
        frame_idx=[1, 2],
        temporal_compression_factor=TCF,
        cached_text_offsets=[3, 4],
    )
    legacy = pack_input_sequence_autoregressive_batch(**kwargs)
    with_k = pack_input_sequence_autoregressive_batch(**kwargs, action_tokens_per_latent=8)
    # Action-free batch packs are unaffected by K.
    _assert_same(legacy, with_k)
    assert with_k.num_action_tokens_per_supertoken == 0


# ------------------------------------------------------------ model helpers
def _causal_model_cls():
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    return OmniMoTCausalModel


def _fake_causal_model(k, chunk=2):
    from types import SimpleNamespace

    cls = _causal_model_cls()
    fake = SimpleNamespace(
        config=SimpleNamespace(
            teacher_forcing_frames_per_chunk=chunk,
            causal_training_strategy="teacher_forcing",
            action_tokens_per_latent=k,
        ),
        tokenizer_vision_gen=SimpleNamespace(
            temporal_compression_factor=TCF, get_pixel_num_frames=lambda t: 1 + TCF * (t - 1)
        ),
    )
    fake._is_chunkwise_tf = lambda: cls._is_chunkwise_tf(fake)
    return fake


@pytest.mark.parametrize("k", [None, 8])
def test_chunkwise_tf_truncation_counts_actions_in_k(k):
    cls = _causal_model_cls()
    per_latent = TCF if k is None else k
    latent_t = 6  # chunk=2 keeps 1 + 2 * ((6 - 1) // 2) = 5 latent frames
    gen_data_clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[_vision(latent_t)],
        x0_tokens_action=[torch.randn((latent_t - 1) * per_latent, ACTION_DIM)],
    )
    out = cls._truncate_for_chunkwise_tf(_fake_causal_model(k), gen_data_clean)
    assert out.x0_tokens_vision[0].shape[2] == 5
    assert out.x0_tokens_action[0].shape[0] == (5 - 1) * per_latent


def test_chunkwise_tf_truncation_rejects_rows_not_multiple_of_k():
    cls = _causal_model_cls()
    gen_data_clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[_vision(6)],
        x0_tokens_action=[torch.randn(5 * TCF, ACTION_DIM)],  # 20 rows: multiple of tcf, not of K=8
    )
    with pytest.raises(ValueError, match="action_tokens_per_latent=8"):
        cls._truncate_for_chunkwise_tf(_fake_causal_model(8), gen_data_clean)


def test_resolve_fps_action_list():
    resolve = _causal_model_cls()._resolve_fps_action_list
    fps_vision = [15.0, 7.5]
    assert resolve({"conditioning_fps_action": torch.tensor([1.0, 2.0])}, fps_vision, TCF, None, 24.0) == [24.0, 24.0]
    assert resolve({}, fps_vision, TCF, 8, 24.0) == [30.0, 15.0]
    assert resolve({"conditioning_fps_action": torch.tensor([30.0, 15.0])}, fps_vision, TCF, 8, 24.0) == [30.0, 15.0]
    assert resolve({"conditioning_fps_action": [torch.tensor(30.0), torch.tensor(15.0)]}, fps_vision, TCF, 8, 24.0) == [
        30.0,
        15.0,
    ]
