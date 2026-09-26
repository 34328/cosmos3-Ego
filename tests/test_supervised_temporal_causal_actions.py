"""CPU tests for supervised (joint video-action) temporal-causal action packing."""

import dataclasses

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder
from cosmos_framework.data.generator.sequence_packing.temporal_causal import pack_supertokens_temporal_causal
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.utils.kv_cache import KVTrainMemoryValue, TFNoisyMemoryValue

SPECIAL_TOKENS = {
    "eos_token_id": 1,
    "start_of_generation": 2,
    "end_of_generation": 3,
    "start_of_video": 4,
    "end_of_video": 5,
}
TCF = 4
K = 8
LATENT_C, LATENT_H, LATENT_W, PATCH = 4, 4, 4, 2
ACTION_DIM = 6


def _pack(latent_t, action_rows, timestep=0.5, condition=(0,), supervise=True, action_none=False):
    builder = PackedSequenceBuilder(uses_single_timestep=not isinstance(timestep, torch.Tensor))
    builder._mrope_reset_spatial = True
    builder.begin_sample(0.0)
    vision = torch.randn(1, LATENT_C, latent_t, LATENT_H, LATENT_W)  # [1,C,T,H,W]
    action = None if action_none else torch.randn(action_rows, ACTION_DIM)  # [rows,D]
    pack_supertokens_temporal_causal(
        seq_builder=builder,
        input_vision_tokens=vision,
        input_action_tokens=action,
        condition_frame_indexes_vision=list(condition),
        input_timestep=timestep,
        latent_patch_size=PATCH,
        temporal_compression_factor=TCF,
        action_dim=ACTION_DIM,
        vision_fps=7.5,
        action_fps=15.0,
        enable_fps_modulation=True,
        base_fps=24.0,
        action_tokens_per_latent=K,
        supervise_action_tokens=supervise,
    )
    return builder


@pytest.mark.parametrize("rows_per_state", ["state_group", "null_group"])
def test_supervised_actions_mark_noisy_groups(rows_per_state):
    latent_t = 5
    rows = latent_t * K if rows_per_state == "state_group" else (latent_t - 1) * K
    builder = _pack(latent_t, rows)
    action = builder.action
    mask = action.condition_mask[0].reshape(-1)
    assert mask.shape == (latent_t * K,)
    assert torch.all(mask[:K] == 1) and torch.all(mask[K:] == 0)
    assert action.noisy_frame_indexes[0].tolist() == list(range(K, latent_t * K))
    # Every non-condition action token is supervised, in sequence order.
    assert list(action.mse_loss_indexes) == list(action.sequence_indexes)[K:]
    assert len(action.timesteps) == (latent_t - 1) * K


def test_supervised_action_timesteps_follow_their_frame():
    latent_t = 4
    per_frame = torch.tensor([0.0, 0.2, 0.5, 0.9])
    builder = _pack(latent_t, latent_t * K, timestep=per_frame)
    expected = [float(per_frame[f]) for f in range(1, latent_t) for _ in range(K)]
    assert builder.action.timesteps == pytest.approx(expected)
    # Vision timesteps use the same per-frame values.
    hw = (LATENT_H // PATCH) * (LATENT_W // PATCH)
    assert builder.vision.timesteps == pytest.approx([float(per_frame[f]) for f in range(1, latent_t) for _ in range(hw)])


def test_supervise_false_is_unchanged():
    torch.manual_seed(0)
    a = _pack(3, 3 * K, supervise=False)
    torch.manual_seed(0)
    builder = PackedSequenceBuilder(uses_single_timestep=True)
    builder._mrope_reset_spatial = True
    builder.begin_sample(0.0)
    pack_supertokens_temporal_causal(
        seq_builder=builder,
        input_vision_tokens=torch.randn(1, LATENT_C, 3, LATENT_H, LATENT_W),
        input_action_tokens=torch.randn(3 * K, ACTION_DIM),
        condition_frame_indexes_vision=[0],
        input_timestep=0.5,
        latent_patch_size=PATCH,
        temporal_compression_factor=TCF,
        action_dim=ACTION_DIM,
        vision_fps=7.5,
        action_fps=15.0,
        enable_fps_modulation=True,
        base_fps=24.0,
        action_tokens_per_latent=K,
    )
    assert torch.all(a.action.condition_mask[0] == 1)
    assert list(a.action.mse_loss_indexes) == [] and list(a.action.timesteps) == []
    assert a.action.noisy_frame_indexes == builder.action.noisy_frame_indexes == []
    assert torch.equal(torch.cat(a.position_ids, dim=1), torch.cat(builder.position_ids, dim=1))


def test_supervise_requires_actions_and_clean_null_frame():
    with pytest.raises(ValueError, match="requires pack_action_tokens"):
        _pack(3, 0, action_none=True)
    with pytest.raises(ValueError, match="null action frame 0"):
        _pack(3, 2 * K, condition=())


def test_pack_input_sequence_forwards_supervision_flag():
    latent_t = 3
    plan = SequencePlan(has_text=True, has_vision=True, has_action=True, condition_frame_indexes_vision=[0])
    gen = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, LATENT_C, latent_t, LATENT_H, LATENT_W)],
        x0_tokens_action=[torch.randn(latent_t * K, ACTION_DIM)],
        fps_vision=torch.tensor([7.5]),
        fps_action=torch.tensor([15.0]),
    )
    packed = pack_input_sequence(
        sequence_plans=[plan],
        input_text_indexes=[[30] * 5],
        gen_data_clean=gen,
        input_timesteps=torch.tensor([[0.0, 0.3, 0.7]]),
        special_tokens=SPECIAL_TOKENS,
        latent_patch_size=PATCH,
        enable_fps_modulation=True,
        base_fps=24.0,
        temporal_compression_factor=TCF,
        video_temporal_causal=True,
        action_dim=ACTION_DIM,
        action_tokens_per_latent=K,
        supervise_action_tokens=True,
    )
    assert packed.num_action_tokens_per_supertoken == K
    assert packed.action.timesteps.tolist() == pytest.approx([0.3] * K + [0.7] * K)
    assert packed.action.noisy_frame_indexes[0].tolist() == list(range(K, latent_t * K))
    assert packed.action.mse_loss_indexes.numel() == (latent_t - 1) * K


def test_memory_values_default_to_no_attention_override():
    for cls in (KVTrainMemoryValue, TFNoisyMemoryValue):
        field = {f.name: f for f in dataclasses.fields(cls)}["gen_attention_override"]
        assert field.default is None and field.kw_only
