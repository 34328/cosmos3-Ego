"""CPU integration tests driving the real greedy loader and collate loop."""

import random

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos3_joint_video_hand_pose.src.ar_dataset import (
    AR_V02_TOKEN_BUDGET_VERSION,
    ar_v02_token_count,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_dataloader import (
    JointChunkPackingDataLoader,
    joint_training_token_budget,
    joint_pack_padding_audit,
    JOINT_PACK_PADDING_RESERVE,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import LAYOUT_VERSION


class Clips(Dataset):
    def __init__(self, lengths):
        self.lengths = lengths
        self.reads = []

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        self.reads.append(index)
        frames = self.lengths[index]
        return dict(
            video=torch.tensor(index, dtype=torch.uint8).expand(3, frames, 368, 640),
            action=torch.full((2 * (frames - 1), 64), float(index)),
            text_token_ids=torch.arange(20),
            ar_boundary_states=torch.zeros((frames - 1) // 4, 64),
            ar_layout_version=LAYOUT_VERSION,
            ar_token_budget_version=AR_V02_TOKEN_BUDGET_VERSION,
            ar_num_tokens=joint_training_token_budget(20, frames, 368, 640),
            dataset_index=index,
        )


def loader(lengths, **kwargs):
    clips = Clips(lengths)
    inner = DataLoader(clips, batch_size=1, num_workers=0, collate_fn=custom_collate_fn)
    options = dict(
        dataloader=inner,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=70000,
        max_samples_per_batch=None,
        prewarm=False,
        lazy_initialize_child_iterators=True,
    )
    options.update(kwargs)
    return JointChunkPackingDataLoader(**options), clips


def ids(batch):
    return [int(x) for x in batch["dataset_index"]]


def verify_batch(batch, lengths):
    indexes = ids(batch)
    assert batch["_num_tokens"] == sum(joint_training_token_budget(20, lengths[i], 368, 640) for i in indexes)
    for j, index in enumerate(indexes):
        # Official ragged list[list[Tensor]] output, usable by the real model.
        rgb, action = batch["video"][j][0], batch["action"][j][0]
        assert rgb.shape == (3, lengths[index], 368, 640)
        assert int(rgb[0, 0, 0, 0]) == index
        assert action.shape == (2 * (lengths[index] - 1), 64)
        assert torch.all(action == index)
        assert batch["ar_boundary_states"][j].shape == (1, (lengths[index] - 1) // 4, 64)


@pytest.mark.parametrize("ceiling", [2, 3, 4])
@pytest.mark.parametrize("prewarm", [False, True])
def test_sample_ceiling_stops_real_reads_without_lookahead(ceiling, prewarm):
    lengths = [33, 65, 33, 65, 33, 65, 33, 33]
    packed, clips = loader(lengths, joint_max_samples=ceiling, prewarm=prewarm)
    iterator = iter(packed)
    first = next(iterator)
    assert ids(first) == list(range(ceiling))
    assert clips.reads == list(range(ceiling))  # No extra ten decoded candidates.
    assert not packed.buffers[0]
    batches = [first, *iterator]
    assert [i for batch in batches for i in ids(batch)] == list(range(len(lengths)))
    assert all(len(ids(batch)) <= ceiling for batch in batches)
    for batch in batches:
        verify_batch(batch, lengths)


def test_two_t129_fit_but_no_third_and_mixed_four_fit():
    lengths = [129, 129, 129, 65, 33, 33]
    packed, _ = loader(lengths)
    batches = list(packed)
    assert [ids(batch) for batch in batches] == [[0, 1], [2, 3, 4, 5]]
    for batch in batches:
        assert batch["_num_tokens"] < 70000
        verify_batch(batch, lengths)


@pytest.mark.parametrize(
    "lookahead, expected",
    [
        (1, [[0], [1, 2], [3, 4]]),
        (2, [[0, 2], [1, 3], [4]]),
    ],
)
def test_lookahead_backfills_short_clips_and_preserves_skipped_order(lookahead, expected):
    lengths = [129, 129, 33, 33, 65]
    packed, clips = loader(lengths, max_sequence_length=40000, lookahead_limit=lookahead)
    batches = list(packed)
    assert [ids(batch) for batch in batches] == expected
    assert sorted(i for batch in batches for i in ids(batch)) == list(range(len(lengths)))
    assert clips.reads == list(range(len(lengths)))
    for batch in batches:
        verify_batch(batch, lengths)


def test_fit_is_pure_and_exact_cap_is_excluded():
    packed, _ = loader([33])
    candidate = dict(
        num_tokens=100, sample_seconds=0.0, packed_tokens=200, packed_sample_seconds=0.0, batch_started=True
    )
    assert all(packed._sample_fits(**candidate) for _ in range(20))
    candidate["packed_tokens"] = 69900 - JOINT_PACK_PADDING_RESERVE
    assert not packed._sample_fits(**candidate)
    candidate["packed_tokens"] = 69899 - JOINT_PACK_PADDING_RESERVE
    assert packed._sample_fits(**candidate)
    assert ids(next(iter(packed))) == [0]


def test_oversized_sample_dropped_and_finite_stream_drains():
    packed, _ = loader([129, 33, 65], max_sequence_length=20000)
    batches = list(packed)
    assert [ids(b) for b in batches] == [[1], [2]]
    assert sum(b["_dropped_count"] for b in batches) == 1
    packed, _ = loader([129], max_sequence_length=20000)
    assert list(packed) == []


def test_native_sample_mode_honors_smaller_requested_cap():
    packed, _ = loader([33] * 5, max_sequence_length=None, max_samples_per_batch=2, joint_max_samples=4)
    assert [len(ids(b)) for b in packed] == [2, 2, 1]
    with pytest.raises(AssertionError, match="Exactly one"):
        loader([33], max_samples_per_batch=2)


def test_time_ceiling_still_limits_real_mixed_packs():
    class Budget:
        def sample_seconds(self, und, gen):
            return 1.0

        def has_room_for(self, current, candidate):
            return current + candidate <= 2.0

        def projected_seconds(self, seconds):
            return seconds

    packed, _ = loader([33, 65, 33, 65, 33])
    packed.iteration_time_budget = Budget()
    batches = list(packed)
    assert [len(ids(b)) for b in batches] == [2, 2, 1]
    assert [b["_projected_iteration_ms"] for b in batches] == [2000, 2000, 1000]


@pytest.mark.parametrize("frames", [33, 65, 129])
def test_budget_matches_dataset_and_counts_both_passes(frames):
    assert joint_training_token_budget(101, frames, 368, 640) == ar_v02_token_count(101, frames)
    n = (frames - 1) // 4
    assert joint_training_token_budget(101, frames, 368, 640) == 2 * (103 + n * (480 + 9))


@pytest.mark.parametrize(
    "patch",
    [
        {"ar_layout_version": "joint_state_single_v1"},
        {"ar_num_tokens": 1},
        {"ar_token_budget_version": "single_pass"},
        {"action": torch.zeros(72, 64)},
        {"video": torch.zeros(2, 3, 33, 1, 1)},
        {"text_token_ids": torch.zeros(2, 20, dtype=torch.long)},
    ],
)
def test_malformed_or_stale_sample_fails_before_admission(patch):
    packed, clips = loader([33])
    sample = clips[0]
    sample.update(patch)
    with pytest.raises(ValueError):
        packed._compute_token_split_per_sample(sample)


@pytest.mark.parametrize("ceiling", [0, -1, 2.5, True])
def test_invalid_sample_limit_rejected(ceiling):
    with pytest.raises(ValueError):
        loader([33], joint_max_samples=ceiling)


def test_invalid_geometry_and_lookahead_rejected():
    for args in [(-1, 33, 368, 640), (20, 34, 368, 640), (20, 33, 0, 640), (20, 33.0, 368, 640)]:
        with pytest.raises(ValueError):
            joint_training_token_budget(*args)
    with pytest.raises(ValueError, match="lookahead"):
        loader([33], lookahead_limit=0)
    with pytest.raises(ValueError, match="temporal compression"):
        loader([33], tokenizer_temporal_compression_factor=8)


def test_long_random_mix_never_loses_or_duplicates_samples():
    rng = random.Random(42)
    lengths = [rng.choice([33, 65, 129]) for _ in range(79)]
    packed, clips = loader(lengths, lookahead_limit=3)
    batches = list(packed)
    assert sorted(i for b in batches for i in ids(b)) == list(range(len(lengths)))
    assert clips.reads == list(range(len(lengths)))
    assert all(1 <= len(ids(b)) <= 4 and b["_num_tokens"] < 70000 for b in batches)
    for batch in batches:
        verify_batch(batch, lengths)


def test_padding_reserve_is_once_per_pack_and_changes_only_admission():
    raw_pair = 2 * joint_training_token_budget(20, 33, 368, 640)
    # Raw rows fit, but padding-inclusive admission must reject a second clip.
    packed, _ = loader([33, 33], max_sequence_length=raw_pair + 300)
    assert [len(ids(b)) for b in packed] == [1, 1]
    packed, _ = loader([33, 33], max_sequence_length=raw_pair + JOINT_PACK_PADDING_RESERVE + 1)
    batches = list(packed)
    assert len(batches) == 1 and ids(batches[0]) == [0, 1]
    batch = batches[0]
    assert batch["_num_tokens"] == batch["_ar_c1_raw_forward_tokens"] == raw_pair
    assert batch["_ar_c1_reserved_budget_tokens"] == raw_pair + JOINT_PACK_PADDING_RESERVE
    assert raw_pair < batch["_ar_c1_attention_padded_capacity"] <= batch["_ar_c1_reserved_budget_tokens"]
    assert batch["_ar_pack_padding_reserve"] == 512


def test_508_does_not_cover_runtime_sentinel_before_flex_rounding():
    audit = joint_pack_padding_audit(128, 129)
    assert audit["padding_overhead"] == 510
    assert audit["padding_overhead"] > 508
    # Cover all text/GEN alignment residues, including already-aligned streams.
    for text in range(1, 129):
        for gen in range(1, 129):
            row = joint_pack_padding_audit(text, gen)
            assert row["padding_overhead"] <= JOINT_PACK_PADDING_RESERVE


@pytest.mark.parametrize("text,latent_frames,patches", [(128, 2, 60), (127, 129, 1)])
def test_audit_matches_runtime_storage_and_flex_wrapper_shapes_cpu(monkeypatch, text, latent_frames, patches):
    from cosmos_framework.data.generator.sequence_packing.runtime import sequence_pack_from_packed_sequence
    from cosmos_framework.model.generator.mot import flex_attention
    from cosmos_framework.model.generator.utils.kv_cache import TFNoisyMemoryValue, TFReplayCleanMemoryValue
    from cosmos3_joint_video_hand_pose.src.ar_v02_attention import JointTeacherForcingAttention
    from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout

    layout = JointChunkLayout(latent_frames, patches, 1)
    gen = layout.num_tokens
    runtime = sequence_pack_from_packed_sequence(
        torch.zeros(text + gen, 1),
        ["causal", "full"],
        [text, gen],
        [text + gen],
        torch.arange(text),
        torch.arange(text, text + gen),
        cp_world_size=1,
        pad_for_cuda_graphs=False,
    )
    assert runtime["causal_seq"].shape[0] == text + 1
    assert runtime["full_only_seq"].shape[0] == gen + 1
    audit = joint_pack_padding_audit(text, gen)
    assert audit["runtime_forward_capacity"] == 2 * (runtime["causal_seq"].shape[0] + runtime["full_only_seq"].shape[0])
    observed = []

    def record_shapes(query, key, value, **kwargs):
        observed.append((query.shape[-2], key.shape[-2]))
        return torch.zeros_like(query)

    # Exercise production padding/cat/reshape, replacing only the GPU kernel/mask.
    monkeypatch.setattr(flex_attention, "_COMPILED_FLEX_ATTENTION", record_shapes)
    attention = JointTeacherForcingAttention(layout, "cpu", text_lengths=[text])
    monkeypatch.setattr(attention, "block_mask", lambda **kwargs: None)
    q = torch.zeros(1, 1, gen, 1, 1)
    tk = runtime["causal_seq"].reshape(1, -1, 1, 1)
    for memory_class in (TFReplayCleanMemoryValue, TFNoisyMemoryValue):
        memory = object.__new__(memory_class)
        memory.has_caption = True
        memory.und_kv_offsets = torch.tensor([0, text])
        memory.cached_clean_gen_k = torch.zeros(1, gen, 1, 1)
        memory.cached_clean_gen_v = torch.zeros(1, gen, 1, 1)
        result = attention(q, q, q, tk, tk, memory)
        assert result.shape == q.shape
    assert observed == [
        (audit["flex_gen_query_capacity_per_pass"], audit["flex_clean_kv_capacity"]),
        (audit["flex_gen_query_capacity_per_pass"], audit["flex_noisy_kv_capacity"]),
    ]
