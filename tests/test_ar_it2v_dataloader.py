"""Native stateful loader + packing buffer recovery, on tiny CPU videos."""
import random

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from cosmos3_ar_it2v.dataset import IT2VIterableDataset
from cosmos3_ar_it2v.dataloader import RecoverablePackingDataLoader, IT2VDataLoaderStateCallback
from cosmos_framework.data.generator.joint_dataloader import RankPartitionedDataLoader
from cosmos_framework.data.generator.sequence_packing import SequencePlan


class TinyVideos(Dataset):
    def __len__(self):
        return 30

    def get_item_at_window(self, index, *, epoch=0, window_start=None, source_frame_indices=None):
        start = epoch + index if window_start is None else int(window_start)
        frames = 17 if index % 2 else 33
        source = torch.arange(start, start + frames)
        if source_frame_indices is not None:
            assert torch.equal(source, torch.as_tensor(source_frame_indices).reshape(-1))
        return {
            'video': torch.full((3,frames,32,32), (start+index)%255, dtype=torch.uint8),
            'dataset_index': index, 'sample_id': str(index), 'window_start': start,
            'source_frame_indices': source, 'text_token_ids': torch.tensor([index+1,2,3]),
            'ai_caption': f'clip {index}', 'conditioning_fps': 30.,
            'image_size': torch.tensor([32,32,32,32]),
            'sequence_plan': SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0]),
        }


def test_native_pending_buffer_restores_media_window_and_rng(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

    def make():
        inner = RankPartitionedDataLoader(
            datasets={'video': {'dataset': IT2VIterableDataset(TinyVideos()), 'ratio': 1}},
            stateful=True, num_workers=0, batch_size=1,
        )
        return RecoverablePackingDataLoader(
            dataloader=inner, tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4, patch_spatial=2,
            max_sequence_length=32, max_samples_per_batch=None,
            prewarm=False, lazy_initialize_child_iterators=True,
        )

    original = make()
    iterator = iter(original)
    next(iterator)
    callback = IT2VDataLoaderStateCallback()
    callback.bind_dataloader(original)
    state = callback.state_dict()
    assert state['buffer']
    assert all('video' not in sample for sample in state['buffer'])
    expected = next(iterator)

    restored = make()
    callback2 = IT2VDataLoaderStateCallback()
    callback2.bind_dataloader(restored)
    callback2.load_state_dict(state)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    actual = next(iter(restored))
    assert random.getstate() == rng[0]
    np.testing.assert_equal(np.random.get_state(), rng[1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    assert actual['sample_id'] == expected['sample_id']
    for key in ('video','source_frame_indices','text_token_ids','conditioning_fps','sequence_plan'):
        RecoverablePackingDataLoader._assert_rebuilt_metadata(actual[key], expected[key], key)
    assert actual['_num_tokens'] == expected['_num_tokens']
    assert original.global_id == restored.global_id


class FullSegments(TinyVideos):
    """All real frames survive; only 0..3 repeated tail frames align the VAE."""
    lengths = (6, 17, 38, 71, 10, 29, 54, 9, 18)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return self.get_item_at_window(index)

    def get_item_at_window(self, index, *, epoch=0, window_start=None, source_frame_indices=None):
        start = index * 100
        assert window_start is None or int(window_start) == start
        frames = self.lengths[index]
        source = torch.arange(start, start + frames)
        if source_frame_indices is not None:
            assert torch.equal(source, torch.as_tensor(source_frame_indices).reshape(-1))
        pad = (-(frames - 1)) % 4
        raw = super().get_item_at_window(index, window_start=start)
        video = torch.arange(frames, dtype=torch.uint8).view(1, frames, 1, 1).expand(3, frames, 32, 32)
        raw.update(
            video=torch.cat([video, video[:, -1:].expand(3, pad, 32, 32)], dim=1),
            source_frame_indices=source, sample_mode='full_segment',
            video_true_num_frames=frames, video_temporal_padding=pad,
            frame_end=start+frames-1, num_frames=frames+pad,
        )
        # Simulate the epoch-keyed CFG tokenizer: buffer rebuild at epoch 0
        # must restore the saved text, not silently change its token count.
        ntext = 1 + ((index + epoch) % 4)
        raw['text_token_ids'] = torch.arange(ntext) + 10 * epoch
        raw['ai_caption'] = f'epoch {epoch} segment {index}'
        return raw


def _full_loader(monkeypatch, *, budget=40, prewarm=False, workers=0):
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    inner = RankPartitionedDataLoader(
        datasets={'video': {'dataset': IT2VIterableDataset(FullSegments()), 'ratio': 1}},
        stateful=True, num_workers=workers, batch_size=1,
    )
    return RecoverablePackingDataLoader(
        dataloader=inner, tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4, patch_spatial=2,
        max_sequence_length=budget, max_samples_per_batch=None,
        lookahead_limit=3, prewarm=prewarm, lazy_initialize_child_iterators=True,
    )


def test_full_segments_native_packing_keeps_every_frame_and_sample_boundary():
    from torch.utils.data import DataLoader
    from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn, PackingDataLoader
    dataset = FullSegments()
    loader = RecoverablePackingDataLoader(
        dataloader=DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn),
        tokenizer_spatial_compression_factor=16, tokenizer_temporal_compression_factor=4,
        patch_spatial=2, max_sequence_length=40, max_samples_per_batch=None,
        lookahead_limit=3, prewarm=False, lazy_initialize_child_iterators=True,
    )
    assert RecoverablePackingDataLoader.__iter__ is PackingDataLoader.__iter__
    seen, batch_sizes = [], []
    for batch in loader:
        batch_sizes.append(len(batch['video']))
        assert batch['_num_tokens'] < 40
        assert batch['_dropped_count'] == 0
        expected_tokens = 0
        for i, sid in enumerate(batch['sample_id']):
            idx = int(sid)
            seen.append(idx)
            raw = dataset[idx]
            assert len(batch['video'][i]) == 1
            assert torch.equal(batch['video'][i][0], raw['video'])
            assert torch.equal(batch['source_frame_indices'][i].flatten(), raw['source_frame_indices'])
            assert batch['sequence_plan'][i] == raw['sequence_plan']
            expected_tokens += (raw['video'].shape[1]-1)//4+1 + len(raw['text_token_ids'])+3
        assert batch['_num_tokens'] == expected_tokens
    assert sorted(seen) == list(range(len(dataset)))
    assert len(set(batch_sizes)) > 1
    assert max(batch_sizes) > 1


def test_official_token_accounting_includes_spatial_ceil_and_markers(monkeypatch):
    loader = _full_loader(monkeypatch, budget=10000)
    sample = loader._split_single_sample(FullSegments()[2])
    # A meta tensor supplies real geometry without allocating a decoded video.
    sample['video'] = [torch.empty(3, 41, 368, 640, device='meta')]
    assert loader._compute_sample_cost(sample)[0] == 11 * 12 * 20 + 3 + 3


def test_oversize_and_exact_ceiling_raise_instead_of_native_discard(monkeypatch):
    import pytest
    loader = _full_loader(monkeypatch, budget=40)
    sample = loader._split_single_sample(FullSegments()[3])
    count = loader._compute_num_tokens_per_sample(sample)
    for ceiling in (count-1, count):
        loader.max_sequence_length = ceiling
        with pytest.raises(ValueError, match='sample_id=.*tokens=.*no sample was truncated or discarded'):
            loader._compute_sample_cost(sample)
    loader.max_sequence_length = count+1
    assert loader._compute_sample_cost(sample)[0] == count
    # Exercise the inherited iterator too: oversize must not disappear after a log.
    loader.max_sequence_length = 2
    with pytest.raises(ValueError, match='exclusive packing token budget'):
        next(iter(loader))


@pytest.mark.parametrize("workers", [0, 2])
def test_full_segment_variable_batch_restore_across_epoch_and_prewarm(monkeypatch, workers):
    original = _full_loader(monkeypatch, prewarm=True, workers=workers)
    iterator = iter(original)
    for _ in range(5):
        next(iterator)
    state = original.state_dict()
    assert state['buffer']
    assert all('video' not in item for item in state['buffer'])
    expected = [next(iterator) for _ in range(8)]
    restored = _full_loader(monkeypatch, prewarm=True, workers=workers)
    restored.load_state_dict(state)
    # Trainer calls this on resume; it must not overwrite the packing cursor.
    restored.set_start_iteration(99999)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    iterator2 = iter(restored)
    actual = [next(iterator2) for _ in range(8)]
    assert random.getstate() == rng[0]
    np.testing.assert_equal(np.random.get_state(), rng[1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    for left, right in zip(actual, expected, strict=True):
        # Worker timers describe elapsed wall time, not training data state.
        for key in ('video', 'source_frame_indices', 'sample_id', 'text_token_ids',
                    'sequence_plan', 'video_true_num_frames', 'video_temporal_padding',
                    'num_frames', 'frame_end', '_num_tokens', '_dropped_count'):
            RecoverablePackingDataLoader._assert_rebuilt_metadata(left[key], right[key], key)
    assert restored.global_id == original.global_id


def test_resume_rejects_changed_packing_budget(monkeypatch):
    import pytest
    original = _full_loader(monkeypatch)
    next(iter(original))
    state = original.state_dict()
    restored = _full_loader(monkeypatch, budget=41)
    with pytest.raises(ValueError, match='different packing token budget'):
        restored.load_state_dict(state)
