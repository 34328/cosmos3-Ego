"""Native stateful loader + packing buffer recovery, on tiny CPU videos."""
import random

import numpy as np
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
