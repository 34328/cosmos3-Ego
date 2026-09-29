"""CPU regression: real worker replay must not advance the trainer RNG."""

import copy
import random

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader
from cosmos3_joint_video_hand_pose.src.dataloader_state import RecoverablePackingDataLoader


class RandomSamples(Dataset):
    def __len__(self):
        return 128

    def get_shuffle_blocks(self):
        return [(index * 16, 16) for index in range(8)]

    def __getitem__(self, index):
        return [index, random.random(), float(np.random.rand()), float(torch.rand(()))]


def make_loader(snapshot_steps):
    inner = StatefulDataLoader(
        ActionIterableShuffleDataset(RandomSamples()),
        batch_size=None,
        num_workers=2,
        multiprocessing_context="fork",
        prefetch_factor=2,
        snapshot_every_n_steps=snapshot_steps,
    )
    return RecoverablePackingDataLoader(
        dataloader=inner,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=None,
        max_samples_per_batch=1,
        prewarm=False,
        lazy_initialize_child_iterators=True,
    )


@pytest.mark.parametrize("snapshot_steps", [1, 3])
def test_real_stateful_resume_preserves_parent_rng_and_worker_replay(snapshot_steps):
    # A non-aligned snapshot also exercises torchdata's prefetched replay path.
    original = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    loaders = []
    try:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        continuous = make_loader(snapshot_steps)
        loaders.append(continuous)
        fresh_rng = torch.get_rng_state().clone()
        continuous._initialize_child_iterators_once()
        assert not torch.equal(fresh_rng, torch.get_rng_state())
        for _ in range(4):
            next(continuous.dataloaders[0])
        saved = copy.deepcopy(continuous.state_dict())
        parent = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        expected = [next(continuous.dataloaders[0]) for _ in range(8)]
        assert torch.equal(parent[2], torch.get_rng_state())

        for protected in (False, True):
            random.setstate(parent[0])
            np.random.set_state(parent[1])
            torch.set_rng_state(parent[2])
            resumed = make_loader(snapshot_steps)
            loaders.append(resumed)
            resumed.load_state_dict(copy.deepcopy(saved))
            assert torch.equal(parent[2], torch.get_rng_state())
            if protected:
                resumed._initialize_child_iterators_once()
            else:
                # Exact old super() path; an empty rebuild is RNG-neutral.
                PackingDataLoader._initialize_child_iterators_once(resumed)
            after = torch.get_rng_state().clone()
            assert torch.equal(parent[2], after) == protected
            assert random.getstate() == parent[0]
            now_numpy = np.random.get_state()
            assert now_numpy[0] == parent[1][0]
            np.testing.assert_array_equal(now_numpy[1], parent[1][1])
            assert now_numpy[2:] == parent[1][2:]
            assert [next(resumed.dataloaders[0]) for _ in range(8)] == expected
            assert torch.equal(after, torch.get_rng_state())
            if not protected:
                torch.set_rng_state(parent[2])
                torch.empty((), dtype=torch.int64).random_()
                assert torch.equal(after, torch.get_rng_state())
            else:
                resumed._initialize_child_iterators_once()
                assert torch.equal(after, torch.get_rng_state())
    finally:
        for loader in loaders:
            for iterator in getattr(loader, "dataloaders", []):
                iterator._shutdown_workers()
        random.setstate(original[0])
        np.random.set_state(original[1])
        torch.set_rng_state(original[2])
