"""CPU regressions for native finite and unsized iterable loader lengths."""

import itertools

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from cosmos_framework.data.generator.joint_dataloader import RankPartitionedDataLoader


class FiniteStream(IterableDataset):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        return iter(range(self.size))

    def __len__(self):
        return self.size


class InfiniteStream(IterableDataset):
    def __iter__(self):
        return itertools.count()


def loader(dataset, monkeypatch, **kwargs):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    return RankPartitionedDataLoader(
        {"test": {"dataset": dataset, "ratio": 1}},
        collate_fn=(torch.utils.data.default_convert if kwargs.get("batch_size") is None else torch.utils.data.default_collate),
        num_workers=0,
        **kwargs,
    )


@pytest.mark.parametrize("size", [0, 1, 5])
@pytest.mark.parametrize("batch_size", [None, 1, 2])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("stateful", [False, True])
def test_finite_iterable_length_matches_native_batches(size, batch_size, drop_last, stateful, monkeypatch):
    if batch_size is None and drop_last:
        pytest.skip("native unbatched mode does not support drop_last")
    dataset = FiniteStream(size)
    wrapped = loader(dataset, monkeypatch, batch_size=batch_size, drop_last=drop_last, stateful=stateful)
    native = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
    assert len(wrapped) == len(native)
    assert len(list(iter(wrapped))) == len(native)


@pytest.mark.parametrize("stateful", [False, True])
def test_unsized_infinite_stream_retains_zero_length(stateful, monkeypatch):
    wrapped = loader(InfiniteStream(), monkeypatch, batch_size=2, stateful=stateful)
    assert len(wrapped) == 0
    batches = list(itertools.islice(iter(wrapped), 3))
    assert [batch.tolist() for batch in batches] == [[0, 1], [2, 3], [4, 5]]


def test_map_dataset_keeps_native_batch_length(monkeypatch):
    wrapped = loader(torch.utils.data.TensorDataset(torch.arange(5)), monkeypatch, batch_size=2, drop_last=True)
    assert len(wrapped) == 2
    assert len(list(iter(wrapped))) == 2


def test_sized_iterable_length_errors_are_not_hidden(monkeypatch):
    class BrokenStream(FiniteStream):
        def __len__(self):
            raise TypeError("broken finite length")

    wrapped = loader(BrokenStream(5), monkeypatch, batch_size=2)
    with pytest.raises(TypeError, match="broken finite length"):
        len(wrapped)
