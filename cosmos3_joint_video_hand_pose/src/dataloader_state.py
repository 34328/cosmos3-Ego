from __future__ import annotations

import copy
import dataclasses
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from cosmos_framework.data.generator.joint_dataloader import (
    _BATCH_TIMING_KEYS,
    PackingDataLoader,
    custom_collate_fn,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


_LARGE_SAMPLE_KEYS = {"video", "images", "action", "action_raw", "sound"}


class RecoverablePackingDataLoader(PackingDataLoader):
    """PackingDataLoader with exact inner-stream and lightweight buffer state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._restored_buffer_metadata: list[dict[str, Any]] | None = None
        self._state_was_restored = False

    @staticmethod
    def _dataset_index(sample: dict[str, Any]) -> int:
        value = sample["dataset_index"]
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            return RecoverablePackingDataLoader._dataset_index({"dataset_index": value[0]})
        return int(value)

    @staticmethod
    def _checkpoint_metadata(sample: dict[str, Any]) -> dict[str, Any]:
        # AR media are deterministic only from dataset_index AND the saved
        # source window. Preserve window/state hashes and CFG text/plan exactly.
        if "ar_layout_version" in sample and "source_frame_indices" not in sample:
            raise ValueError("AR packing checkpoint requires exact source_frame_indices")
        return copy.deepcopy({key: value for key, value in sample.items() if key not in _LARGE_SAMPLE_KEYS})

    def state_dict(self) -> dict[str, Any]:
        buffer = list(self.buffers[0]) if self._child_iterators_initialized else []
        return {
            "version": 2,
            "global_id": self.global_id,
            "inner": self.dataloader_list[0].state_dict(),
            "buffer": (
                [self._checkpoint_metadata(sample) for sample in buffer]
                if self._child_iterators_initialized
                else copy.deepcopy(self._restored_buffer_metadata or [])
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._child_iterators_initialized:
            raise RuntimeError("Dataloader state must be restored before worker iterators are initialized.")
        if int(state_dict.get("version", 0)) not in (1, 2):
            raise ValueError(f"Unsupported dataloader checkpoint version: {state_dict.get('version')!r}")
        self.dataloader_list[0].load_state_dict(state_dict["inner"])
        self.global_id = int(state_dict["global_id"])
        self._restored_buffer_metadata = copy.deepcopy(list(state_dict.get("buffer", [])))
        self._state_was_restored = True
        log.info(
            f"Restored packed dataloader at global_id={self.global_id} with "
            f"{len(self._restored_buffer_metadata)} pending sample(s).",
            rank0_only=False,
        )

    def set_start_iteration(self, iteration: int) -> None:
        # The checkpoint owns the true packed-batch cursor. Trainer iteration is
        # different when context parallelism reuses one fetched batch.
        if not self._state_was_restored:
            super().set_start_iteration(iteration)

    def _map_dataset(self) -> Any:
        dataset = self.dataloader_list[0].dataset
        while isinstance(dataset, IterableDataset):
            dataset = getattr(dataset, "_dataset", getattr(dataset, "dataset", None))
            if dataset is None:
                raise TypeError("Cannot locate the map-style dataset used to rebuild the packing buffer.")
        return dataset

    def _split_single_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        batch = custom_collate_fn([raw_sample])
        sample: dict[str, Any] = {}
        for key, value in batch.items():
            if key in _BATCH_TIMING_KEYS:
                sample[key] = value
            elif isinstance(value, list) and key in self._MULTI_ITEM_KEYS:
                elem = value[0]
                sample[key] = elem if isinstance(elem, list) else value[0:1]
            elif isinstance(value, list):
                sample[key] = value[0]
            else:
                sample[key] = value[0:1]
        return sample

    @staticmethod
    def _assert_rebuilt_metadata(actual, saved, key):
        """Never hide a changed physical/state/source payload behind old metadata."""
        if isinstance(saved, torch.Tensor):
            if (
                not isinstance(actual, torch.Tensor)
                or actual.shape != saved.shape
                or not torch.equal(actual.cpu(), saved.cpu())
            ):
                raise ValueError(f"restored dataset differs from checkpoint: {key}")
        elif isinstance(saved, np.ndarray):
            if not isinstance(actual, np.ndarray) or not np.array_equal(actual, saved):
                raise ValueError(f"restored dataset differs from checkpoint: {key}")
        elif isinstance(saved, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(saved):
                raise ValueError(f"restored dataset differs from checkpoint: {key}")
            for left, right in zip(actual, saved, strict=True):
                RecoverablePackingDataLoader._assert_rebuilt_metadata(left, right, key)
        elif isinstance(saved, dict):
            if not isinstance(actual, dict) or actual.keys() != saved.keys():
                raise ValueError(f"restored dataset differs from checkpoint: {key}")
            for name, value in saved.items():
                RecoverablePackingDataLoader._assert_rebuilt_metadata(actual[name], value, f"{key}.{name}")
        elif dataclasses.is_dataclass(saved):
            if type(actual) is not type(saved):
                raise ValueError(f"restored dataset differs from checkpoint: {key}")
            for field in dataclasses.fields(saved):
                RecoverablePackingDataLoader._assert_rebuilt_metadata(
                    getattr(actual, field.name), getattr(saved, field.name), f"{key}.{field.name}"
                )
        elif actual != saved:
            raise ValueError(f"restored dataset differs from checkpoint: {key}")

    def _rebuild_buffer(self, metadata_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Re-decoding must not perturb trainer-side noise/timestep RNG state.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            dataset = self._map_dataset()
            rebuilt = []
            for metadata in metadata_items:
                index = self._dataset_index(metadata)
                exact_window = getattr(dataset, "get_item_at_window", None)
                is_windowed = "window_start" in metadata or "ar_layout_version" in metadata
                if is_windowed:
                    if exact_window is None or "source_frame_indices" not in metadata:
                        raise ValueError("windowed checkpoint needs get_item_at_window and saved source indexes")
                    first = metadata.get("window_start")
                    while isinstance(first, (list, tuple)) and len(first) == 1:
                        first = first[0]
                    source = metadata["source_frame_indices"]
                    while isinstance(source, (list, tuple)) and len(source) == 1:
                        source = source[0]
                    raw = exact_window(index, window_start=first, source_frame_indices=source)
                else:
                    # Compatibility for legacy datasets with deterministic index lookup.
                    raw = dataset[index]
                sample = self._split_single_sample(raw)
                # These fields can change when re-running the tokenizer's CFG
                # dropout. Restore them, but never overwrite reconstructed media,
                # source indexes, visibility, boundary states or artifact hashes.
                stochastic = {
                    "ai_caption",
                    "text_token_ids",
                    "text_token_lengths",
                    "sequence_plan",
                    "ar_num_tokens",
                    "num_tokens",
                    "text",
                    "caption",
                }
                for key, value in metadata.items():
                    if key in stochastic or key in _BATCH_TIMING_KEYS:
                        continue
                    if key not in sample:
                        raise ValueError(f"restored dataset missing checkpoint field: {key}")
                    self._assert_rebuilt_metadata(sample[key], value, key)
                for key in stochastic:
                    if key in metadata:
                        sample[key] = copy.deepcopy(metadata[key])
                rebuilt.append(sample)
            return rebuilt
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)

    def _initialize_child_iterators_once(self) -> None:
        # Recreating a StatefulDataLoader iterator draws a temporary CPU worker
        # base seed before restoring its saved seed. Do not consume the trainer's
        # already-restored RNG: Cosmos samples video/action sigmas on the CPU.
        preserve_rng = self._state_was_restored and not self._child_iterators_initialized
        if preserve_rng:
            python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            super()._initialize_child_iterators_once()
            if self._restored_buffer_metadata is None:
                return
            pending = self._rebuild_buffer(self._restored_buffer_metadata)
            # Keep saved pending samples ahead of the newly prewarmed sample.
            self.buffers[0] = deque(pending + list(self.buffers[0]))
            log.info(f"Rebuilt {len(pending)} pending packed sample(s).", rank0_only=False)
            self._restored_buffer_metadata = None
        finally:
            if preserve_rng:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(torch_state)


class EgoVerseDataLoaderStateCallback(Callback):
    """DCP adapter plus a rank-local JSONL sample trace for spike replay."""

    checkpoint_component = "dataloader"

    def __init__(self) -> None:
        super().__init__()
        self._loader: RecoverablePackingDataLoader | None = None
        self._trace_handle = None

    def bind_dataloader(self, dataloader: Any) -> None:
        if not isinstance(dataloader, RecoverablePackingDataLoader):
            raise TypeError(f"Expected RecoverablePackingDataLoader, got {type(dataloader).__name__}.")
        self._loader = dataloader

    def has_checkpoint_state(self) -> bool:
        return True

    def state_dict(self) -> dict[str, Any]:
        if self._loader is None:
            raise RuntimeError("Train dataloader has not been bound to the checkpoint callback.")
        return self._loader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._loader is None:
            raise RuntimeError("Train dataloader has not been bound to the checkpoint callback.")
        self._loader.load_state_dict(state_dict)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                result.extend(EgoVerseDataLoaderStateCallback._as_list(item))
            return result
        return [value]

    def on_training_step_batch_start(
        self,
        model: Any,
        data_batch: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        del model
        if self._trace_handle is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            trace_dir = Path(self.config.job.path_local) / "dataloader_trace"
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_handle = (trace_dir / f"rank_{rank:05d}.jsonl").open("a", encoding="utf-8")
        record = {
            "iteration": int(iteration),
            "packed_global_id": None if self._loader is None else int(self._loader.global_id),
            "dataset_indices": [int(item) for item in self._as_list(data_batch.get("dataset_index", []))],
            "sample_ids": [str(item) for item in self._as_list(data_batch.get("sample_id", []))],
        }
        self._trace_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._trace_handle.flush()

    def on_app_end(self) -> None:
        if self._trace_handle is not None:
            self._trace_handle.close()
            self._trace_handle = None
