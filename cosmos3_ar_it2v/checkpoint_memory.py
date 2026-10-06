"""Release unused allocator blocks before native DCP/NCCL checkpoint work."""
import gc
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


class CheckpointMemory(Callback):
    """Keep live tensors on GPU; return only unused cache to the CUDA driver."""

    @staticmethod
    def _snapshot():
        free, total = torch.cuda.mem_get_info()
        return dict(allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    driver_free_bytes=free, driver_total_bytes=total)

    def _record(self, phase, iteration, **values):
        rank = dist.get_rank() if dist.is_initialized() else 0
        root = Path(self.config.job.path_local) / 'checkpoint_memory'
        root.mkdir(parents=True, exist_ok=True)
        row = dict(phase=phase, step=int(iteration), rank=rank, **values)
        with (root / f'rank_{rank:05d}.jsonl').open('a') as file:
            file.write(json.dumps(row) + '\n')

    def on_save_checkpoint_start(self, model, iteration=0):
        if not torch.cuda.is_available():
            return
        start = time.monotonic()
        # No new collective here: NCCL may need driver memory to initialize it.
        torch.cuda.synchronize()
        before = self._snapshot()
        gc.collect()
        torch.cuda.empty_cache()
        after = self._snapshot()
        self._record('before_save', iteration, before=before, after=after,
                     cleanup_seconds=time.monotonic() - start)

    def on_save_checkpoint_end(self, model, iteration=0):
        if torch.cuda.is_available():
            self._record('after_save', iteration, memory=self._snapshot())
