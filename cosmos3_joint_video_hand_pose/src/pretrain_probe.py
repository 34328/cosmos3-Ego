"""Opt-in measurements around the official Trainer; never a training loop."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import torch
import torch.distributed as dist
from cosmos_framework.utils.callback import Callback
from .wandb_metrics import extract_loss_metrics


def payload_digest(value):
    """Hash exact tensor values and nested metadata without consuming RNG."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu().contiguous()
            digest.update(str((str(x.dtype), list(x.shape))).encode())
            digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode())
        digest.update(b"\0")
    visit(value)
    return digest.hexdigest()


class PretrainProbeCallback(Callback):
    """Disabled by default. CPU traces remain local; official W&B logs losses."""
    def __init__(self, enabled=False):
        super().__init__()
        self.enabled = enabled

    def on_train_start(self, model, iteration=0):
        model._record_pretrain_probe = self.enabled

    def on_training_step_start(self, model, data, iteration=0):
        if not self.enabled:
            return
        if self.config.trainer.grad_accum_iter != 1:
            raise ValueError("pretrain probe currently requires grad_accum_iter=1")
        keys = ("sample_id", "dataset_index", "source_frame_indices", "window_start",
                "text_token_ids", "action", "ar_boundary_states", "hand_visibility")
        self._data_hash = payload_digest({k: data[k] for k in keys if k in data})
        self._frames = []
        for video in data["video"]:
            while isinstance(video, (list, tuple)) and len(video) == 1:
                video = video[0]
            self._frames.append(int(video.shape[-3]))
        self._tokens = {k: int(v) for k, v in data.items()
                        if k.startswith("_ar_c4_") and isinstance(v, (int, float))}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._start = time.perf_counter()

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        rank = dist.get_rank() if dist.is_initialized() else 0
        row = dict(step=int(iteration)+1, rank=rank,
                   data_sha256=self._data_hash, frame_counts=self._frames,
                   clips=len(self._frames), token_budget=self._tokens,
                   train_step_seconds=time.perf_counter()-self._start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   noise=model._pretrain_noise_trace,
                   losses={k: float(v.detach().float().mean()) for k,v in extract_loss_metrics(output_batch).items()})
        path = Path(self.config.job.path_local)/"pretrain_probe"/f"rank_{rank:05d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False)+"\n")
            handle.flush()
