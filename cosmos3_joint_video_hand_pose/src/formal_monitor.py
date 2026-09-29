"""Read-only measurements and fail-fast guard via the native Trainer callbacks."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
import wandb

from cosmos_framework.callbacks.grad_clip import GradClip
from cosmos_framework.utils.callback import Callback
from .wandb_metrics import LOSS_METRIC_SOURCES, extract_loss_metrics


def optimizer_lr_receipt(net, optimizer, config):
    """Inspect native optimizer groups, including their scheduled starting LR."""
    names = {id(p): name for name, p in net.named_parameters()}
    multipliers = dict(config.get("lr_multipliers", {}))
    children = getattr(optimizer, "optimizers", [optimizer])
    children = children.values() if isinstance(children, Mapping) else children
    rows, seen = [], set()
    for child in children:
        for group in child.param_groups:
            tags = set()
            for p in group["params"]:
                name = names[id(p)]
                tag = next((key for key in multipliers if key in name), "default")
                expected = float(config["lr"]) * multipliers.get(tag, 1)
                if not math.isclose(float(group.get("initial_lr", group["lr"])), expected, rel_tol=1e-8):
                    raise ValueError(f"native optimizer LR mismatch for {name}")
                tags.add(tag)
                seen.add(tag)
            rows.append(dict(tags=sorted(tags), parameters=len(group["params"]),
                             initial_lr=float(group.get("initial_lr", group["lr"])),
                             actual_lr=float(group["lr"])))
    if set(multipliers) - seen:
        raise ValueError(f"LR multiplier matched no optimizer parameters: {set(multipliers)-seen}")
    return rows


class StopPolicy:
    """No tuning: 9/10 clips; 3x initial loss mean; sustained resident growth."""
    def __init__(self):
        self.clips = deque(maxlen=10)
        self.losses = deque(maxlen=10)
        self.memory = deque(maxlen=10)
        self.baseline = None

    def update(self, clipped, losses, resident_gib):
        if not all(math.isfinite(x) for x in (*losses, resident_gib)):
            return "nonfinite loss or memory metric"
        self.clips.append(bool(clipped))
        self.losses.append(tuple(losses))
        self.memory.append(resident_gib)
        if len(self.clips) == 10 and sum(self.clips) >= 9:
            return "grad_clip triggered in at least 9 of the last 10 steps"
        if len(self.losses) == 10:
            means = [sum(x[i] for x in self.losses)/10 for i in range(len(losses))]
            if self.baseline is None:
                self.baseline = means
            elif any(x > 3 * max(b, 1e-12) for x, b in zip(means, self.baseline)):
                return "10-step mean loss exceeds 3x first-10-step mean"
        if len(self.memory) == 10 and self.memory[-1] - self.memory[0] > 2:
            if all(b > a for a, b in zip(self.memory, list(self.memory)[1:])):
                return "resident allocated memory increased every step for 10 steps by >2 GiB"
        return None


class FormalTrainingMonitor(Callback):
    """Use native GradClip measurements; never modify the optimizer or loss."""
    def on_train_start(self, model, iteration=0):
        self.policy = StopPolicy()
        self.clip = next(c for c in self.trainer.callbacks._callbacks if isinstance(c, GradClip))
        self.root = Path(self.config.job.path_local)
        if iteration:
            for line in (self.root / "formal_monitor.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row["step"] <= iteration:
                    self.policy.update(row["grad_clip_triggered"],
                                       [row[k] for k in ("loss/video_raw", "loss/action_raw", "loss/total")],
                                       row["resident_allocated_gib"])
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.root.mkdir(parents=True, exist_ok=True)
            receipt = model._optimizer_lr_receipt
            (self.root / "optimizer_lr_groups.json").write_text(json.dumps(receipt, indent=2))
            print("ACTUAL_OPTIMIZER_LR_GROUPS " + json.dumps(receipt), flush=True)

    def on_training_step_start(self, model, data, iteration=0):
        torch.cuda.synchronize()
        self.resident = torch.cuda.memory_allocated()/2**30
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        self.clips = len(data["video"])

    def on_before_backward(self, model, loss, iteration=0):
        bad = (~torch.isfinite(loss.detach())).any().to(dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item():
            raise FloatingPointError("formal training stopped: nonfinite loss before backward")

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        norm = float(self.clip._last_global_norm[self.clip._state_key])
        # Match the official GradClip incidence metric exactly.
        clipped = float(norm > self.clip.clip_norm)
        metrics = extract_loss_metrics(output_batch)
        self.names = tuple(LOSS_METRIC_SOURCES)
        values = [self.clips, torch.cuda.max_memory_allocated()/2**30,
                  torch.cuda.max_memory_reserved()/2**30, time.perf_counter()-self.start,
                  self.resident, norm, clipped]
        values += [float(metrics[k].detach().float().mean()) for k in self.names]
        local = torch.tensor(values, dtype=torch.float64, device=loss.device)
        rows = [torch.empty_like(local) for _ in range(dist.get_world_size())] if dist.is_initialized() else [local]
        if dist.is_initialized():
            dist.all_gather(rows, local)
        self.rows = torch.stack(rows).cpu()

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        rows = self.rows
        metrics = dict(zip(self.names, rows[:, 7:].mean(0).tolist()))
        reason = self.policy.update(bool(rows[:, 6].max()),
                                    [metrics[k] for k in ("loss/video_raw", "loss/action_raw", "loss/total")],
                                    float(rows[:, 4].max()))
        row = dict(step=iteration, global_batch=int(rows[:, 0].sum()),
                   clips_per_rank=rows[:, 0].int().tolist(),
                   peak_allocated_gib=float(rows[:, 1].max()), peak_reserved_gib=float(rows[:, 2].max()),
                   train_step_seconds=float(rows[:, 3].max()), preclip_norm=float(rows[:, 5].max()),
                   resident_allocated_gib=float(rows[:, 4].max()), grad_clip_triggered=bool(rows[:, 6].max()),
                   grad_clip_trigger_rate=sum(self.policy.clips)/len(self.policy.clips),
                   report_due=(iteration <= 100 and iteration % 10 == 0) or iteration % 100 == 0,
                   stop_reason=reason, **metrics)
        if not dist.is_initialized() or dist.get_rank() == 0:
            with (self.root / "formal_monitor.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False)+"\n")
            if wandb.run is not None:
                wandb.log({"formal/"+k: v for k, v in row.items() if isinstance(v, (int, float))},
                          step=iteration, commit=False)
            if row["report_due"] or reason:
                print("FORMAL_PROGRESS " + json.dumps(row), flush=True)
            if reason:
                (self.root / "STOPPED.json").write_text(json.dumps(row, indent=2))
        if reason:
            raise RuntimeError("formal training stopped: " + reason)
