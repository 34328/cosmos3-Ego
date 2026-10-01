"""Read-only native Trainer diagnostics for the V0.2/V0.3 short comparison."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import weakref

import torch
import torch.distributed as dist

from cosmos_framework.callbacks.grad_clip import GradClip
from cosmos_framework.utils.callback import Callback

from .ar_v02_layout import ACTION, CONDITION_VIDEO, STATE, VIDEO


def _json_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(x) for x in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def batch_identity(data):
    """Hash metadata already in the batch; never load or change a window."""
    keys = ("sample_id", "dataset_index", "window_start", "clip_frames",
            "ar_num_tokens", "ar_token_budget_version", "ar_layout_version",
            "ar_valid_windows_sha256", "ar_state_normalizer_sha256",
            "ar_future_normalizer_sha256", "ar_right_hand_codec_sha256",
            "ar_left_hand_codec_sha256")
    result = {key: _json_value(data[key]) for key in keys if key in data}
    for key in ("source_frame_indices", "action_source_frame_indices",
                "future_action_source_frame_indices", "ar_boundary_source_indices"):
        if key in data:
            payload = _json_value(data[key])
            result[key + "_sha256"] = hashlib.sha256(
                json.dumps(payload, separators=(",", ":")).encode()
            ).hexdigest()
    if "window_start" not in data and "source_frame_indices" not in data:
        # Legacy transforms may omit the exact-window metadata. Hash the
        # already-loaded tensors as a costly but honest fallback, never fetch
        # another clip or claim a segment ID alone identifies its window.
        digest = hashlib.sha256()

        def hash_input(value):
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(value, (list, tuple)):
                digest.update(str(len(value)).encode())
                for item in value:
                    hash_input(item)
            else:
                raise ValueError("exact-window fallback requires loaded video/action tensors")

        for key in ("video", "action"):
            if key not in data:
                raise ValueError("batch lacks exact-window metadata and loaded input tensors")
            digest.update(key.encode())
            hash_input(data[key])
        result["loaded_video_action_sha256"] = digest.hexdigest()
    result["window_identity_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result["packing_audit"] = {key: _json_value(value) for key, value in data.items()
                               if key == "_num_tokens" or key.startswith("_ar_c4_")}
    return result


def actual_sigma_snapshot(step, layouts):
    """Read the prepared RF noising tensors while ARStepContext is alive.

    Chunk slot zero and padded blocks are unused draws, not conditioning rows.
    Only real layout blocks enter the snapshot; U/S values come from the RF
    noised-data result rather than substituting theoretical clean values.
    """
    if step is None or not hasattr(step, "video_chunk_sigmas"):
        return None  # The preserved V0.2 baseline has no V0.3 sigma receipt.
    required = ("action_sigmas", "noised_video_sigmas", "noised_action_sigmas")
    if any(getattr(step, key, None) is None for key in required):
        raise RuntimeError("actual sigma diagnostic requires the prepared RF noised tensors")
    video = step.video_chunk_sigmas.detach().cpu()
    action = step.action_sigmas.detach().cpu()
    if video.ndim != 2 or action.shape != video.shape or len(layouts) != video.shape[0]:
        raise RuntimeError("actual sigma diagnostic layout/chunk matrices disagree")
    if len(step.noised_video_sigmas) != len(layouts) or len(step.noised_action_sigmas) != len(layouts):
        raise RuntimeError("actual sigma diagnostic requires one noised item per sample")
    metadata = _json_value(getattr(step, "prefix_low_noise", []))
    if metadata and len(metadata) != len(layouts):
        raise RuntimeError("actual sigma diagnostic prefix metadata differs from sample count")
    samples = []
    for index, layout in enumerate(layouts):
        vr, vc, _ = layout.video_metadata()
        ar, ac, _ = layout.action_metadata()
        vrows = step.noised_video_sigmas[index].detach().cpu().reshape(-1)
        arows = step.noised_action_sigmas[index].detach().cpu().reshape(-1)
        if len(vrows) != len(vr) or len(arows) != len(ar):
            raise RuntimeError("actual sigma diagnostic RF row count differs from layout")
        if not torch.isfinite(vrows).all() or not torch.isfinite(arows).all():
            raise FloatingPointError("actual sigma diagnostic contains nonfinite RF sigmas")
        if torch.count_nonzero(vrows[vr == CONDITION_VIDEO]) or torch.count_nonzero(arows[ar == STATE]):
            raise RuntimeError("actual sigma diagnostic observed noised U/S conditioning rows")
        prefix = metadata[index] if metadata else None
        chunks = []
        for boundary in layout.boundaries:
            chunk = boundary.chunk_id
            vi, ai = (vr == VIDEO) & (vc == chunk), (ar == ACTION) & (ac == chunk)
            if not vi.any() or not ai.any():
                raise RuntimeError("actual sigma diagnostic found a block without both future modalities")
            if not torch.all(vrows[vi] == video[index, chunk]) or not torch.all(arows[ai] == action[index, chunk]):
                raise RuntimeError("actual sigma diagnostic RF row sigmas differ from prepared chunk values")
            chunks.append(dict(
                chunk_id=chunk, video_sigma=float(vrows[vi][0]), action_sigma=float(arows[ai][0]),
                is_low_noise_prefix=bool(prefix and chunk in prefix["prefix_chunks"]),
                video_future_frames=int(vi.sum()), action_future_rows=int(ai.sum()),
                condition_video_sigmas=vrows[(vr == CONDITION_VIDEO) & (vc == chunk)].tolist(),
                state_action_sigmas=arows[(ar == STATE) & (ac == chunk)].tolist(),
            ))
        if prefix and (prefix["sample_index"] != index or prefix["n_chunks"] != len(chunks)):
            raise RuntimeError("actual sigma diagnostic prefix metadata differs from real chunk layout")
        samples.append(dict(sample_index=index, n_chunks=len(chunks),
                            prefix_low_noise=prefix, chunks=chunks))
    return dict(schema="ar_v03_actual_sigma_v1", source="official_RF_noised_data_at_net_pre_hook",
                excludes_unused_slot_zero_and_padding=True, samples=samples)


class ARV03SmokeMonitor(Callback):
    """Observe forward counts, fair window identity, raw grads and step time.

    Installed only for the short comparison. No extra backward, gradient edit,
    loss edit, or official logger replacement is performed. Raw gradients are
    read in on_after_backward, before official GradClip runs; the official
    preclip_norm is read separately after the optimizer. The timer covers
    training preparation/forward/backward/optimizer, matching FormalMonitor.
    """

    def __init__(self, *, output_dir=None, run_label="", expected_forwards=None,
                 fail_on_bad_gradients=True):
        super().__init__()
        self.output_dir = output_dir
        self.run_label = run_label
        self.expected_forwards = expected_forwards
        self.fail_on_bad_gradients = fail_on_bad_gradients
        self.parameter_fragments = ("action2llm", "llm2action", "action_modality_embed",
                                    "action_state_embed", "vision_condition_embed", "vae2llm", "llm2vae")
        self._hook = None
        self._in_forward = False
        self._step_model = None

    def on_train_start(self, model, iteration=0):
        if int(self.config.trainer.grad_accum_iter) != 1:
            raise ValueError("the fair 20-step diagnostic requires grad_accum_iter=1")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.root = Path(self.output_dir or self.config.job.path_local)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"smoke_monitor_rank{self.rank:03d}.jsonl"
        strategy = model.config.causal_training_strategy
        self.require_sigma_actual = bool(getattr(model.config, "prefix_low_noise_enabled", False))
        if self.expected_forwards is None:
            self.expected_forwards = 1 if strategy == "diffusion_forcing" else 2
        self.clip = next((callback for callback in self.trainer.callbacks._callbacks
                          if isinstance(callback, GradClip)), None)
        self.parameters = {fragment: [(name, param) for name, param in model.net.named_parameters()
                                      if fragment in name and param.requires_grad]
                           for fragment in self.parameter_fragments}
        missing = [fragment for fragment, params in self.parameters.items() if not params]
        if missing:
            raise ValueError(f"smoke diagnostic parameter groups missing: {missing}")
        self._hook = model.net.register_forward_pre_hook(self._before_net, with_kwargs=True)
        parallel = getattr(model.config, "parallelism", None)
        hardware = dict(
            hostname=socket.gethostname(), rank=self.rank,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            pid=os.getpid(), cpu_affinity=len(os.sched_getaffinity(0)),
            cpu_quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip()
                if Path("/sys/fs/cgroup/cpu.max").exists() else None,
            load_average=list(os.getloadavg()),
            threads={key: os.environ.get(key) for key in
                     ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
            parallelism={key: getattr(parallel, key, None) for key in
                         ("data_parallel_shard_degree", "data_parallel_replicate_degree",
                          "context_parallel_shard_degree")},
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            hardware.update(gpu_name=props.name, gpu_total_memory_bytes=props.total_memory)
        self._write(dict(event="start", run_label=self.run_label, iteration=iteration,
                         causal_training_strategy=strategy, expected_forwards=self.expected_forwards,
                         hardware=hardware,
                         gradient_parameters={key: [name for name, _ in params]
                                              for key, params in self.parameters.items()}))

    def _write(self, row):
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def _before_net(self, module, args, kwargs):
        if not self._in_forward:
            return
        self.forward_calls += 1
        packed = kwargs.get("packed_seq", args[0] if args else None)
        self.forward_packs.append(dict(
            sequence_length=int(getattr(packed, "sequence_length", 0)),
            sample_lens=list(getattr(packed, "sample_lens", ())),
            gen_tokens=sum(layout.num_tokens for layout in getattr(packed, "joint_layouts", ())),
            text_lengths=list(getattr(packed, "joint_text_lengths", ())),
        ))
        if self.forward_calls == 1:
            diagnostic_start = time.perf_counter()
            model = self._step_model() if self._step_model is not None else None
            self.sigma_actual = actual_sigma_snapshot(
                getattr(model, "_ar_step", None), getattr(packed, "joint_layouts", ()))
            if self.require_sigma_actual and self.sigma_actual is None:
                raise RuntimeError("prefix smoke diagnostic did not observe actual RF sigmas")
            self.sigma_diagnostic_seconds = time.perf_counter() - diagnostic_start

    def on_training_step_start(self, model, data, iteration=0):
        metadata_start = time.perf_counter()
        self.identity = batch_identity(data)
        self.metadata_diagnostic_seconds = time.perf_counter() - metadata_start
        self.forward_calls, self.forward_packs = 0, []
        self._step_model = weakref.ref(model)
        self.sigma_actual = None
        self.sigma_diagnostic_seconds = 0.0
        self.gradients = None
        self.gradient_diagnostic_seconds = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_before_forward(self, iteration=0):
        self._in_forward = True

    def on_after_forward(self, iteration=0):
        self._in_forward = False
        self._step_model = None
        if self.forward_calls != self.expected_forwards:
            raise RuntimeError(f"expected {self.expected_forwards} net forwards, observed {self.forward_calls}")

    @torch.no_grad()
    def on_after_backward(self, model, iteration=0):
        diagnostic_start = time.perf_counter()
        measurements = []
        for params in self.parameters.values():
            grads = [param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
                     for _, param in params if param.grad is not None]
            device = grads[0].device if grads else next(model.net.parameters()).device
            norm_squared = torch.zeros((), device=device, dtype=torch.float64)
            nonfinite = torch.zeros_like(norm_squared)
            nonempty = torch.zeros_like(norm_squared)
            for grad in grads:
                values = grad.detach().float()
                norm_squared += values.square().sum().double()
                nonfinite += (~torch.isfinite(values)).any().double()
                nonempty += int(values.numel() > 0)
            measurements.append(torch.stack((norm_squared, nonfinite, nonempty)))
        summary = torch.stack(measurements)
        # Local shards may be empty. Reduce only seven scalar summaries so
        # finite/nonzero requirements apply to the actual distributed model.
        if dist.is_initialized():
            dist.all_reduce(summary, op=dist.ReduceOp.SUM)
        rows = summary.cpu().tolist()
        self.gradients = {}
        for group, (squared, nonfinite, nonempty) in zip(self.parameters, rows):
            finite = nonfinite == 0 and math.isfinite(squared)
            self.gradients[group] = dict(l2_norm=math.sqrt(squared) if finite else None,
                                         finite=finite, nonzero=finite and squared > 0,
                                         nonempty_local_gradients=int(nonempty))
        self.gradient_diagnostic_seconds = time.perf_counter() - diagnostic_start

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.step_seconds = time.perf_counter() - self.started

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        norm = None
        if self.clip is not None:
            value = self.clip._last_global_norm.get(self.clip._state_key)
            if value is not None:
                norm = float(value)
        losses = {key: float(output_batch[key].detach().float().mean()) for key in
                  ("flow_matching_loss_vision", "flow_matching_loss_action") if key in output_batch}
        bad = [key for key, value in (self.gradients or {}).items()
               if not value["finite"] or not value["nonzero"]]
        self._write(dict(
            event="step", run_label=self.run_label, step=iteration, rank=self.rank,
            net_forward_calls=self.forward_calls, forward_packs=self.forward_packs,
            batch=self.identity, train_step_seconds=self.step_seconds,
            sigma_actual=self.sigma_actual, sigma_diagnostic_seconds=self.sigma_diagnostic_seconds,
            gradient_diagnostic_seconds=self.gradient_diagnostic_seconds,
            metadata_diagnostic_seconds=self.metadata_diagnostic_seconds,
            gradient_stage="after_backward_before_optimizer_hook_and_grad_clip",
            gradient_norm_definition="sqrt(sum squared local-shard gradients across ranks); replicas counted per replica",
            raw_gradient_groups=self.gradients, bad_gradient_groups=bad,
            official_preclip_norm=norm if norm is None or math.isfinite(norm) else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else None,
            loss_total=float(loss.detach()), **losses,
        ))
        if self.fail_on_bad_gradients and (self.gradients is None or bad):
            raise FloatingPointError(f"short-test raw-gradient diagnostic failed: {bad or 'no gradients observed'}")

    def on_train_end(self, model, iteration=0):
        self._in_forward = False
        self._step_model = None
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
