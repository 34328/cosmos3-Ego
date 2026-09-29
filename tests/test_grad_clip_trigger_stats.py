import math
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks import grad_clip as grad_clip_module
from cosmos_framework.callbacks.grad_clip import GradClip


def _reference_logs(norms, clip_norm, logging_iter):
    """The original per-step ``.item()`` bookkeeping, replayed on host floats."""
    window = total = last_triggered = 0
    last_scale = 1.0
    logs = []
    for step, norm_value in enumerate(norms, start=1):
        triggered = int(math.isfinite(norm_value) and norm_value > clip_norm)
        last_triggered = triggered
        window += triggered
        total += triggered
        last_scale = min(1.0, clip_norm / (norm_value + 1.0e-6)) if math.isfinite(norm_value) else 0.0
        if step % logging_iter == 0:
            logs.append(
                {
                    "grad_clip/triggered": last_triggered,
                    "grad_clip/trigger_count_window": window,
                    "grad_clip/trigger_count_since_process_start": total,
                    "grad_clip/applied_scale": last_scale,
                }
            )
            window = 0
    return logs


def _module_with_param(size):
    param = torch.nn.Parameter(torch.zeros(size))
    model = torch.nn.Module()
    model.register_parameter("weight", param)
    return model, param


def _callback(monkeypatch, logged, clip_norm, logging_iter, force_finite=False):
    monkeypatch.setattr(
        grad_clip_module, "wandb", SimpleNamespace(run=True, log=lambda d, step: logged.append((step, d)))
    )
    callback = GradClip(clip_norm=clip_norm, force_finite=force_finite, track_per_modality=False)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=logging_iter))
    return callback


def test_trigger_stats_sync_only_on_logging_steps(monkeypatch):
    clip_norm, logging_iter = 1.0, 2
    norms = [0.5, 2.0, 4.0, 0.5, 3.0, math.inf, 0.25, 1.0]
    logged = []
    callback = _callback(monkeypatch, logged, clip_norm, logging_iter)
    model, param = _module_with_param(4)

    def forbidden_item(self):
        raise AssertionError("on_before_optimizer_step must not synchronize via .item()")

    for step, norm_value in enumerate(norms, start=1):
        # ||[c/2]*4|| == c exactly; an inf entry makes the global norm inf.
        if math.isfinite(norm_value):
            param.grad = torch.full((4,), norm_value / 2)
        else:
            param.grad = torch.tensor([math.inf, 0.0, 0.0, 0.0])
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", forbidden_item)
            callback.on_before_optimizer_step([model], None, None, None, iteration=step - 1)
        callback.on_training_step_end([model], {}, {}, torch.zeros(()), iteration=step)

    expected = _reference_logs(norms, clip_norm, logging_iter)
    assert [step for step, _ in logged] == [2, 4, 6, 8]
    for (_, actual), reference in zip(logged, expected, strict=True):
        for key, value in reference.items():
            assert actual[key] == value, key
            assert type(actual[key]) is type(value), key


def test_trigger_stats_default_when_no_step_recorded(monkeypatch):
    logged = []
    callback = _callback(monkeypatch, logged, clip_norm=1.0, logging_iter=1)
    callback.on_training_step_end([], {}, {}, torch.zeros(()), iteration=1)
    assert logged == [(1, {"iteration": 1})]


@pytest.mark.parametrize("clip_norm", [0.1, 0.3, 1.0])
def test_trigger_matches_host_comparison_at_float32_boundary(monkeypatch, clip_norm):
    # float32(clip_norm) may round above or below the float64 threshold; the
    # device-side test must agree with the original ``float(norm) > clip_norm``.
    norm32 = torch.tensor(clip_norm, dtype=torch.float32)
    logged = []
    callback = _callback(monkeypatch, logged, clip_norm, logging_iter=1)
    model, param = _module_with_param(1)
    param.grad = norm32.reshape(1).clone()
    callback.on_before_optimizer_step([model], None, None, None, iteration=0)
    callback.on_training_step_end([model], {}, {}, torch.zeros(()), iteration=1)
    expected = int(float(norm32) > clip_norm)
    assert logged[0][1]["grad_clip/triggered"] == expected
    assert logged[0][1]["grad_clip/trigger_count_window"] == expected
