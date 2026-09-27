"""CPU tests for EgoVerseARModel's per-chunk noise mapping (no network is built)."""

from types import SimpleNamespace

import pytest
import torch

from cosmos3_joint_video_hand_pose.src import ar_model as ar_module
from cosmos3_joint_video_hand_pose.src.ar_attention import frame_chunk_ids
from cosmos3_joint_video_hand_pose.src.ar_model import ARStepContext, EgoVerseARModel


def _bare_model():
    model = object.__new__(EgoVerseARModel)
    torch.nn.Module.__init__(model)
    model.train_chunk_sizes = (1, 2, 3, 4)
    model.train_window_range = (4, 64)
    model.ar_seed = 42
    model._ar_step = None
    return model


def test_step_context_is_deterministic_and_in_range():
    model = _bare_model()
    seen = set()
    for iteration in range(200):
        a, b = model._sample_step_context(iteration), model._sample_step_context(iteration)
        assert (a.chunk_size, a.window) == (b.chunk_size, b.window)
        assert a.chunk_size in (1, 2, 3, 4) and 4 <= a.window <= 64
        seen.add(a.chunk_size)
    assert seen == {1, 2, 3, 4}


def test_vision_sigma_is_constant_within_each_chunk(monkeypatch):
    model = _bare_model()
    model._ar_step = ARStepContext(chunk_size=3, window=10)
    calls = {}

    def fake_parent(self, batch_size, is_image_batch, num_vision_latent_frames, resolutions=None, num_tokens=None, iteration=None):
        calls.update(batch_size=batch_size, frames=num_vision_latent_frames, resolutions=resolutions, tokens=num_tokens)
        sigmas = torch.arange(batch_size, dtype=torch.float32)[:, None] / 10  # distinct per (sample, chunk)
        return sigmas * 1000, sigmas

    monkeypatch.setattr(ar_module.OmniMoTCausalModel, "_get_train_noise_level_vision", fake_parent)
    timesteps, sigmas = model._get_train_noise_level_vision(
        batch_size=1, is_image_batch=False, num_vision_latent_frames=[9], resolutions=["480"], num_tokens=[2160]
    )
    chunks = frame_chunk_ids(9, 3)  # [0,1,1,1,2,2,2,3,3]
    assert calls == {"batch_size": 4, "frames": [9] * 4, "resolutions": ["480"] * 4, "tokens": [2160] * 4}
    assert sigmas.shape == (1, 9) and timesteps.shape == (1, 9)
    assert torch.allclose(sigmas[0], chunks.float() / 10)
    assert torch.allclose(timesteps, sigmas * 1000)
    assert torch.equal(model._ar_step.chunk_ids, chunks)


def test_action_sigma_is_drawn_per_chunk_and_expanded_per_row(monkeypatch):
    model = _bare_model()
    model._ar_step = ARStepContext(chunk_size=2, window=None, chunk_ids=frame_chunk_ids(5, 2))  # [0,1,1,2,2]

    def fake_parent(self, batch_size, iteration=None):
        sigmas = (torch.arange(batch_size, dtype=torch.float32)[:, None] + 1) / 10
        return sigmas * 1000, sigmas

    monkeypatch.setattr(ar_module.OmniMoTCausalModel, "_get_train_noise_level_action", fake_parent)
    timesteps, sigmas = model._get_train_noise_level_action(batch_size=1)
    assert torch.allclose(model._ar_step.action_sigmas, torch.tensor([[0.1, 0.2, 0.3]]))
    assert sigmas.shape == (1, 1) and float(sigmas) == pytest.approx(0.2)

    captured = {}

    def fake_add_noise(self, gen_data_clean, packed_sequence, sigmas, sigmas_action=None, **kwargs):
        captured["per_row"] = sigmas_action
        return SimpleNamespace()

    monkeypatch.setattr(ar_module.OmniMoTCausalModel, "_add_noise_to_input", fake_add_noise)
    model.rectified_flow_action = SimpleNamespace(noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)))
    k = 2
    packed = SimpleNamespace(
        num_action_tokens_per_supertoken=k,
        action=SimpleNamespace(noisy_frame_indexes=[torch.arange(k, 5 * k)], timesteps=None),
        uses_single_timestep=True,
    )
    gen = SimpleNamespace(x0_tokens_action=[torch.zeros(5 * k, 6)])
    model._add_noise_to_input(gen, packed, torch.zeros(1, 5))
    expected = torch.tensor([0.1] * k + [0.2] * (2 * k) + [0.3] * (2 * k))
    assert torch.allclose(captured["per_row"][0], expected)
    assert torch.allclose(packed.action.timesteps, expected[k:] * 1000)
    assert packed.uses_single_timestep is False
