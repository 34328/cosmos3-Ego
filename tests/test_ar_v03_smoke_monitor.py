"""Native callback ordering, exact-window identity, and read-only diagnostics."""

import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.grad_clip import GradClip
from cosmos3_joint_video_hand_pose.src import ar_v03_smoke_monitor as module
from cosmos3_joint_video_hand_pose.src.ar_v03_smoke_monitor import ARV03SmokeMonitor, batch_identity


class SmallJointNet(torch.nn.Module):
    """A real joint forward/backward with all seven monitored parameter groups."""
    def __init__(self):
        super().__init__()
        self.action2llm = torch.nn.Linear(4, 8)
        self.llm2action = torch.nn.Linear(8, 4)
        self.vae2llm = torch.nn.Linear(5, 8)
        self.llm2vae = torch.nn.Linear(8, 5)
        self.action_modality_embed = torch.nn.Parameter(torch.randn(8) * 0.1)
        self.action_state_embed = torch.nn.Parameter(torch.randn(8) * 0.1)
        self.vision_condition_embed = torch.nn.Parameter(torch.randn(8) * 0.1)

    def forward(self, packed_seq):
        hidden = torch.tanh(self.action2llm(packed_seq.action) + self.vae2llm(packed_seq.vision)
                            + self.action_modality_embed + self.action_state_embed
                            + self.vision_condition_embed)
        return self.llm2action(hidden), self.llm2vae(hidden)


def setup_monitor(tmp_path, monkeypatch, strategy="diffusion_forcing"):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(42)
    net = SmallJointNet()
    model = SimpleNamespace(net=net, config=SimpleNamespace(causal_training_strategy=strategy, parallelism=None))
    clip = GradClip(clip_norm=1.0)
    clip._last_global_norm[clip._state_key] = torch.tensor(3.25)
    monitor = ARV03SmokeMonitor(output_dir=str(tmp_path), run_label=strategy)
    monitor.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=1),
                                    job=SimpleNamespace(path_local=str(tmp_path)))
    monitor.trainer = SimpleNamespace(callbacks=SimpleNamespace(_callbacks=[clip]))
    monitor.on_train_start(model)
    batch = dict(sample_id=[["episode:2:0:100"]], dataset_index=[torch.tensor([7])],
                 window_start=[torch.tensor([4])], clip_frames=[torch.tensor([33])],
                 source_frame_indices=[torch.arange(4, 69, 2)],
                 action_source_frame_indices=[torch.arange(4, 69)],
                 _num_tokens=400, _ar_c4_raw_forward_tokens=400)
    packed = SimpleNamespace(action=torch.randn(3, 4), vision=torch.randn(3, 5),
                             sequence_length=200, sample_lens=[200], joint_layouts=[], joint_text_lengths=[3])
    return monitor, model, clip, batch, packed


def forward_and_backward(monitor, model, batch, packed, forwards):
    monitor.on_training_step_start(model, batch)
    monitor.on_before_forward()
    action_losses, video_losses = [], []
    for _ in range(forwards):
        action, video = model.net(packed_seq=packed)
        action_losses.append((action - 0.4).square().mean())
        video_losses.append((video + 0.2).square().mean())
    action_loss, video_loss = sum(action_losses), sum(video_losses)
    loss = action_loss + video_loss
    monitor.on_after_forward()
    # A module invocation in the backward/checkpoint interval is excluded.
    model.net(packed)
    loss.backward()
    output = dict(flow_matching_loss_action=action_loss, flow_matching_loss_vision=video_loss)
    return output, loss


@pytest.mark.parametrize("strategy,forwards", [("diffusion_forcing", 1), ("teacher_forcing", 2)])
def test_callback_records_real_gradients_without_mutation_and_excludes_recompute(tmp_path, monkeypatch, strategy, forwards):
    monitor, model, clip, batch, packed = setup_monitor(tmp_path, monkeypatch, strategy)
    output, loss = forward_and_backward(monitor, model, batch, packed, forwards)
    before = [param.grad.clone() for param in model.net.parameters()]
    monitor.on_after_backward(model)
    for param, expected in zip(model.net.parameters(), before):
        torch.testing.assert_close(param.grad, expected, atol=0, rtol=0)
    assert monitor.forward_calls == forwards
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.01)
    parameter_before = model.net.action2llm.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(model.net.action2llm.weight, parameter_before)
    monitor.on_training_step_batch_end(model, batch, output, loss)
    monitor.on_training_step_end(model, batch, output, loss, iteration=1)
    monitor.on_train_end(model, iteration=1)
    assert not model.net._forward_pre_hooks
    rows = [json.loads(line) for line in monitor.path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "step"]
    row = rows[-1]
    assert row["net_forward_calls"] == forwards and len(row["forward_packs"]) == forwards
    assert row["batch"]["sample_id"] == [["episode:2:0:100"]]
    assert row["batch"]["window_start"] == [4]
    assert len(row["raw_gradient_groups"]) == 7
    assert all(value["finite"] and value["nonzero"] and value["l2_norm"] > 0
               for value in row["raw_gradient_groups"].values())
    assert row["official_preclip_norm"] == 3.25
    torch.testing.assert_close(clip._last_global_norm[clip._state_key], torch.tensor(3.25), atol=0, rtol=0)
    assert row["flow_matching_loss_action"] == float(output["flow_matching_loss_action"].detach())
    assert row["flow_matching_loss_vision"] == float(output["flow_matching_loss_vision"].detach())
    assert row["train_step_seconds"] >= row["gradient_diagnostic_seconds"] >= 0


def test_step_timer_finishes_after_optimizer_boundary(tmp_path, monkeypatch):
    monitor, model, _, batch, packed = setup_monitor(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    output, loss = forward_and_backward(monitor, model, batch, packed, 1)
    clock[0] = 2.0
    monitor.on_after_backward(model)
    # Native Trainer performs optimizer.step before batch_end, which must
    # remain inside this interval even when callbacks add diagnostics.
    clock[0] = 5.0
    torch.optim.SGD(model.net.parameters(), lr=0.01).step()
    clock[0] = 9.0
    monitor.on_training_step_batch_end(model, batch, output, loss)
    monitor.on_training_step_end(model, batch, output, loss, iteration=1)
    monitor.on_train_end(model, iteration=1)
    assert json.loads(monitor.path.read_text().splitlines()[-1])["train_step_seconds"] == 9.0


@pytest.mark.parametrize("bad", ["zero", "nan"])
def test_bad_raw_gradient_is_reported_before_failure_without_repair(tmp_path, monkeypatch, bad):
    monitor, model, _, batch, packed = setup_monitor(tmp_path, monkeypatch)
    output, loss = forward_and_backward(monitor, model, batch, packed, 1)
    model.net.action_state_embed.grad.fill_(0 if bad == "zero" else float("nan"))
    original = model.net.action_state_embed.grad.clone()
    monitor.on_after_backward(model)
    torch.testing.assert_close(model.net.action_state_embed.grad, original, atol=0, rtol=0, equal_nan=True)
    monitor.on_training_step_batch_end(model, batch, output, loss)
    with pytest.raises(FloatingPointError, match="action_state_embed"):
        monitor.on_training_step_end(model, batch, output, loss, iteration=1)
    monitor.on_train_end(model, iteration=1)
    row = json.loads(monitor.path.read_text().splitlines()[-1])
    assert row["bad_gradient_groups"] == ["action_state_embed"]
    assert row["raw_gradient_groups"]["action_state_embed"]["nonzero"] is False
    assert row["raw_gradient_groups"]["action_state_embed"]["finite"] is (bad == "zero")
    assert not model.net._forward_pre_hooks


def test_forward_count_mismatch_rejects_hidden_clean_pass(tmp_path, monkeypatch):
    monitor, model, _, batch, packed = setup_monitor(tmp_path, monkeypatch)
    monitor.on_training_step_start(model, batch)
    monitor.on_before_forward()
    model.net(packed)
    model.net(packed)
    with pytest.raises(RuntimeError, match="expected 1 net forwards, observed 2"):
        monitor.on_after_forward()
    monitor.on_train_end(model)
    assert not model.net._forward_pre_hooks


def test_window_identity_distinguishes_segment_windows_and_source_paths():
    batch = dict(sample_id=[["same_segment"]], window_start=[torch.tensor([10])], clip_frames=[33],
                 source_frame_indices=[torch.arange(10, 75, 2)],
                 action_source_frame_indices=[torch.arange(10, 75)],
                 video=object(), action=object())
    original = batch_identity(batch)
    assert original == batch_identity(dict(reversed(list(batch.items()))))
    for key, change in (("window_start", [torch.tensor([11])]),
                        ("source_frame_indices", [torch.arange(11, 76, 2)]),
                        ("action_source_frame_indices", [torch.arange(11, 76)])):
        observed = batch_identity({**batch, key: change})
        assert observed["window_identity_sha256"] != original["window_identity_sha256"]
    assert "loaded_video_action_sha256" not in original  # metadata path never reads the loaded inputs


def test_missing_window_metadata_uses_loaded_input_hash_including_bfloat16():
    batch = dict(sample_id=["legacy_segment"], video=[torch.ones(3, 5, 2, 2, dtype=torch.bfloat16)],
                 action=[torch.ones(4, 64, dtype=torch.bfloat16)])
    original = batch_identity(batch)
    assert original == batch_identity(batch)
    assert "loaded_video_action_sha256" in original
    for key in ("video", "action"):
        changed = batch_identity({**batch, key: [batch[key][0] + 1]})
        assert changed["window_identity_sha256"] != original["window_identity_sha256"]
    with pytest.raises(ValueError, match="lacks exact-window metadata"):
        batch_identity({"sample_id": ["incomplete_segment"]})
