"""Optimizer coverage, raw gradient checks, and reproducible input audits."""

from datetime import timedelta
import json
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cosmos3_joint_video_hand_pose.src.ar_v02_contract import assert_optimizer_covers_trainable


class TwoHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.4))
        self.video = torch.nn.Parameter(torch.tensor(0.2))
        self.action = torch.nn.Parameter(torch.tensor(-0.1))

    def forward(self, v, a):
        return (self.shared * v + self.video).square(), (self.shared * a + self.action).square()


def test_optimizer_detects_frozen_new_parameters_and_duplicates():
    model = TwoHead()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert_optimizer_covers_trainable(model, optimizer, required_names=["action"])
    model.action.requires_grad_(False)  # emulate keys_to_select
    with pytest.raises(ValueError, match="required"):
        assert_optimizer_covers_trainable(model, optimizer, required_names=["action"])
    model.action.requires_grad_(True)
    optimizer.param_groups[0]["params"].append(model.action)
    with pytest.raises(ValueError, match="duplicate"):
        assert_optimizer_covers_trainable(model, optimizer)


def test_optimizer_container_coverage_and_cross_child_duplicates():
    from cosmos_framework.utils.generator.optimizer import OptimizersContainer

    net = torch.nn.Linear(2, 1)
    # Use the actual wrapper type without creating GPU fused optimizers.
    container = object.__new__(OptimizersContainer)
    container.optimizers = [torch.optim.SGD([net.weight], lr=0.1), torch.optim.SGD([net.bias], lr=0.1)]
    assert_optimizer_covers_trainable(net, container)
    container.optimizers[1].param_groups[0]["params"].append(net.weight)
    with pytest.raises(ValueError, match="duplicate"):
        assert_optimizer_covers_trainable(net, container)


def test_raw_gradient_check_rejects_inf_and_absent_gradients():
    from cosmos3_joint_video_hand_pose.src.ar_v02_contract import assert_finite_gradients

    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(RuntimeError, match="no gradients"):
        assert_finite_gradients([parameter])
    parameter.grad = torch.ones(2)
    assert assert_finite_gradients([parameter]) == 1
    parameter.grad[0] = float("inf")
    with pytest.raises(FloatingPointError, match="raw gradients"):
        assert_finite_gradients([parameter])


def test_framework_clean_kv_write_preserves_gradient_and_pass2_is_read_only():
    from cosmos_framework.model.generator.utils.kv_cache import TeacherForcingMemoryState

    memory = object.__new__(TeacherForcingMemoryState)
    memory.pass_number = 1
    memory.selected_clean_gen_token_indexes = None
    memory.detach_clean_kv = False
    memory.vision_token_shapes = [(2, 1, 1)]
    memory.num_action_tokens_per_supertoken = 0
    memory.null_action_supertokens = False
    memory._clean_gen_kv = [None]
    memory.target_only_no_text_pass2 = False
    memory.has_new_caption_py = False
    k = torch.ones(1, 2, 1, 2, requires_grad=True)
    v = torch.full_like(k, 2.0, requires_grad=True)
    memory.write_for_layer(0, (k, v, None, None))
    stored = memory._clean_gen_kv[0]
    (stored[0].square().sum() + stored[1].sum()).backward()
    torch.testing.assert_close(k.grad, 2 * k)
    torch.testing.assert_close(v.grad, torch.ones_like(v))
    memory.pass_number = 2
    memory.write_for_layer(0, (k * 100, v * 100, None, None))
    assert memory._clean_gen_kv[0] is stored


def test_smoke_digest_is_typed_bf16_safe_and_read_only():
    from cosmos3_joint_video_hand_pose.src.smoke_train import input_digest

    x = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).requires_grad_()
    before, rng = x.detach().clone(), torch.get_rng_state().clone()
    value = input_digest({"x": x, "empty": torch.empty(0), "scalar": torch.tensor(2.0)})
    assert value == input_digest({"scalar": torch.tensor(2.0), "empty": torch.empty(0), "x": x.clone()})
    assert input_digest(x) != input_digest(x.float())
    assert input_digest(x) != input_digest(x.reshape(3, 2))
    assert input_digest([1, 23]) != input_digest([12, 3])
    torch.testing.assert_close(x, before)
    assert torch.equal(torch.get_rng_state(), rng)
    x.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_smoke_audit_distinguishes_window_source_cfg_and_content():
    from copy import deepcopy
    from cosmos3_joint_video_hand_pose.src.smoke_train import batch_input_audit

    batch = {
        "video": [torch.zeros(1, 3, 2, 2)],
        "action": [torch.ones(2, 57)],
        "text_token_ids": [torch.tensor([[1, 7, 3]])],
        "window_start": [[10]],
        "source_frame_indices": [torch.tensor([[10, 14, 18]])],
        "ar_boundary_states": [torch.zeros(1, 64)],
    }
    baseline = batch_input_audit(batch)
    assert baseline["samples"][0]["window_start"] == 10
    assert baseline["samples"][0]["source_frame_indices"]["count"] == 3
    for key in ("window_start", "source_frame_indices", "text_token_ids", "video", "ar_boundary_states"):
        changed = deepcopy(batch)
        if key == "window_start":
            changed[key][0][0] += 1
        else:
            changed[key][0].reshape(-1)[0] += 1
        assert batch_input_audit(changed)["raw_batch_sha256"] != baseline["raw_batch_sha256"]
    assert "tensor(" not in json.dumps(baseline)


def test_smoke_denoise_audit_captures_noise_sigma_and_preserves_backward():
    from cosmos3_joint_video_hand_pose.src.smoke_train import install_denoise_input_audit
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
    from cosmos_framework.data.generator.sequence_packing.modality import ModalityData

    x = torch.ones(2, 3, dtype=torch.bfloat16, requires_grad=True)
    packed = PackedSequence(
        action=ModalityData(tokens=[x], timesteps=torch.tensor([0.5])), text_ids=torch.tensor([1, 2])
    )
    received = []

    def denoise(**kwargs):
        assert kwargs["data_batch_packed"] is packed
        return packed.action.tokens[0].float().square().sum()

    model = SimpleNamespace(denoise=denoise, _ar_step=SimpleNamespace(chunk_size=2, window=15))
    install_denoise_input_audit(model, received.append)
    memory = SimpleNamespace(pass_number=1)
    rng = torch.get_rng_state().clone()
    loss = model.denoise(data_batch_packed=packed, memory=memory)
    packed.action.timesteps = torch.tensor([0.7])
    model.denoise(data_batch_packed=packed, memory=memory)
    assert received[0]["training_input_sha256"] != received[1]["training_input_sha256"]
    assert received[0]["action_tokens_sha256"] == received[1]["action_tokens_sha256"]
    packed.action.tokens = [x + 1]
    memory.pass_number = 2
    model.denoise(data_batch_packed=packed, memory=memory)
    assert received[1]["action_tokens_sha256"] != received[2]["action_tokens_sha256"]
    assert received[2]["pass_number"] == 2
    assert torch.equal(torch.get_rng_state(), rng)
    loss.backward()
    torch.testing.assert_close(x.grad, 2 * torch.ones_like(x))


def test_fixed_input_roundtrip_restores_all_cpu_rng_and_rejects_tamper(tmp_path):
    import random
    import numpy as np
    from cosmos3_joint_video_hand_pose.src.smoke_train import (
        capture_rng_state,
        restore_rng_state,
        save_fixed_input,
        load_fixed_input,
        input_digest,
    )

    batch = dict(
        video=[torch.zeros(1, 3, 2, 2)],
        action=[torch.ones(2, 57)],
        text_token_ids=[torch.tensor([1, 2])],
        window_start=[[5]],
    )
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    path = tmp_path / "fixed.pt"
    save_fixed_input(path, batch, state, rank=0, world_size=8, iteration=2)
    payload = load_fixed_input(path, rank=0, world_size=8, iteration=2)
    restore_rng_state(payload["rng"])
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    assert input_digest(payload["rng"]) == input_digest(state)
    with pytest.raises(FileExistsError):
        save_fixed_input(path, batch, state, rank=0, world_size=8, iteration=2)
    with pytest.raises(ValueError, match="iteration"):
        load_fixed_input(path, rank=0, world_size=8, iteration=3)
    payload["batch"]["video"][0].add_(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="batch digest"):
        load_fixed_input(path, rank=0, world_size=8, iteration=2)


def test_gradient_audit_is_raw_readonly_and_groups_layers():
    from cosmos3_joint_video_hand_pose.src.smoke_train import gradient_audit

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 1, bias=False)])
    model.layers[0].weight.grad = torch.tensor([[6.0, 8.0]])
    state = torch.get_rng_state().clone()
    before = model.layers[0].weight.grad.clone()
    result = gradient_audit(model, scale=2.0)
    assert result["global_l2"] == 5.0
    assert result["layers"] == {"layers.0": 5.0}
    assert torch.equal(before, model.layers[0].weight.grad)
    assert torch.equal(state, torch.get_rng_state())
    model.layers[0].weight.grad[0, 0] *= -1
    changed = gradient_audit(model, scale=2.0)
    assert changed["global_l2"] == result["global_l2"]
    assert changed["local_gradient_sha256"] != result["local_gradient_sha256"]


def _gradient_audit_shard_worker(rank, init_file):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard, Replicate
    from cosmos3_joint_video_hand_pose.src.smoke_train import gradient_audit

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        model = torch.nn.Module()
        for name, placement, values in (("shard", Shard(0), [3.0, 4.0, 0.0, 0.0]), ("replica", Replicate(), [2.0])):
            tensor = torch.tensor(values)
            parameter = torch.nn.Parameter(distribute_tensor(torch.zeros_like(tensor), mesh, [placement]))
            parameter.grad = distribute_tensor(tensor, mesh, [placement])
            model.register_parameter(name, parameter)
        result = gradient_audit(model)
        assert result["global_l2"] == pytest.approx(29**0.5)
        assert result["parameters"]["shard"]["global_l2"] == 5.0
        assert result["parameters"]["replica"]["global_l2"] == 2.0
    finally:
        dist.destroy_process_group()


def test_gradient_audit_counts_fsdp_shards_and_replicas_once(tmp_path):
    mp.spawn(_gradient_audit_shard_worker, args=(str(tmp_path / "grad_gloo"),), nprocs=2, join=True)


def test_fixed_vae_replay_keeps_metadata_and_freezes_actual_denoiser_input(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from cosmos3_joint_video_hand_pose.src import smoke_train

    monkeypatch.setattr(smoke_train.dist, "get_rank", lambda: 0)
    context = {"microstep": 0}
    calls = []

    def encode():
        calls.append("encoded")
        return [torch.randn(1, 2, 3)]

    capture = SimpleNamespace(_encode_vision_x0_tokens=encode)
    smoke_train.install_fixed_vae_inputs(capture, tmp_path, replay=False, context=context)
    expected = capture._encode_vision_x0_tokens()
    after = torch.get_rng_state().clone()
    replay = SimpleNamespace(_encode_vision_x0_tokens=encode)
    smoke_train.install_fixed_vae_inputs(replay, tmp_path, replay=True, context=context)
    actual = replay._encode_vision_x0_tokens()
    assert calls == ["encoded", "encoded"]
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), after, atol=0, rtol=0)
    import json

    records = json.loads((tmp_path / "rank00000.step00000.vae.replay.json").read_text())
    assert records[0]["max_abs"] > 0 and records[0]["equal"] is False
