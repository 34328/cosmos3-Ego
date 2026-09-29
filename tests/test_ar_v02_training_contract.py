"""CPU contract tests; real two-rank Gloo/DDP, no accelerator required."""

from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos3_joint_video_hand_pose.src.ar_v02_contract import (
    ARTrainingContract,
    GlobalSampleMeanWindow,
    assert_optimizer_covers_trainable,
)
from cosmos3_joint_video_hand_pose.src.loss import whole_video_flow_loss


def test_video_condition_and_sigma_weight_alignment():
    p = torch.ones(2, 3, 1, 1, requires_grad=True)
    p.data[:, 0] = 1e30
    loss, stats = whole_video_flow_loss(
        pred=[p],
        target=[torch.zeros_like(p)],
        condition_mask=[torch.tensor([1, 0, 0])],
        time_weight=lambda *args: torch.tensor([9.0, 2.0, 4.0]),
    )
    torch.testing.assert_close(loss, torch.tensor(3.0))
    torch.testing.assert_close(stats["per_sample_losses"], torch.tensor([3.0]))
    loss.backward()
    assert torch.count_nonzero(p.grad[:, 0]) == 0
    torch.testing.assert_close(p.grad[:, 1], torch.ones(2, 1, 1))
    torch.testing.assert_close(p.grad[:, 2], torch.full((2, 1, 1), 2.0))


def test_window_uses_update_counts_not_average_of_microbatch_means():
    p = torch.tensor(0.4, requires_grad=True)
    window = GlobalSampleMeanWindow([[1, 0], [3, 2]])
    losses = []
    for vx, ax in (([1.0], []), ([2.0, 3.0, 4.0], [7.0, 8.0])):
        v = (p * torch.tensor(vx)).square()
        a = (p * torch.tensor(ax)).square()
        loss, _ = window.reduce(v, torch.ones(len(v), dtype=torch.bool), a, torch.ones(len(a), dtype=torch.bool))
        losses.append(loss / 2)  # Cosmos trainer division
    grad = torch.autograd.grad(sum(losses), p)[0]
    ref = (p * torch.tensor([1.0, 2.0, 3.0, 4.0])).square().mean() + 1.0 * (
        p * torch.tensor([7.0, 8.0])
    ).square().mean()
    torch.testing.assert_close(sum(losses), ref)
    torch.testing.assert_close(grad, torch.autograd.grad(ref, p)[0])
    assert window.complete
    with pytest.raises(RuntimeError, match="consumed"):
        window.reduce(v, v.bool(), a, a.bool())


def test_empty_modalities_are_differentiable_and_plans_checked():
    p = torch.tensor(3.0, requires_grad=True)
    w = GlobalSampleMeanWindow([[0, 0]])
    loss, _ = w.reduce(p[None] * 0, torch.tensor([False]), p[None] * 0, torch.tensor([False]))
    loss.backward()
    assert p.grad == 0
    w = GlobalSampleMeanWindow([[1, 0]])
    with pytest.raises(ValueError, match="planned"):
        w.reduce(p[None], torch.tensor([False]), p[None], torch.tensor([False]))
    assert w.position == 0


class TwoHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.4))
        self.video = torch.nn.Parameter(torch.tensor(0.2))
        self.action = torch.nn.Parameter(torch.tensor(-0.1))

    def forward(self, v, a):
        return (self.shared * v + self.video).square(), (self.shared * a + self.action).square()


# Counts differ across ranks, microsteps AND modalities. Rank 1 has no active
# action in the full window; rank 0 has no active video in its final microstep.
BATCHES = [
    [([1.0], [5.0, 7.0]), ([2.0, 3.0], [8.0]), ([], [9.0])],
    [([4.0, 5.0, 6.0], []), ([7.0], []), ([8.0, 9.0], [])],
]


def _distributed_worker(rank, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        from contextlib import nullcontext

        model = DDP(TwoHead())
        counts = [[len(v), len(a)] for v, a in BATCHES[rank]]
        w = GlobalSampleMeanWindow(counts)
        report = torch.zeros(2)
        for i, (vx, ax) in enumerate(BATCHES[rank]):
            # Keep both heads connected, including empty modalities.
            with model.no_sync() if i < 2 else nullcontext():
                v, a = model(torch.tensor(vx), torch.tensor(ax))
                loss, stats = w.reduce(v, torch.ones(len(v), dtype=torch.bool), a, torch.ones(len(a), dtype=torch.bool))
                (loss / 3).backward()
                report += torch.stack((stats["video_contribution"], stats["action_contribution"]))
        ref = TwoHead()
        all_v = torch.tensor([x for rank_batches in BATCHES for v, a in rank_batches for x in v])
        all_a = torch.tensor([x for rank_batches in BATCHES for v, a in rank_batches for x in a])
        rv, ra = ref(all_v, all_a)
        (rv.mean() + 1.0 * ra.mean()).backward()
        for actual, expected in zip(model.module.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-6)
        dist.all_reduce(report)
        torch.testing.assert_close(report, torch.stack((rv.mean(), ra.mean())).detach())
        # Different window lengths fail identically on both ranks before backward.
        with pytest.raises(ValueError, match="same number"):
            GlobalSampleMeanWindow([[1, 1]] * (rank + 1))
    finally:
        dist.destroy_process_group()


def test_two_rank_dynamic_pack_accumulation_matches_single_process(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def make_contract(tmp_path, *, value=1, schema="chunk_camera_v1"):
    state = tmp_path / f"state{value}.json"
    state.write_text(
        json.dumps(dict(schema=schema, frozen=True, split="train", manifest_sha256="a" * 64, stats={"center": [value]}))
    )
    action = tmp_path / "action.json"
    action.write_text(json.dumps({"stats": {"scale": [1.0]}}))
    return ARTrainingContract(state_normalizer=state, action_normalizer=action, manifest_sha256="a" * 64)


def test_checkpoint_binding_roundtrip_rejects_missing_changed_and_f0(tmp_path):
    source = torch.nn.Module()
    source.add_module("ar_training_contract", make_contract(tmp_path))
    saved = source.state_dict()
    target = torch.nn.Module()
    target.add_module("ar_training_contract", make_contract(tmp_path))
    target.load_state_dict(saved)
    with pytest.raises(ValueError, match="missing"):
        target.load_state_dict({}, strict=False)
    target.ar_training_contract = make_contract(tmp_path, value=2)
    with pytest.raises(ValueError, match="mismatch"):
        target.load_state_dict(saved, strict=False)
    saved["ar_training_contract._extra_state"]["layout"] = "old"
    assert source.ar_training_contract.get_extra_state()["layout"] == "joint_chunk_cond_v1"
    with pytest.raises(ValueError, match="F0"):
        make_contract(tmp_path, schema="ar_v02_boundary_state_f0_v1")


def test_dcp_contract_roundtrip_and_mismatch(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

    # A nested net reproduces the namespace retained by OmniMoT.
    model = torch.nn.Module()
    model.net = torch.nn.Module()
    model.net.ar_training_contract = make_contract(tmp_path)
    checkpoint = str(tmp_path / "checkpoint")
    dcp.save({"model": get_model_state_dict(model)}, checkpoint_id=checkpoint, no_dist=True)
    model.net.ar_training_contract = make_contract(tmp_path, value=2)
    state = get_model_state_dict(model)
    dcp.load({"model": state}, checkpoint_id=checkpoint, no_dist=True)
    with pytest.raises(ValueError, match="mismatch"):
        set_model_state_dict(model, state)
    model.net.ar_training_contract = make_contract(tmp_path)
    set_model_state_dict(model, state)


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


def test_mixin_bypasses_native_scaling_and_passes_coordinate_validity():
    from cosmos3_joint_video_hand_pose.src.model import EgoVerseLossMixin

    class DummyParent(torch.nn.Module):
        def _compute_losses(self, **kwargs):
            raise AssertionError("native loss scaling must not run")

        def _sample_level_loss_scale(self, **kwargs):
            raise AssertionError("native sample scaler must not run")

        def _loss_averaging_group(self):
            return None, 1

        def training_step(self, data, iteration):
            loss, logs = self._compute_losses(**data)
            return logs, loss

    class Model(EgoVerseLossMixin, DummyParent):
        def __init__(self):
            torch.nn.Module.__init__(self)

    model = Model()
    model.whole_action_loss = True
    model.config = SimpleNamespace(
        vision_gen=True,
        action_gen=True,
        rectified_flow_training_config=SimpleNamespace(
            sample_level_loss_averaging=True, loss_scale=1.0, action_loss_weight=1.0, image_loss_scale=None
        ),
    )
    model.rectified_flow_video = SimpleNamespace(
        train_time_weight=lambda t, kw: torch.ones_like(t),
        noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
    )
    model.tensor_kwargs_fp32 = dict(device="cpu", dtype=torch.float32)
    model._current_hand_visibility = [torch.ones(2, 2)] * 2
    v = [torch.ones(1, 2, 1, 1, requires_grad=True)] * 2
    a = [torch.full((2, 64), x, requires_grad=True) for x in (2.0, 100.0)]
    masks = [torch.ones(64, dtype=torch.bool), torch.zeros(64, dtype=torch.bool)]
    packed = SimpleNamespace(
        sample_lens=[1, 1],
        vision=SimpleNamespace(condition_mask=[torch.tensor([1, 0])] * 2),
        action=SimpleNamespace(
            raw_action_dim=[57, 57], condition_mask=[torch.tensor([1, 0])] * 2, action_valid_mask=masks
        ),
    )
    data = dict(
        out_net={"preds_vision": v, "preds_action": a},
        data_batch_packed=packed,
        gen_data_noised=SimpleNamespace(
            vt_target_vision=[torch.zeros_like(x) for x in v], vt_target_action=[torch.zeros_like(x) for x in a]
        ),
        timesteps=torch.tensor([[0.0, 100.0], [0.0, 200.0]]),
        is_image_batch=False,
    )
    output, loss = model.training_step(data, 0)
    torch.testing.assert_close(loss, torch.tensor(5.0))
    torch.testing.assert_close(output["_backward_loss"], loss)
    for key, expected in (("train_objective_numerator", loss.detach()),
                          ("train_objective_denominator", torch.ones_like(loss))):
        assert not output[key].requires_grad
        assert output[key].grad_fn is None
        assert output[key].shape == loss.shape
        assert output[key].dtype == loss.dtype
        assert output[key].device == loss.device
        torch.testing.assert_close(output[key], expected)
    # The logger's detached metadata must not change the training objective or
    # its gradient; compare with the existing two-term objective independently.
    from cosmos3_joint_video_hand_pose.src.loss import whole_action_flow_loss
    reference_video, _ = whole_video_flow_loss(
        pred=v, target=data["gen_data_noised"].vt_target_vision,
        condition_mask=packed.vision.condition_mask,
        time_weight=lambda *args: torch.ones(2),
    )
    reference_action, _ = whole_action_flow_loss(
        pred=a, target=data["gen_data_noised"].vt_target_action,
        condition_mask=packed.action.condition_mask,
        visibility=model._current_hand_visibility, valid_mask=masks,
    )
    reference = reference_video + reference_action
    torch.testing.assert_close(loss, reference)
    expected_grads = torch.autograd.grad(reference, [v[0], *a])
    output["_backward_loss"].backward()
    for tensor, expected in zip([v[0], *a], expected_grads, strict=True):
        torch.testing.assert_close(tensor.grad, expected)
    assert torch.count_nonzero(a[1].grad) == 0
    assert output["egoverse_global_action_samples"] == 1
    model.configure_ar_loss_accumulation(2)
    with pytest.raises(RuntimeError, match="begin_ar_loss_window"):
        model.training_step(data, 1)
    model.begin_ar_loss_window([[2, 1], [2, 1]], device="cpu")
    for i in range(2):
        output, loss = model.training_step(data, i)
        torch.testing.assert_close(output["_backward_loss"] / 2, loss)
        torch.testing.assert_close(output["train_objective_numerator"], loss.detach())
        torch.testing.assert_close(output["train_objective_denominator"], torch.ones_like(loss))
        assert not output["train_objective_numerator"].requires_grad
        assert not output["train_objective_denominator"].requires_grad
    assert model._ar_loss_window.complete


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


def make_callback(tmp_path):
    import hashlib
    from cosmos3_joint_video_hand_pose.src.ar_chunk_state import state_profile_sha256
    from cosmos3_joint_video_hand_pose.src.ar_v02_contract import ARTrainingContractCallback

    manifest = tmp_path / "windows.json"
    manifest.write_text("{}")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    profile = dict(
        schema="ar_v02_chunk_camera_state_v2",
        layout_version="joint_chunk_cond_v1",
        split="train",
        frozen=True,
        manifest_sha256=digest,
        fit_samples_sha256="b" * 64,
        method="piecewise_asinh_rot",
        camera_encoding="zero_identity",
        beta=1.0,
        stats=dict(center=[0.0] * 18, scale=[1.0] * 18),
    )
    profile["profile_sha256"] = state_profile_sha256(profile)
    state = tmp_path / "state_callback.json"
    state.write_text(json.dumps(profile))
    action = tmp_path / "action_callback.json"
    action.write_text(json.dumps({"stats": {"scale": [1.0]}}))
    callback = ARTrainingContractCallback(
        state_normalizer=state,
        action_normalizer=action,
        valid_windows_manifest=manifest,
        official_checkpoint=tmp_path / "official",
    )
    callback.config = SimpleNamespace(
        trainer=SimpleNamespace(grad_accum_iter=1),
        checkpoint=SimpleNamespace(strict_resume=True, keys_to_skip_loading=[]),
    )
    return callback


class CallbackModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 1)
        self.net.action_state_embed = torch.nn.Parameter(torch.zeros(2))
        self.net.vision_condition_embed = torch.nn.Parameter(torch.zeros(2))
        self.whole_action_loss = True
        self._ar_loss_window = None

    def configure_ar_loss_accumulation(self, count):
        self.accumulation = count


def bind_source(callback, path, *, official=False):
    import torch.distributed.checkpoint as dcp

    # This fixture exercises explicit model-only warm start, including legacy v1.
    # Full training resumes are covered separately below.
    source = SimpleNamespace(path=str(path), warm_start=True, uses_object_store=False, backend_key=None)
    callback.trainer = SimpleNamespace(
        checkpointer=SimpleNamespace(
            keys_to_resume_during_load=lambda: ({"model"}, source),
            get_storage_reader=lambda path, src: dcp.FileSystemReader(path),
        )
    )


def test_callback_official_init_save_preflight_resume_and_batch_binding(tmp_path):
    import torch.distributed.checkpoint as dcp

    callback = make_callback(tmp_path)
    model = CallbackModel()
    bind_source(callback, tmp_path / "official", official=True)
    callback.on_load_checkpoint_start(model)
    assert not hasattr(model.net, "ar_training_contract")
    callback.on_load_checkpoint_end(model, checkpoint_path=str(tmp_path / "official"))
    callback.on_train_start(model)
    callback.on_save_checkpoint_start(model)
    snapshot = {"model": model.state_dict()}
    callback.on_save_checkpoint(model, snapshot)
    run = tmp_path / "run"
    dcp.save(snapshot["model"], checkpoint_id=str(run / "model"), no_dist=True)

    resumed = CallbackModel()
    callback2 = make_callback(tmp_path)
    bind_source(callback2, run)
    callback2.on_load_checkpoint_start(resumed)
    # Preflight only reads the binding; full model weights remain untouched.
    callback2.on_load_checkpoint_end(resumed, iteration=50, checkpoint_path=str(run))
    payload = callback2.contract.get_extra_state()
    batch = dict(
        ar_layout_version=[payload["layout"]] * 2,
        ar_state_normalizer_sha256=[payload["artifacts"]["state_normalizer"]["sha256"]] * 2,
        ar_valid_windows_sha256=[payload["manifest_sha256"]] * 2,
    )
    callback2.on_training_step_batch_start(resumed, batch)
    batch["ar_layout_version"][1] = "joint_state_single_v1"
    with pytest.raises(ValueError, match="ar_layout_version"):
        callback2.on_training_step_batch_start(resumed, batch)
    callback2.config.trainer.grad_accum_iter = 3
    with pytest.raises(ValueError, match="grad_accum_iter=1"):
        callback2.on_train_start(resumed)
    assert resumed.accumulation == 1
    resumed._ar_loss_window = SimpleNamespace(complete=False)
    with pytest.raises(RuntimeError, match="partially"):
        callback2.on_save_checkpoint_start(resumed)


def test_callback_rejects_unbound_resume_and_changed_stats(tmp_path):
    import torch.distributed.checkpoint as dcp

    callback = make_callback(tmp_path)
    model = CallbackModel()
    run = tmp_path / "unbound"
    dcp.save(model.state_dict(), checkpoint_id=str(run / "model"), no_dist=True)
    bind_source(callback, run)
    # DCP represents missing metadata as CheckpointException (BaseException).
    from torch.distributed.checkpoint.api import CheckpointException

    with pytest.raises((ValueError, CheckpointException)):
        callback.on_load_checkpoint_start(model)
    good = tmp_path / "good"
    dcp.save(model.state_dict(), checkpoint_id=str(good / "model"), no_dist=True)
    callback.contract._expected["layout"] = "bad"
    bind_source(callback, good)
    with pytest.raises(ValueError, match="mismatch"):
        callback.on_load_checkpoint_start(CallbackModel())


def test_config_enables_formal_trainer_contract_and_exact_loader():
    from cosmos3_joint_video_hand_pose.src.config import _ar_v02_experiment
    from cosmos3_joint_video_hand_pose.src.ar_v02_contract import ARTrainingContractCallback
    from cosmos3_joint_video_hand_pose.src.dataloader_state import RecoverablePackingDataLoader

    config = _ar_v02_experiment(True)
    hook = config["trainer"]["callbacks"]["ar_v02_contract"]
    dataset = config["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
    assert hook["_target_"] is ARTrainingContractCallback
    assert hook["check_raw_gradients"] is True
    assert hook["action_normalizer"] == dataset["future_normalizer"]
    assert config["checkpoint"]["strict_resume"] is True
    assert issubclass(config["dataloader_train"]["_target_"], RecoverablePackingDataLoader)


class ExactDataset:
    def __init__(self):
        self.called = []
        self.corrupt = False

    def __getitem__(self, index):
        raise AssertionError("resume must NEVER redraw the random window")

    def get_item_at_window(self, index, *, window_start, source_frame_indices):
        import random
        import numpy as np

        self.called.append((index, int(window_start)))
        random.random()
        np.random.rand()
        torch.rand(1)
        start = int(window_start)
        return dict(
            dataset_index=index,
            window_start=start,
            source_frame_indices=source_frame_indices.clone(),
            video=torch.full((2,), start),
            action=torch.full((3, 64), start),
            action_raw=torch.full((3, 57), start),
            ar_boundary_states=torch.tensor([start + int(self.corrupt)]),
            ar_layout_version="joint_chunk_cond_v1",
            ar_state_normalizer_sha256="a" * 64,
            hand_visibility=torch.ones(3, 2),
            text_token_ids=torch.tensor([99]),
            sequence_plan="new random CFG",
            ar_num_tokens=999,
        )


def recovery_loader(dataset):
    from cosmos3_joint_video_hand_pose.src.dataloader_state import RecoverablePackingDataLoader

    loader = object.__new__(RecoverablePackingDataLoader)
    loader._map_dataset = lambda: dataset
    loader._split_single_sample = lambda raw: raw
    return loader


def test_recovery_reconstructs_exact_window_and_preserves_saved_cfg_rng():
    import random
    import numpy as np

    dataset = ExactDataset()
    loader = recovery_loader(dataset)
    raw = dataset.get_item_at_window(7, window_start=12, source_frame_indices=torch.tensor([12, 14]))
    raw["text_token_ids"] = torch.tensor([3])
    raw["sequence_plan"] = "saved CFG"
    raw["ar_num_tokens"] = 333
    metadata = loader._checkpoint_metadata(raw)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    restored = loader._rebuild_buffer([metadata])[0]
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.equal(restored["video"], raw["video"])
    assert torch.equal(restored["action"], raw["action"])
    assert torch.equal(restored["ar_boundary_states"], raw["ar_boundary_states"])
    assert restored["sequence_plan"] == "saved CFG"
    assert restored["ar_num_tokens"] == 333
    assert restored["text_token_ids"].tolist() == [3]
    dataset.corrupt = True
    with pytest.raises(ValueError, match="ar_boundary_states"):
        loader._rebuild_buffer([metadata])


def test_recovery_rejects_missing_exact_interface_and_keeps_unconsumed_pending():
    loader = recovery_loader(ExactDataset())
    raw = loader._map_dataset().get_item_at_window(1, window_start=12, source_frame_indices=torch.tensor([12, 14]))
    metadata = loader._checkpoint_metadata(raw)
    loader._map_dataset = lambda: [raw]
    with pytest.raises(ValueError, match="get_item_at_window"):
        loader._rebuild_buffer([metadata])
    loader._child_iterators_initialized = False
    loader._restored_buffer_metadata = [metadata]
    loader.global_id = 9
    loader.dataloader_list = [SimpleNamespace(state_dict=lambda: {"cursor": 8})]
    saved = loader.state_dict()
    assert saved["version"] == 2
    assert len(saved["buffer"]) == 1
    assert saved["buffer"][0] is not metadata


def test_real_omnimot_flat_checkpoint_preserves_binding(tmp_path):
    from cosmos_framework.checkpoint.dcp import ModelWrapper
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(ema=SimpleNamespace(enabled=False), exclude_reasoner_weights_from_checkpoint=False)
    model.net = torch.nn.Linear(2, 1)
    model.net.ar_training_contract = make_contract(tmp_path)
    saved = ModelWrapper(model).state_dict()
    assert "net.ar_training_contract._extra_state" in saved
    ModelWrapper(model).load_state_dict(saved)
    model.net.ar_training_contract = make_contract(tmp_path, value=2)
    with pytest.raises(ValueError, match="mismatch"):
        ModelWrapper(model).load_state_dict(saved)


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


# Reuse the data owner's synthetic tracking fixture, exercising the actual raw
# dataset, exact-window wrapper AND the production collate/split reconstruction.
from test_ar_v02_data import inputs as exact_inputs


def test_recovery_real_dataset_wrapper_and_collated_metadata(exact_inputs, monkeypatch):
    from test_ar_v02_data import dataset
    from cosmos3_joint_video_hand_pose.src import ar_dataset as module
    from cosmos3_joint_video_hand_pose.src.dataloader_state import RecoverablePackingDataLoader

    args, _ = exact_inputs
    ds = dataset(args, random_window=True)

    def transform(sample, resolution=None):
        # Native ActionTransformPipeline renders the raw caption dict to text.
        sample["ai_caption"] = json.dumps(sample["ai_caption"])
        sample["action_raw"] = sample["action"].clone()
        sample["action"] = torch.nn.functional.pad(sample["action"], (0, 7))
        sample["raw_action_dim"] = 57
        from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessingRecord

        sample["action_processing_record"] = ActionProcessingRecord(
            raw_action_dim=57, action_normalizer=None, action_valid_mask=torch.ones(57, dtype=torch.bool)
        )
        sample["text_token_ids"] = torch.arange(11)
        return sample

    wrapper = module.EgoVerseARCosmosDataset(ds, module.ARV02BudgetTransform(transform, 70_000))
    loader = object.__new__(RecoverablePackingDataLoader)
    loader._map_dataset = lambda: wrapper
    saved = loader._split_single_sample(wrapper.get_item_at_window(0, window_start=9))
    metadata = loader._checkpoint_metadata(saved)
    # Any random lookup would choose a different valid source window.
    monkeypatch.setattr(ds, "window_start", lambda row: 4)
    restored = loader._rebuild_buffer([metadata])[0]
    for key in (
        "video",
        "action",
        "action_raw",
        "ar_boundary_states",
        "ar_source_poses",
        "ar_hand_latents",
        "source_frame_indices",
        "future_action_source_frame_indices",
    ):
        loader._assert_rebuilt_metadata(restored[key], saved[key], key)


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


@pytest.mark.parametrize("count", [0, 2, 3, True, 1.5])
def test_callback_rejects_accumulation_before_checkpoint_access(tmp_path, count):
    callback = make_callback(tmp_path)
    callback.config.trainer.grad_accum_iter = count
    with pytest.raises(ValueError, match="grad_accum_iter=1"):
        callback.on_load_checkpoint_start(CallbackModel())


def rank_state_checkpoint(tmp_path, *, omitted_rng=None, omitted_loader=None, world_size=8):
    import pickle
    import torch.distributed.checkpoint as dcp
    from cosmos3_joint_video_hand_pose.src.ar_v02_contract import _RNG_STATE_FIELDS

    state = {"iteration": 50}
    for rank in range(world_size):
        state[f"rng_state_{rank}"] = {
            field: torch.ones(1, dtype=torch.uint8)
            for field in _RNG_STATE_FIELDS if (rank, field) != omitted_rng
        }
    dcp.save(state, checkpoint_id=str(tmp_path / "trainer"), no_dist=True)
    (tmp_path / "dataloader").mkdir(exist_ok=True)
    for rank in range(world_size):
        if rank != omitted_loader:
            (tmp_path / "dataloader" / f"rank_{rank}.pkl").write_bytes(
                pickle.dumps(dict(version=2, global_id=50, inner={}, buffer=[]))
            )
    source = SimpleNamespace(path=str(tmp_path), backend_key=None, warm_start=False)
    checkpointer = SimpleNamespace(get_storage_reader=lambda path, source: dcp.FileSystemReader(path))
    return checkpointer, source


def test_eight_rank_resume_preflight(tmp_path):
    from cosmos3_joint_video_hand_pose.src.ar_v02_contract import assert_resume_rank_state

    checkpointer, source = rank_state_checkpoint(tmp_path)
    assert_resume_rank_state(checkpointer, source, world_size=8)
    with pytest.raises(ValueError, match="expected world_size=4"):
        assert_resume_rank_state(checkpointer, source, world_size=4)


@pytest.mark.parametrize("rank", range(8))
def test_resume_rejects_each_missing_rank_loader(tmp_path, rank):
    from cosmos3_joint_video_hand_pose.src.ar_v02_contract import assert_resume_rank_state

    checkpointer, source = rank_state_checkpoint(tmp_path, omitted_loader=rank)
    with pytest.raises(ValueError, match=f"missing dataloader ranks=\\[{rank}\\]"):
        assert_resume_rank_state(checkpointer, source, world_size=8)


@pytest.mark.parametrize("field", ["torch", "torch_cuda", "numpy_packed_len", "numpy_packed_bytes", "random_packed_len", "random_packed_bytes"])
def test_resume_rejects_incomplete_rng(tmp_path, field):
    from cosmos3_joint_video_hand_pose.src.ar_v02_contract import assert_resume_rank_state

    checkpointer, source = rank_state_checkpoint(tmp_path, omitted_rng=(7, field))
    with pytest.raises(ValueError, match=f"rng_state_7.{field}"):
        assert_resume_rank_state(checkpointer, source, world_size=8)


def test_callback_auto_resume_requires_full_training_state(tmp_path):
    callback = make_callback(tmp_path)
    bind_source(callback, tmp_path / "run")
    keys, source = callback.trainer.checkpointer.keys_to_resume_during_load()
    source.warm_start = False
    with pytest.raises(ValueError, match="model/optim/scheduler/trainer/dataloader"):
        callback.on_load_checkpoint_start(CallbackModel())


def test_callback_runs_rank_preflight_before_binding_model(tmp_path, monkeypatch):
    callback = make_callback(tmp_path)
    bind_source(callback, tmp_path / "run")
    keys, source = callback.trainer.checkpointer.keys_to_resume_during_load()
    keys.update({"optim", "scheduler", "trainer", "dataloader"})
    callback.trainer.checkpointer.keys_to_resume_during_load = lambda: (keys, source)
    from cosmos3_joint_video_hand_pose.src import ar_v02_contract as module
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 8)
    def reject(checkpointer, loaded_source, *, world_size):
        assert world_size == 8 and loaded_source is source
        raise ValueError("rank preflight reached")
    monkeypatch.setattr(module, "assert_resume_rank_state", reject)
    model = CallbackModel()
    with pytest.raises(ValueError, match="rank preflight reached"):
        callback.on_load_checkpoint_start(model)
    assert not hasattr(model.net, "ar_training_contract")


def test_callback_complete_training_resume_cpu(tmp_path):
    import torch.distributed.checkpoint as dcp

    callback = make_callback(tmp_path)
    model = CallbackModel()
    callback._attach(model)
    run = tmp_path / "complete_resume"
    checkpointer, source = rank_state_checkpoint(run, world_size=1)
    dcp.save(model.state_dict(), checkpoint_id=str(run / "model"), no_dist=True)
    checkpointer.keys_to_resume_during_load = lambda: (
        {"model", "optim", "scheduler", "trainer", "dataloader"}, source
    )
    callback.trainer = SimpleNamespace(checkpointer=checkpointer)
    restored = CallbackModel()
    callback.on_load_checkpoint_start(restored)
    callback.on_load_checkpoint_end(restored, iteration=50, checkpoint_path=str(run))
    callback.on_train_start(restored, iteration=50)
    assert callback._ready and restored.accumulation == 1
