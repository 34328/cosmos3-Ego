"""Checkpoint binding, exact data recovery, and native training resume."""

import json
from types import SimpleNamespace
import pytest
import torch
from cosmos3_joint_video_hand_pose.src.ar_v02_contract import ARTrainingContract
from test_ar_v02_data import inputs as exact_inputs


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
