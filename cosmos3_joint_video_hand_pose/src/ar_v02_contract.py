"""CPU-testable v0.2 loss and checkpoint contracts.

Cosmos divides backward loss by grad_accum_iter; FSDP/DDP averages gradients.
Do NOT apply OmniMoT sample_level_loss_scale on top of this reduction.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist


LEGACY_ACTION_REPRESENTATION = "legacy_local_delta_absolute_hand_v1"
FIXED_CAMERA_ACTION_REPRESENTATION = "fixed_camera_wrist_local_delta_latent_v1"


def _file_sha256(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required v0.2 artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GlobalSampleMeanWindow:
    """Independent video/action means over DP ranks AND accumulation microsteps.

    Plan local_counts=[microsteps,2] (active video/action samples) BEFORE forward.
    All ranks must consume the same number of microsteps, including empty ranks.
    group must match gradient averaging; CP1 uses WORLD. Returned loss compensates
    the trainer's /microsteps, which must happen exactly once. Detached contribution
    metrics should be SUMMED across ranks/steps, not averaged again.
    """

    def __init__(self, local_counts, *, device=None, group=None):
        counts = torch.as_tensor(local_counts, device=device)
        if counts.ndim != 2 or counts.shape[1] != 2 or counts.shape[0] < 1:
            raise ValueError("local_counts must be [microsteps,2] for video/action")
        if not torch.isfinite(counts).all() or (counts < 0).any() or (counts != counts.round()).any():
            raise ValueError("sample counts must be finite non-negative integers")
        self.local_counts = counts.detach().to(dtype=torch.int64).clone()
        self.microsteps = counts.shape[0]
        self.position = 0
        self.group = group
        self.world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        totals = self.local_counts.sum(0)
        summary = torch.cat((totals, totals.new_tensor([self.microsteps, self.microsteps**2])))
        if self.world_size > 1:
            dist.all_reduce(summary, group=group)
        if self.world_size * summary[3] != summary[2] ** 2:
            raise ValueError("all ranks must plan the same number of microsteps")
        self.global_counts = summary[:2]

    @property
    def complete(self):
        return self.position == self.microsteps

    def reduce(self, video_losses, video_active, action_losses, action_active, *, action_weight=1.0):
        if self.complete:
            raise RuntimeError("loss window already consumed")
        terms, actual = [], []
        for losses, active in ((video_losses, video_active), (action_losses, action_active)):
            if losses.ndim != 1 or active.shape != losses.shape:
                raise ValueError("per-sample losses and active masks must be matching vectors")
            if losses.device != self.global_counts.device:
                raise ValueError("loss window counts must reside on the loss device")
            if not torch.isfinite(losses).all() or not ((active == 0) | (active == 1)).all():
                raise ValueError("non-finite loss or non-binary active mask")
            actual.append(active.sum().to(self.local_counts))
            terms.append(torch.where(active.to(device=losses.device, dtype=torch.bool), losses, 0).sum())
        if not torch.equal(torch.stack(actual), self.local_counts[self.position]):
            raise ValueError("observed active samples differ from the planned loss window")
        contributions = [value / self.global_counts[i].clamp_min(1) for i, value in enumerate(terms)]
        self.position += 1
        total = contributions[0] + action_weight * contributions[1]
        metrics = {
            "video_contribution": contributions[0].detach(),
            "action_contribution": contributions[1].detach(),
            "global_video_samples": self.global_counts[0].detach().clone(),
            "global_action_samples": self.global_counts[1].detach().clone(),
        }
        return total * (self.world_size * self.microsteps), metrics


def assert_optimizer_covers_trainable(net, optimizer, *, required_names=()):
    """Capture required_names BEFORE init: keys_to_select can freeze new embeddings."""
    parameters = dict(net.named_parameters())

    def groups(value):
        # Cosmos uses one optimizer per mesh; retain cross-optimizer duplicate
        # detection rather than checking each child independently.
        children = getattr(value, "optimizers", None)
        if children is not None:
            from collections.abc import Mapping

            children = children.values() if isinstance(children, Mapping) else children
            for child in children:
                yield from groups(child)
        elif hasattr(value, "param_groups"):
            yield from value.param_groups
        else:
            raise TypeError(f"unsupported optimizer wrapper: {type(value).__name__}")

    counts = Counter(id(p) for group in groups(optimizer) for p in group["params"])
    missing = [name for name, p in parameters.items() if p.requires_grad and counts[id(p)] == 0]
    duplicate = [name for name, p in parameters.items() if counts[id(p)] > 1]
    required_bad = [
        name
        for name in required_names
        if name not in parameters or not parameters[name].requires_grad or counts[id(parameters[name])] != 1
    ]
    if missing or duplicate or required_bad:
        raise ValueError(f"optimizer contract: missing={missing}, duplicate={duplicate}, required={required_bad}")


class ARTrainingContract(torch.nn.Module):
    """Attach to model.net before FSDP/DCP; OmniMoT ignores top-level extra_state.

    Includes actual normalizer JSON. DCP resume MUST use strict_resume=True or
    preflight metadata keys: partial load can retain initialized missing entries.
    Explicitly skip this subtree only for official warm-start, never on resume.
    Opt-in until the main model supplies the new frozen chunk-camera artifacts.
    """

    def __init__(
        self,
        *,
        state_normalizer,
        action_normalizer,
        manifest_sha256,
        layout="joint_chunk_cond_v1",
        representation=LEGACY_ACTION_REPRESENTATION,
        right_hand_codec=None,
        left_hand_codec=None,
    ):
        super().__init__()
        if layout != "joint_chunk_cond_v1":
            raise ValueError("v0.2 training requires layout joint_chunk_cond_v1")
        if len(manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in manifest_sha256):
            raise ValueError("manifest_sha256 must be a SHA256 digest")
        if representation not in (LEGACY_ACTION_REPRESENTATION, FIXED_CAMERA_ACTION_REPRESENTATION):
            raise ValueError(f"unsupported action representation: {representation}")
        artifacts = {}
        for name, path in (("state_normalizer", state_normalizer), ("action_normalizer", action_normalizer)):
            raw = Path(path).read_bytes()
            payload = json.loads(raw)
            if representation == LEGACY_ACTION_REPRESENTATION and name == "state_normalizer":
                if payload.get("frozen") is not True or payload.get("split") != "train":
                    raise ValueError("state normalizer must be frozen and train-only")
                if "f0" in str(payload.get("schema", "")).lower():
                    raise ValueError("old F0 state normalizer cannot initialize chunk-camera training")
                if payload.get("manifest_sha256") != manifest_sha256:
                    raise ValueError("state normalizer/manifest binding mismatch")
            artifacts[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "payload": payload}
        if representation == FIXED_CAMERA_ACTION_REPRESENTATION:
            from .action_fixed_normalization import FixedCameraNormalizer
            from .codec_fixed_camera import FrozenFixedCameraHandCodec

            state = FixedCameraNormalizer(state_normalizer, kind="state")
            future = FixedCameraNormalizer(
                action_normalizer, kind="future", codec_sha256=state.codec_sha256
            )
            if state.profile["manifest_sha256"] != manifest_sha256:
                raise ValueError("state normalizer/manifest binding mismatch")
            if future.profile["manifest_sha256"] != manifest_sha256:
                raise ValueError("future normalizer/manifest binding mismatch")
            codec_paths = (right_hand_codec, left_hand_codec)
            if any(path is None for path in codec_paths):
                raise ValueError("fixed-camera contract requires explicit right/left hand codecs")
            codec_hashes = tuple(_file_sha256(path) for path in codec_paths)
            if codec_hashes != state.codec_sha256:
                raise ValueError("normalizers and right/left hand codec SHA256 identities differ")
            for side, path, digest in zip(("right", "left"), codec_paths, codec_hashes, strict=True):
                codec = FrozenFixedCameraHandCodec(path, expected_sha256=digest)
                if codec.representation != FIXED_CAMERA_ACTION_REPRESENTATION:
                    raise ValueError("legacy hand codec cannot initialize fixed-camera training")
                if codec.checkpoint_sha256 != digest or codec.metadata.get("side") != side:
                    raise ValueError(f"fixed-camera {side} hand codec identity/side mismatch")
                artifacts[f"{side}_hand_codec"] = {
                    "sha256": digest,
                    "representation": codec.representation,
                    "input_frame": codec.input_frame,
                }
            self._expected = dict(
                schema="ar_v02_training_contract_v2",
                representation=representation,
                layout=layout,
                manifest_sha256=manifest_sha256,
                codec_sha256=list(codec_hashes),
                normalizer_profile_sha256={
                    "state": state.profile["profile_sha256"],
                    "future": future.profile["profile_sha256"],
                },
                artifacts=artifacts,
            )
        else:
            # Preserve the v1 payload byte-for-byte so historical legacy v0.2
            # checkpoints can still be inspected/resumed explicitly.
            self._expected = dict(
                schema="ar_v02_training_contract_v1",
                layout=layout,
                manifest_sha256=manifest_sha256,
                artifacts=artifacts,
            )

    def get_extra_state(self):
        return copy.deepcopy(self._expected)

    def set_extra_state(self, state):
        if state != self._expected:
            raise ValueError("checkpoint layout/normalizer/manifest training contract mismatch")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if prefix + "_extra_state" not in state_dict:
            raise ValueError("checkpoint missing v0.2 training contract (explicit warm-start skip required)")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


def assert_finite_gradients(parameters, *, group=None):
    """Check raw gradients before clipping; synchronize failure across all ranks."""
    parameters = list(parameters)
    if not parameters:
        raise ValueError("gradient check needs model parameters")
    flags = torch.zeros(2, dtype=torch.int64, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        if hasattr(gradient, "to_local"):
            gradient = gradient.to_local()
        flags[0] += (~torch.isfinite(gradient)).any().to(flags)
        flags[1] += 1
    if dist.is_initialized():
        dist.all_reduce(flags, group=group)
    if flags[0].item():
        raise FloatingPointError("non-finite raw gradients before clipping/optimizer update")
    if not flags[1].item():
        raise RuntimeError("backward produced no gradients on any rank")
    return int(flags[1].item())


# Cosmos' get_rand_state_dict format; inspect metadata without touching CUDA/RNG.
_RNG_STATE_FIELDS = (
    "torch", "torch_cuda", "numpy_packed_len", "numpy_packed_bytes",
    "random_packed_len", "random_packed_bytes",
)


def assert_resume_rank_state(checkpointer, source, *, world_size):
    """Fail before model loading if any current rank cannot resume exactly.

    Use Cosmos storage readers/backends; do not change its DCP loader. Check all
    ranks on every process so rank-local missing files cannot silently restart
    a stream. Actual payload deserialization remains with the official loader.
    """
    from cosmos_framework.utils.easy_io import easy_io

    root = str(source.path).rstrip("/")
    reader = checkpointer.get_storage_reader(root + "/trainer", source)
    metadata = reader.read_metadata().state_dict_metadata
    required = ["iteration"] + [
        f"rng_state_{rank}.{field}"
        for rank in range(world_size) for field in _RNG_STATE_FIELDS
    ]
    missing_rng = [key for key in required if key not in metadata]
    expected_ranks = {f"rng_state_{rank}" for rank in range(world_size)}
    saved_ranks = {key.split(".", 1)[0] for key in metadata if key.startswith("rng_state_")}
    missing_loaders = [
        rank for rank in range(world_size)
        if not easy_io.exists(root + f"/dataloader/rank_{rank}.pkl", backend_key=source.backend_key)
    ]
    if missing_rng or missing_loaders or saved_ranks != expected_ranks:
        raise ValueError(
            f"strict resume rank state incomplete: missing RNG={missing_rng}, "
            f"missing dataloader ranks={missing_loaders}, "
            f"saved RNG ranks={sorted(saved_ranks)}, expected world_size={world_size}"
        )


def _require_single_microstep(value):
    if type(value) is not int or value != 1:
        raise ValueError("v0.2 trainer requires grad_accum_iter=1; accumulation >1 is not supported")


# Kept here so the training config has one explicit contract integration hook.
from cosmos_framework.utils.callback import Callback


class ARTrainingContractCallback(Callback):
    """Bind artifacts before resume; allow absent binding ONLY for official init.

    Config hook (LazyCall): trainer.callbacks.ar_v02_contract =
        L(ARTrainingContractCallback)(state_normalizer=..., action_normalizer=...,
            valid_windows_manifest=..., official_checkpoint=...)
    No contract key skip is needed: the parameter-free module is attached AFTER
    official initialization, but BEFORE loading any v0.2 checkpoint. A small DCP
    preflight validates metadata without modifying model tensors.
    """

    def __init__(
        self,
        *,
        state_normalizer,
        action_normalizer,
        valid_windows_manifest,
        official_checkpoint,
        representation=LEGACY_ACTION_REPRESENTATION,
        right_hand_codec=None,
        left_hand_codec=None,
        check_raw_gradients=True,
    ):
        super().__init__()
        self.state_normalizer = str(state_normalizer)
        self.action_normalizer = str(action_normalizer)
        self.valid_windows_manifest = str(valid_windows_manifest)
        self.official_checkpoint = Path(official_checkpoint).resolve()
        self.representation = representation
        self.right_hand_codec = None if right_hand_codec is None else str(right_hand_codec)
        self.left_hand_codec = None if left_hand_codec is None else str(left_hand_codec)
        self.check_raw_gradients = check_raw_gradients
        self.contract = self._make_contract()
        self._ready = False
        self._source_path = None
        self._official_init = False

    def _make_contract(self):
        digest = hashlib.sha256(Path(self.valid_windows_manifest).read_bytes()).hexdigest()
        if self.representation == LEGACY_ACTION_REPRESENTATION:
            from .ar_chunk_state import ChunkCameraStateNormalizer

            # Use the legacy data owner's strict schema/profile/18D validation.
            ChunkCameraStateNormalizer(self.state_normalizer, expected_manifest_sha256=digest)
        return ARTrainingContract(
            state_normalizer=self.state_normalizer,
            action_normalizer=self.action_normalizer,
            manifest_sha256=digest,
            representation=self.representation,
            right_hand_codec=self.right_hand_codec,
            left_hand_codec=self.left_hand_codec,
        )

    def _attach(self, model):
        for name in ("net", "net_ema"):
            net = getattr(model, name, None)
            if net is None:
                continue
            current = getattr(net, "ar_training_contract", None)
            if current is None:
                net.add_module("ar_training_contract", copy.deepcopy(self.contract))
            elif not isinstance(current, ARTrainingContract):
                raise TypeError(f"{name}.ar_training_contract already has an incompatible owner")
            else:
                current.set_extra_state(self.contract.get_extra_state())

    def on_load_checkpoint_start(self, model):
        import torch.distributed.checkpoint as dcp

        self._ready = False
        if not getattr(model, "whole_action_loss", False):
            raise ValueError("ARTrainingContractCallback requires the v0.2 whole-loss model")
        model_representation = getattr(model, "action_representation", LEGACY_ACTION_REPRESENTATION)
        if model_representation != self.representation:
            raise ValueError("model and checkpoint contract action representations differ")
        _require_single_microstep(self.config.trainer.grad_accum_iter)
        model.configure_ar_loss_accumulation(self.config.trainer.grad_accum_iter)
        keys, source = self.trainer.checkpointer.keys_to_resume_during_load()
        if source is None or "model" not in keys:
            raise ValueError("v0.2 requires official initialization or a bound model checkpoint")
        self._source_path = str(source.path)
        self._official_init = (
            source.warm_start
            and not source.uses_object_store
            and Path(source.path).resolve() == self.official_checkpoint
        )
        if self._official_init:
            if set(keys) != {"model"}:
                raise ValueError("official initialization must not restore optimizer/trainer state")
            if hasattr(model.net, "ar_training_contract"):
                raise ValueError("attach the contract after official initialization, not before")
            return

        if not self.config.checkpoint.strict_resume:
            raise ValueError("v0.2 resume requires checkpoint.strict_resume=True")
        if not source.warm_start or set(keys) != {"model"}:
            required_components = {"model", "optim", "scheduler", "trainer", "dataloader"}
            if not required_components.issubset(keys):
                raise ValueError("strict resume requires model/optim/scheduler/trainer/dataloader")
            assert_resume_rank_state(
                self.trainer.checkpointer, source,
                world_size=dist.get_world_size() if dist.is_initialized() else 1,
            )
        self._attach(model)
        reader = self.trainer.checkpointer.get_storage_reader(str(Path(source.path) / "model"), source)
        metadata = reader.read_metadata().state_dict_metadata
        required = [
            f"net.{name}"
            for name, _ in model.net.named_parameters()
            if any(tag in name for tag in ("state_embed", "condition_embed", "observation_embed"))
        ]
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"checkpoint missing v0.2 type embeddings: {missing}")
        # External v0.2 warm starts must not silently reinitialize learned types.
        skips = self.config.checkpoint.keys_to_skip_loading if source.warm_start else []
        protected = required + ["net.ar_training_contract._extra_state"]
        if any(pattern in key for pattern in skips for key in protected):
            raise ValueError("v0.2 checkpoint loading may not skip contract/type embeddings")
        binding = {"net.ar_training_contract._extra_state": self.contract.get_extra_state()}
        # Default strict planner rejects EVERY missing binding leaf, even when a
        # partial main loader would otherwise retain the initialized value.
        dcp.load(binding, storage_reader=reader, no_dist=True)
        self.contract.set_extra_state(binding["net.ar_training_contract._extra_state"])

    def on_load_checkpoint_end(self, model, iteration=0, checkpoint_path=None):
        if str(checkpoint_path) != self._source_path:
            raise ValueError("checkpoint source changed after contract preflight")
        if self._official_init and iteration != 0:
            raise ValueError("official initialization unexpectedly restored a training iteration")
        self._attach(model)
        self._ready = True

    def on_train_start(self, model, iteration=0):
        if not self._ready:
            raise RuntimeError("checkpoint contract was not validated before training")
        _require_single_microstep(self.config.trainer.grad_accum_iter)
        model.configure_ar_loss_accumulation(self.config.trainer.grad_accum_iter)

    def on_training_step_batch_start(self, model, data_batch, iteration=0):
        if not self._ready:
            raise RuntimeError("checkpoint contract not ready")
        expected = self.contract.get_extra_state()
        fields = {
            "ar_layout_version": expected["layout"],
            "ar_state_normalizer_sha256": expected["artifacts"]["state_normalizer"]["sha256"],
            "ar_valid_windows_sha256": expected["manifest_sha256"],
        }
        if self.representation == FIXED_CAMERA_ACTION_REPRESENTATION:
            fields.update(
                ar_action_representation=self.representation,
                ar_future_normalizer_sha256=expected["artifacts"]["action_normalizer"]["sha256"],
                ar_right_hand_codec_sha256=expected["artifacts"]["right_hand_codec"]["sha256"],
                ar_left_hand_codec_sha256=expected["artifacts"]["left_hand_codec"]["sha256"],
            )

        def flatten(value):
            if isinstance(value, (list, tuple)):
                return [x for item in value for x in flatten(item)]
            return [value]

        for key, value in fields.items():
            observed = flatten(data_batch.get(key, []))
            if not observed or any(item != value for item in observed):
                raise ValueError(f"batch/checkpoint contract mismatch: {key}")

    def on_after_backward(self, model, iteration=0):
        if self.check_raw_gradients:
            assert_finite_gradients(model.net.parameters())

    def on_save_checkpoint_start(self, model, iteration=0):
        if not self._ready:
            raise RuntimeError("cannot save an unvalidated v0.2 checkpoint")
        self.contract.set_extra_state(self._make_contract().get_extra_state())
        self._attach(model)
        window = getattr(model, "_ar_loss_window", None)
        if window is not None and not window.complete:
            raise RuntimeError("cannot checkpoint a partially accumulated loss window")

    def on_save_checkpoint(self, model, state_dict):
        saved = state_dict["model"].get("net.ar_training_contract._extra_state")
        self.contract.set_extra_state(saved)
