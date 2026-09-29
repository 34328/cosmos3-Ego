from __future__ import annotations

import json
import os
from typing import Any

import torch

from .loss import visibility_weighted_action_flow_loss, whole_action_flow_loss, whole_video_flow_loss
from .ar_v02_contract import GlobalSampleMeanWindow, assert_optimizer_covers_trainable
from .action_representation import FIXED_CAMERA, LEGACY


try:
    from cosmos_framework.model.generator.mot.context_parallel_utils import broadcast_context_parallel_object
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
except ImportError as error:  # pragma: no cover - exercised only outside the Cosmos environment
    raise ImportError(
        "EgoVerseOmniMoTModel requires PYTHONPATH=<repository-root>:<repository-root>/packages/cosmos3"
    ) from error


def _visibility_from_batch(data_batch: dict[str, Any]) -> list[torch.Tensor]:
    raw = data_batch.get("hand_visibility")
    if raw is None:
        raise KeyError("hand_visibility is required in EgoVerse batch metadata")
    items = raw if isinstance(raw, list) else [raw]
    result = []
    for item in items:
        tensor = torch.as_tensor(item)
        if not ((tensor == 0) | (tensor == 1)).all():
            raise ValueError("hand_visibility must be binary, not missing/NaN labels")
        tensor = tensor.bool()
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError(f"hand_visibility must be [T,2], got {tuple(tensor.shape)}")
        result.append(tensor.cpu())
    return result


class EgoVerseLossMixin:
    """57D representation-specific action loss and logging for EgoVerse models.

    Mix in before a Cosmos ``OmniMoTModel`` subclass; the generator architecture is unchanged.
    """

    def __init__(
        self,
        config,
        lambda_out_of_fov: float = 0.0,
        subblock_equal_weight: bool = False,
        whole_action_loss: bool = False,
    ):
        super().__init__(config)
        if not 0 <= lambda_out_of_fov <= 1:
            raise ValueError("lambda_out_of_fov must be in [0,1]")
        self.lambda_out_of_fov = float(lambda_out_of_fov)
        self.subblock_equal_weight = bool(subblock_equal_weight)
        self.whole_action_loss = bool(whole_action_loss)
        if self.whole_action_loss and (self.lambda_out_of_fov != 0 or self.subblock_equal_weight):
            raise ValueError("whole_action_loss requires binary visibility and no sub-block weighting")
        self._current_hand_visibility: list[torch.Tensor] | None = None
        self._cp_local_hand_visibility: list[torch.Tensor] | None = None
        # Sub-block action losses written by the current _compute_losses call only.
        self._last_visibility_loss_metrics: dict[str, torch.Tensor] = {}
        self._ar_loss_window = None
        self._ar_gradient_accumulation = 1
        self._ar_backward_loss = None

    def configure_ar_loss_accumulation(self, microsteps: int):
        """Trainer/callback must pass its actual grad_accum_iter before training."""
        if not isinstance(microsteps, int) or microsteps < 1:
            raise ValueError("gradient accumulation must be a positive integer")
        window = getattr(self, "_ar_loss_window", None)
        if window is not None and not window.complete:
            raise RuntimeError("cannot reconfigure an unfinished loss window")
        self._ar_gradient_accumulation = microsteps

    def begin_ar_loss_window(self, local_counts, *, device=None):
        """Main trainer preplans [microsteps,2] effective modality sample counts.

        Call once on ALL ranks before a multi-microbatch optimizer update.
        With grad_accum_iter=1, the loss path obtains counts automatically.
        """
        previous = getattr(self, "_ar_loss_window", None)
        if previous is not None and not previous.complete:
            raise RuntimeError("previous loss window is unfinished")
        cp = getattr(self, "parallel_dims", None)
        if cp is not None and cp.cp_enabled:
            raise ValueError("v0.2 global sample loss currently requires CP1")
        group, size = self._loss_averaging_group()
        window = GlobalSampleMeanWindow(local_counts, device=device, group=group)
        if window.world_size != size:
            raise ValueError("loss count group must match the gradient averaging group")
        if window.microsteps != getattr(self, "_ar_gradient_accumulation", 1):
            raise ValueError("planned microsteps must match configured trainer grad_accum_iter")
        self._ar_loss_window = window

    def training_step(self, data_batch, iteration):
        self._ar_backward_loss = None
        try:
            output, loss = super().training_step(data_batch, iteration)
            if self._ar_backward_loss is not None:
                output["_backward_loss"] = self._ar_backward_loss
            return output, loss
        finally:
            self._ar_backward_loss = None

    def on_before_optimizer_step(self, optimizer, scheduler, iteration):
        window = getattr(self, "_ar_loss_window", None)
        if getattr(self, "whole_action_loss", False) and window is not None and not window.complete:
            raise RuntimeError("optimizer step before the planned loss window completed")
        return super().on_before_optimizer_step(optimizer, scheduler, iteration)

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        # Save names before Cosmos keys_to_select can silently freeze them.
        required = [
            name
            for name, _ in self.net.named_parameters()
            if any(tag in name for tag in ("state_embed", "condition_embed", "observation_embed"))
        ]
        optimizer, scheduler = super().init_optimizer_scheduler(optimizer_config, scheduler_config)
        if getattr(self, "whole_action_loss", False):
            assert_optimizer_covers_trainable(self.net, optimizer, required_names=required)
            from .formal_monitor import optimizer_lr_receipt
            self._optimizer_lr_receipt = optimizer_lr_receipt(self.net, optimizer, optimizer_config)
        return optimizer, scheduler

    def _compute_whole_losses(self, out_net, packed, noised, timesteps, is_image_batch):
        """v0.2 owns both reductions; bypass the parent's whole-pack multiplier."""
        cfg = self.config
        rf = cfg.rectified_flow_training_config
        if (
            is_image_batch
            or not cfg.vision_gen
            or not cfg.action_gen
            or getattr(cfg, "sound_gen", False)
            or getattr(cfg, "lidar_gen", False)
        ):
            raise ValueError("v0.2 whole loss requires paired video/action only")
        if rf.loss_scale != 1.0 or rf.action_loss_weight != 1.0:
            raise ValueError("v0.2 objective is L_video + L_action")
        for kind in ("und", "gen"):
            if out_net.get(f"lbl_metadata_{kind}") is not None and getattr(
                getattr(cfg, "lbl", None), f"coeff_{kind}", 0
            ):
                raise ValueError("v0.2 two-term objective does not include auxiliary load balancing")
        if packed.vision is None or packed.action is None:
            raise ValueError("v0.2 pack must retain both modalities, even when fully conditioned")
        n = len(out_net["preds_action"])
        if len(out_net["preds_vision"]) != n or len(packed.sample_lens) != n:
            raise ValueError("v0.2 loss needs one video/action item per logical sample")
        dims = packed.action.raw_action_dim
        if len(dims) != n or any(d is None or int(d) != 57 for d in dims):
            raise ValueError("v0.2 loss requires explicit raw_action_dim=57 for every sample")
        if self._current_hand_visibility is None:
            raise RuntimeError("action loss reached without synchronized hand visibility")

        def video_weight(index, frames, reference):
            ts = timesteps[index, :frames] if timesteps.ndim > 1 else timesteps[index]
            return self.rectified_flow_video.train_time_weight(ts, self.tensor_kwargs_fp32)

        _, video = whole_video_flow_loss(
            pred=out_net["preds_vision"],
            target=noised.vt_target_vision,
            condition_mask=packed.vision.condition_mask,
            time_weight=video_weight,
        )
        _, action = whole_action_flow_loss(
            pred=out_net["preds_action"],
            target=noised.vt_target_action,
            condition_mask=packed.action.condition_mask,
            visibility=self._current_hand_visibility,
            valid_mask=packed.action.action_valid_mask,
            mask_out_of_fov=getattr(self, "action_representation", LEGACY) != FIXED_CAMERA,
            collect_field_metrics=getattr(self, "action_representation", LEGACY) == FIXED_CAMERA,
        )
        window = getattr(self, "_ar_loss_window", None)
        if window is None or window.complete:
            if getattr(self, "_ar_gradient_accumulation", 1) != 1:
                raise RuntimeError("call begin_ar_loss_window with all microbatch counts before accumulation")
            counts = torch.stack((video["active_samples"].sum(), action["active_samples"].sum()))[None]
            self.begin_ar_loss_window(counts, device=counts.device)
            window = self._ar_loss_window
        backward, stats = window.reduce(
            video["per_sample_losses"], video["active_samples"], action["per_sample_losses"], action["active_samples"],
            action_weight=rf.action_loss_weight,
        )
        self._ar_backward_loss = backward
        self._last_visibility_loss_metrics = {
            name + "_loss": values.sum() * window.world_size * window.microsteps
            / window.global_counts[1].clamp_min(1)
            for name, values in action.get("field_per_sample_losses", {}).items()
        }
        # Rank-mean of these metrics, SUMMED over microsteps, is the update mean.
        v = stats["video_contribution"] * window.world_size
        a = stats["action_contribution"] * window.world_size
        logged_loss = backward / window.microsteps
        return logged_loss, {
            # Native WandBCallback must average these already globally scaled
            # rank contributions, not weight them by the local pack size again.
            "train_objective_numerator": logged_loss.detach(),
            "train_objective_denominator": torch.ones_like(logged_loss.detach()),
            "flow_matching_loss_vision": v,
            "flow_matching_loss_action": a,
            "flow_matching_loss_vision_per_instance": video["unweighted_per_sample_losses"].detach(),
            "egoverse_global_video_samples": stats["global_video_samples"],
            "egoverse_global_action_samples": stats["global_action_samples"],
        }

    def _get_training_inputs(self, data_batch: dict[str, torch.Tensor], iteration: int):
        cp_enabled = self.parallel_dims is not None and self.parallel_dims.cp_enabled
        owner_slot = self._cp_window_slot
        if not cp_enabled:
            self._current_hand_visibility = _visibility_from_batch(data_batch)
            return super()._get_training_inputs(data_batch, iteration)

        cp_size = self.parallel_dims.cp_mesh.size()
        if owner_slot == 0:
            self._cp_local_hand_visibility = _visibility_from_batch(data_batch)
        result = super()._get_training_inputs(data_batch, iteration)
        self._current_hand_visibility = broadcast_context_parallel_object(
            self._cp_local_hand_visibility,
            self.parallel_dims,
            owner_rank=owner_slot,
        )
        if owner_slot == cp_size - 1:
            self._cp_local_hand_visibility = None
        return result

    def _compute_flow_matching_loss(
        self,
        pred,
        target,
        condition_mask,
        timesteps,
        has_valid_tokens,
        rectified_flow,
        raw_action_dim=None,
        action_valid_mask=None,
        normalize_by_active=False,
        exclude_fully_conditioned_items=False,
        action_slot_stats=None,
    ):
        if raw_action_dim is None:
            return super()._compute_flow_matching_loss(
                pred=pred,
                target=target,
                condition_mask=condition_mask,
                timesteps=timesteps,
                has_valid_tokens=has_valid_tokens,
                rectified_flow=rectified_flow,
                raw_action_dim=raw_action_dim,
                action_valid_mask=action_valid_mask,
                normalize_by_active=normalize_by_active,
                exclude_fully_conditioned_items=exclude_fully_conditioned_items,
                action_slot_stats=action_slot_stats,
            )
        # EgoVerse samples carry no per-channel action_valid_mask; its 57D
        # visibility-weighted objective owns channel selection itself.
        if (
            not getattr(self, "whole_action_loss", False)
            and action_valid_mask is not None
            and any(mask is not None for mask in action_valid_mask)
        ):
            raise NotImplementedError("EgoVerse action loss does not support action_valid_mask")
        if exclude_fully_conditioned_items and not getattr(self, "whole_action_loss", False):
            raise NotImplementedError("EgoVerse action loss does not support exclude_fully_conditioned_items")
        # action_slot_stats only collects unified-schema (raw_action_dim == 59)
        # slot losses; for the 57D EgoVerse contract they stay zero, matching
        # the native path, so the collector is intentionally left untouched.
        del action_slot_stats
        if not has_valid_tokens:
            dummy = 0.0 * sum(item.sum() for item in pred)
            return dummy, dummy.unsqueeze(0)
        if self._current_hand_visibility is None:
            raise RuntimeError("action loss reached without synchronized hand visibility")
        for dim in raw_action_dim:
            if dim is not None and int(dim) != 57:
                raise ValueError(f"EgoVerse expects raw_action_dim=57, got {int(dim)}")

        def time_weight(sample_index: int, frames: int, reference: torch.Tensor) -> torch.Tensor:
            ts = timesteps[sample_index, :frames] if timesteps.dim() > 1 else timesteps[sample_index]
            return rectified_flow.train_time_weight(ts, self.tensor_kwargs_fp32).to(reference)

        # This 57D path is normalized by visibility_weighted_action_flow_loss
        # itself; ``normalize_by_active`` (rf_cfg.normalize_loss_by_active) does not apply here.
        if getattr(self, "whole_action_loss", False):
            loss, metrics = whole_action_flow_loss(
                pred=pred,
                target=target,
                condition_mask=condition_mask,
                visibility=self._current_hand_visibility,
                valid_mask=action_valid_mask,
                mask_out_of_fov=getattr(self, "action_representation", LEGACY) != FIXED_CAMERA,
            )
        else:
            loss, metrics = visibility_weighted_action_flow_loss(
                pred=pred,
                target=target,
                condition_mask=condition_mask,
                visibility=self._current_hand_visibility,
                time_weight=time_weight,
                lambda_out_of_fov=self.lambda_out_of_fov,
                subblock_equal_weight=self.subblock_equal_weight,
            )
        per_sample_losses = metrics["per_sample_losses"]
        self._last_visibility_loss_metrics = {
            name: value.detach() for name, value in metrics.items() if name != "per_sample_losses"
        }
        return loss, per_sample_losses

    def _compute_losses(
        self,
        out_net,
        data_batch_packed,
        gen_data_noised,
        timesteps,
        is_image_batch,
        timesteps_action=None,
        timesteps_sound=None,
        timesteps_lidar=None,
    ):
        """Expose raw and actually weighted components for distributed logging."""
        # Only report sub-block losses produced by this step's 57D action loss;
        # steps without it (no action, dummy branch) must not repeat stale values.
        self._last_visibility_loss_metrics = {}
        if getattr(self, "whole_action_loss", False):
            total_loss, losses = self._compute_whole_losses(
                out_net, data_batch_packed, gen_data_noised, timesteps, is_image_batch
            )
        else:
            total_loss, losses = super()._compute_losses(
                out_net=out_net,
                data_batch_packed=data_batch_packed,
                gen_data_noised=gen_data_noised,
                timesteps=timesteps,
                is_image_batch=is_image_batch,
                timesteps_action=timesteps_action,
                timesteps_sound=timesteps_sound,
                timesteps_lidar=timesteps_lidar,
            )
        rf_cfg = self.config.rectified_flow_training_config
        if getattr(self, "_record_pretrain_probe", False):
            # Capture the actual schedules without drawing RNG or retaining a graph.
            self._pretrain_noise_trace = {
                "video_timesteps": timesteps.detach().float().cpu().tolist(),
                "action_timesteps": None if timesteps_action is None else timesteps_action.detach().float().cpu().tolist(),
            }
        sample_scale = torch.ones((), device=total_loss.device, dtype=total_loss.dtype)
        if (
            not getattr(self, "whole_action_loss", False)
            and rf_cfg.sample_level_loss_averaging
            and self.config.vision_gen
        ):
            sample_scale = self._sample_level_loss_scale(
                is_image_batch=is_image_batch,
                num_samples=len(out_net["preds_vision"]),
                device=self.tensor_kwargs_fp32["device"],
            ).to(device=total_loss.device, dtype=total_loss.dtype)

        video_raw = losses["flow_matching_loss_vision"] * sample_scale
        action_raw = losses["flow_matching_loss_action"] * sample_scale
        video_weight = (
            rf_cfg.image_loss_scale if is_image_batch and rf_cfg.image_loss_scale is not None else rf_cfg.loss_scale
        )
        losses.update(
            egoverse_loss_video_raw=video_raw,
            egoverse_loss_action_raw=action_raw,
            egoverse_loss_video_weighted=video_raw * video_weight,
            egoverse_loss_action_weighted=action_raw * rf_cfg.action_loss_weight,
            egoverse_loss_total=total_loss,
        )
        # Record the *actual*, post-scheduler video noise level used by this
        # forward pass.  This is deliberately derived from ``timesteps`` rather
        # than re-sampling the configured distribution, so checkpoint replay
        # can attribute a loss/gradient spike to the exact sigma without
        # consuming or perturbing RNG state.
        max_timestep = float(self.rectified_flow_video.noise_scheduler.config.num_train_timesteps)
        video_sigma = timesteps.detach().float() / max_timestep
        losses.update(
            egoverse_sigma_video_mean=video_sigma.mean(),
            egoverse_sigma_video_min=video_sigma.min(),
            egoverse_sigma_video_max=video_sigma.max(),
            egoverse_sigma_video_low_0_1_fraction=(video_sigma < 0.1).float().mean(),
            egoverse_sigma_video_low_0_2_fraction=(video_sigma < 0.2).float().mean(),
        )
        for name, value in self._last_visibility_loss_metrics.items():
            if name.endswith("_loss"):
                losses[f"egoverse_loss_action_{name.removesuffix('_loss')}_raw"] = value * sample_scale
        return total_loss, losses


class EgoVerseOmniMoTModel(EgoVerseLossMixin, OmniMoTModel):
    """Thin loss adapter; the Cosmos Generator architecture is unchanged."""

    def __init__(self, config, lambda_out_of_fov: float = 0.0, subblock_equal_weight: bool = False):
        super().__init__(config, lambda_out_of_fov=lambda_out_of_fov, subblock_equal_weight=subblock_equal_weight)
        self._fixed_pack_action_intervention = "original"

    def _add_noise_to_input(self, *args, **kwargs):
        result = super()._add_noise_to_input(*args, **kwargs)
        mode = self._fixed_pack_action_intervention
        if mode == "original":
            return result
        if mode not in {"zero", "reverse"}:
            raise ValueError(f"Unknown fixed-pack action intervention: {mode}")
        packed_sequence = args[1] if len(args) > 1 else kwargs["packed_sequence"]
        if result.xt_tokens_action is None or packed_sequence.action is None:
            raise RuntimeError("Fixed-pack action intervention requires future action tokens")
        for noisy_action, condition_mask in zip(
            result.xt_tokens_action,
            packed_sequence.action.condition_mask,
            strict=True,
        ):
            future = condition_mask.reshape(-1) == 0
            if mode == "zero":
                noisy_action[future] = 0
            else:
                noisy_action[future] = noisy_action[future].flip(0)
        return result

    @staticmethod
    def _relative_l2(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        squared_diff = sum(
            (lhs.float() - rhs.float()).square().sum() for lhs, rhs in zip(reference, candidate, strict=True)
        )
        squared_ref = sum(lhs.float().square().sum() for lhs in reference)
        max_abs = torch.stack(
            [(lhs.float() - rhs.float()).abs().max() for lhs, rhs in zip(reference, candidate, strict=True)]
        ).max()
        return torch.sqrt(squared_diff / squared_ref.clamp_min(1e-20)), max_abs

    def training_step(self, data_batch: dict[str, torch.Tensor], iteration: int):
        diagnose = os.environ.get("EGOVERSE_FIXED_PACK_ACTION_INTERVENTION") == "1" and iteration == int(
            os.environ.get("EGOVERSE_FIXED_PACK_ACTION_INTERVENTION_ITER", "0")
        )
        if not diagnose:
            return super().training_step(data_batch, iteration)

        cpu_rng_before = torch.get_rng_state()
        cuda_rng_before = torch.cuda.get_rng_state_all()
        metrics: dict[str, float] = {}
        try:
            reference_video: list[torch.Tensor] | None = None
            original_action_loss: torch.Tensor | None = None
            for label, mode in (
                ("original", "original"),
                ("repeat", "original"),
                ("zero", "zero"),
                ("reverse", "reverse"),
            ):
                torch.set_rng_state(cpu_rng_before)
                torch.cuda.set_rng_state_all(cuda_rng_before)
                self._fixed_pack_action_intervention = mode
                with torch.no_grad():
                    candidate_output, candidate_loss = super().training_step(data_batch, iteration)
                if label == "original":
                    reference_video = [tensor.detach().clone() for tensor in candidate_output["model_pred"]]
                    original_action_loss = candidate_output["egoverse_loss_action_raw"].detach().float()
                    del candidate_output, candidate_loss
                    continue
                assert reference_video is not None and original_action_loss is not None
                relative_l2, max_abs = self._relative_l2(reference_video, candidate_output["model_pred"])
                action_loss_delta = (
                    candidate_output["egoverse_loss_action_raw"].detach().float() - original_action_loss
                ).abs()
                metrics[f"video_{label}_relative_l2"] = float(relative_l2.item())
                metrics[f"video_{label}_max_abs"] = float(max_abs.item())
                metrics[f"action_loss_{label}_abs_delta"] = float(action_loss_delta.item())
                del candidate_output, candidate_loss
        finally:
            self._fixed_pack_action_intervention = "original"
            torch.set_rng_state(cpu_rng_before)
            torch.cuda.set_rng_state_all(cuda_rng_before)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("FIXED_PACK_ACTION_INTERVENTION " + json.dumps(metrics, sort_keys=True), flush=True)
        original_output, original_loss = super().training_step(data_batch, iteration)
        return original_output, original_loss
