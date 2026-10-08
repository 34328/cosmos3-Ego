from __future__ import annotations

import json
import os
from typing import Any

import torch

from .action import CODEC_ROOT
from .codec import FrozenHandMLPAE15
from .geometry_loss import (
    GeometryLossConfig,
    LEFT_HAND_LATENT,
    RIGHT_HAND_LATENT,
    clean_action_from_target,
    hand_geometry_losses,
    predict_clean_action,
)
from .loss import visibility_weighted_action_flow_loss


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
        raise KeyError("hand_visibility is required for EgoVerse action loss")
    items = raw if isinstance(raw, list) else [raw]
    result = []
    for item in items:
        tensor = torch.as_tensor(item, dtype=torch.bool)
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError(f"hand_visibility must be [T,2], got {tuple(tensor.shape)}")
        result.append(tensor.cpu())
    return result


def _geometry_targets_from_batch(data_batch: dict[str, Any]) -> dict[str, list[torch.Tensor]]:
    result: dict[str, list[torch.Tensor]] = {}
    for key in ("right_hand_local_gt", "left_hand_local_gt"):
        raw = data_batch.get(key)
        if raw is None:
            raise KeyError(f"{key} is required when geometry loss is enabled")
        items = raw if isinstance(raw, list) else [raw]
        tensors = []
        for item in items:
            while isinstance(item, list) and len(item) == 1:
                item = item[0]
            tensor = torch.as_tensor(item, dtype=torch.float32)
            while tensor.ndim > 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 3 or tensor.shape[-2:] != (20, 3):
                raise ValueError(f"{key} must be [T,20,3], got {tuple(tensor.shape)}")
            tensors.append(tensor.cpu())
        result[key] = tensors
    return result


class EgoVerseOmniMoTModel(OmniMoTModel):
    """Thin loss adapter; the Cosmos Generator architecture is unchanged."""

    def __init__(
        self,
        config,
        lambda_out_of_fov: float = 0.0,
        subblock_equal_weight: bool = False,
        geometry_loss: GeometryLossConfig | dict[str, Any] | None = None,
        right_hand_codec: str = str(CODEC_ROOT / "right_mlp15_primary.pt"),
        left_hand_codec: str = str(CODEC_ROOT / "left_mlp15_primary.pt"),
    ):
        super().__init__(config)
        if not 0 <= lambda_out_of_fov <= 1:
            raise ValueError("lambda_out_of_fov must be in [0,1]")
        self.lambda_out_of_fov = float(lambda_out_of_fov)
        self.subblock_equal_weight = bool(subblock_equal_weight)
        self.geometry_loss_config = GeometryLossConfig.from_value(geometry_loss)
        if self.geometry_loss_config.enabled:
            # Keep fixed codecs out of FSDP, optimizer groups and DCP state.
            object.__setattr__(self, "right_hand_geometry_codec", FrozenHandMLPAE15(right_hand_codec))
            object.__setattr__(self, "left_hand_geometry_codec", FrozenHandMLPAE15(left_hand_codec))
        else:
            self.right_hand_geometry_codec = None
            self.left_hand_geometry_codec = None
        self._current_hand_visibility: list[torch.Tensor] | None = None
        self._cp_local_hand_visibility: list[torch.Tensor] | None = None
        self._current_geometry_targets: dict[str, list[torch.Tensor]] | None = None
        self._cp_local_geometry_targets: dict[str, list[torch.Tensor]] | None = None
        self._geometry_iteration = 0
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
        squared_diff = sum((lhs.float() - rhs.float()).square().sum() for lhs, rhs in zip(reference, candidate, strict=True))
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

    def _get_training_inputs(self, data_batch: dict[str, torch.Tensor], iteration: int):
        self._geometry_iteration = int(iteration)
        cp_enabled = self.parallel_dims is not None and self.parallel_dims.cp_enabled
        owner_slot = self._cp_window_slot
        if not cp_enabled:
            self._current_hand_visibility = _visibility_from_batch(data_batch)
            if self.geometry_loss_config.enabled:
                self._current_geometry_targets = _geometry_targets_from_batch(data_batch)
            return super()._get_training_inputs(data_batch, iteration)

        cp_size = self.parallel_dims.cp_mesh.size()
        if owner_slot == 0:
            self._cp_local_hand_visibility = _visibility_from_batch(data_batch)
            if self.geometry_loss_config.enabled:
                self._cp_local_geometry_targets = _geometry_targets_from_batch(data_batch)
        result = super()._get_training_inputs(data_batch, iteration)
        self._current_hand_visibility = broadcast_context_parallel_object(
            self._cp_local_hand_visibility,
            self.parallel_dims,
            owner_rank=owner_slot,
        )
        if self.geometry_loss_config.enabled:
            self._current_geometry_targets = broadcast_context_parallel_object(
                self._cp_local_geometry_targets, self.parallel_dims, owner_rank=owner_slot
            )
        if owner_slot == cp_size - 1:
            self._cp_local_hand_visibility = None
            self._cp_local_geometry_targets = None
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
        normalize_by_active=False,
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
                normalize_by_active=normalize_by_active,
            )
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
    ):
        """Expose raw and actually weighted components for distributed logging."""
        total_loss, losses = super()._compute_losses(
            out_net=out_net,
            data_batch_packed=data_batch_packed,
            gen_data_noised=gen_data_noised,
            timesteps=timesteps,
            is_image_batch=is_image_batch,
            timesteps_action=timesteps_action,
            timesteps_sound=timesteps_sound,
        )
        rf_cfg = self.config.rectified_flow_training_config
        sample_scale = torch.ones((), device=total_loss.device, dtype=total_loss.dtype)
        if rf_cfg.sample_level_loss_averaging and self.config.vision_gen:
            sample_scale = self._sample_level_loss_scale(
                is_image_batch=is_image_batch,
                num_samples=len(out_net["preds_vision"]),
                device=self.tensor_kwargs_fp32["device"],
            ).to(device=total_loss.device, dtype=total_loss.dtype)

        video_raw = losses["flow_matching_loss_vision"] * sample_scale
        action_raw = losses["flow_matching_loss_action"] * sample_scale
        video_weight = (
            rf_cfg.image_loss_scale
            if is_image_batch and rf_cfg.image_loss_scale is not None
            else rf_cfg.loss_scale
        )
        losses.update(
            egoverse_loss_video_raw=video_raw,
            egoverse_loss_action_raw=action_raw,
            egoverse_loss_video_weighted=video_raw * video_weight,
            egoverse_loss_action_weighted=action_raw * rf_cfg.action_loss_weight,
            egoverse_loss_total=total_loss,
        )
        if self.geometry_loss_config.enabled:
            geometry_total, geometry_metrics = self._compute_geometry_loss(
                out_net=out_net,
                data_batch_packed=data_batch_packed,
                gen_data_noised=gen_data_noised,
            )
            geometry_total = geometry_total * sample_scale
            total_loss = total_loss + geometry_total
            losses["egoverse_loss_total"] = total_loss
            losses["egoverse_loss_geometry_weighted"] = geometry_total
            losses.update({f"egoverse_geometry_{name}": value for name, value in geometry_metrics.items()})
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
        for name, value in getattr(self, "_last_visibility_loss_metrics", {}).items():
            if name.endswith("_loss"):
                losses[f"egoverse_loss_action_{name.removesuffix('_loss')}_raw"] = value * sample_scale
        return total_loss, losses

    def _compute_geometry_loss(self, *, out_net, data_batch_packed, gen_data_noised):
        config = self.geometry_loss_config
        if self._current_geometry_targets is None or self._current_hand_visibility is None:
            raise RuntimeError("geometry loss reached without synchronized GT geometry")
        if data_batch_packed.action is None or gen_data_noised.xt_tokens_action is None:
            dummy = 0.0 * sum(item.sum() for item in out_net["preds_action"])
            return dummy, {"decode_raw": dummy.detach(), "bone_raw": dummy.detach(), "velocity_raw": dummy.detach()}

        sample_metrics = []
        for index, (prediction, noisy, epsilon, target_velocity, sigma, condition_mask, visibility) in enumerate(
            zip(
                out_net["preds_action"],
                gen_data_noised.xt_tokens_action,
                gen_data_noised.epsilon_action,
                gen_data_noised.vt_target_action,
                gen_data_noised.sigmas_action,
                data_batch_packed.action.condition_mask,
                self._current_hand_visibility,
                strict=True,
            )
        ):
            if prediction.shape[-1] < 57:
                raise ValueError("geometry loss requires action predictions with at least 57 channels")
            x0_pred = predict_clean_action(noisy[:, :57], prediction[:, :57], sigma)
            x0_oracle = clean_action_from_target(epsilon[:, :57], target_velocity[:, :57])
            if not torch.isfinite(x0_pred).all():
                raise FloatingPointError("non-finite predicted clean action")
            sigma_frames = sigma.reshape(-1).to(device=prediction.device)
            if sigma_frames.numel() == 1:
                sigma_frames = sigma_frames.expand(len(prediction))
            active = (1.0 - condition_mask.reshape(-1).to(prediction)).bool()
            active &= (sigma_frames >= config.sigma_min) & (sigma_frames <= config.sigma_max)
            visible = visibility.to(device=prediction.device)

            hand_results = []
            for side, channel_slice, codec, visibility_index in (
                ("right", RIGHT_HAND_LATENT, self.right_hand_geometry_codec, 0),
                ("left", LEFT_HAND_LATENT, self.left_hand_geometry_codec, 1),
            ):
                assert codec is not None
                if codec.input_mean.device != prediction.device:
                    codec.to(prediction.device)
                predicted_points = codec.decode_differentiable(x0_pred[:, channel_slice])
                oracle_points = codec.decode_differentiable(x0_oracle[:, channel_slice]).detach()
                raw_gt = self._current_geometry_targets[f"{side}_hand_local_gt"][index].to(
                    device=prediction.device, dtype=torch.float32
                )
                if len(raw_gt) != len(prediction):
                    raise ValueError(f"{side} geometry target length does not match action length")
                hand_results.append(
                    hand_geometry_losses(
                        predicted_points=predicted_points,
                        oracle_points=oracle_points,
                        raw_gt_points=raw_gt,
                        visible=visible[:, visibility_index],
                        active=active,
                        config=config,
                    )
                )
            sample_metrics.append(
                {key: torch.stack([item[key] for item in hand_results]).mean() for key in hand_results[0]}
            )

        metrics = {key: torch.stack([item[key] for item in sample_metrics]).mean() for key in sample_metrics[0]}
        ramp = config.ramp(self._geometry_iteration)
        weighted = ramp * (
            config.decode_weight * metrics["decode"]
            + config.bone_weight * metrics["bone"]
            + config.velocity_weight * metrics["velocity"]
        )
        detached = {f"{key}_raw": value.detach() for key, value in metrics.items()}
        detached["ramp"] = weighted.new_tensor(ramp)
        return weighted, detached
