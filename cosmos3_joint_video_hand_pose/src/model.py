from __future__ import annotations

from typing import Any

import torch

from .loss import visibility_weighted_action_flow_loss


try:
    from cosmos_framework.model.generator.mot.context_parallel_utils import broadcast_context_parallel_object
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
except ImportError as error:  # pragma: no cover - exercised only outside the Cosmos environment
    raise ImportError(
        "EgoVerseOmniMoTModel requires PYTHONPATH=/mnt/lzh/cosmos/packages/cosmos3"
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


class EgoVerseOmniMoTModel(OmniMoTModel):
    """Thin loss adapter; the Cosmos Generator architecture is unchanged."""

    def __init__(self, config, lambda_out_of_fov: float = 0.0, subblock_equal_weight: bool = False):
        super().__init__(config)
        if not 0 <= lambda_out_of_fov <= 1:
            raise ValueError("lambda_out_of_fov must be in [0,1]")
        self.lambda_out_of_fov = float(lambda_out_of_fov)
        self.subblock_equal_weight = bool(subblock_equal_weight)
        self._current_hand_visibility: list[torch.Tensor] | None = None
        self._cp_local_hand_visibility: list[torch.Tensor] | None = None

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
        for name, value in getattr(self, "_last_visibility_loss_metrics", {}).items():
            if name.endswith("_loss"):
                losses[f"egoverse_loss_action_{name.removesuffix('_loss')}_raw"] = value * sample_scale
        return total_loss, losses
