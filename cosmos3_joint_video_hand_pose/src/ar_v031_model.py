"""V0.3.1: suppress prefix numerators without changing the V0.3.0 denominator."""
from __future__ import annotations

import attrs
import torch

from .ar_v03_loss import whole_action_flow_loss
from .ar_v03_model import EgoVerseARV03Model, EgoVerseARV03ModelConfig
from .ar_v031_loss_mask import numerator_only_predictions, prefix_rows, raw_group_moments


@attrs.define(slots=False)
class EgoVerseARV031ModelConfig(EgoVerseARV03ModelConfig):
    mask_prefix_loss: bool = True


class EgoVerseARV031Model(EgoVerseARV03Model):
    def __init__(self, config, **kwargs):
        if not isinstance(config.mask_prefix_loss, bool):
            raise ValueError("mask_prefix_loss must be a bool")
        super().__init__(config, **kwargs)

    def _compute_whole_losses(self, out_net, packed, noised, timesteps, is_image_batch):
        plan = getattr(self._ar_step, "prefix_low_noise_plan", None)
        if self.config.prefix_low_noise_enabled and plan is None:
            raise RuntimeError("loss arrived before the actual prefix noise plan")
        prefixes = prefix_rows(packed, plan)
        self._ar_v031_group_mse_moments = raw_group_moments(out_net, noised, packed, prefixes)
        loss_out = out_net
        if self.config.mask_prefix_loss:
            loss_out = dict(out_net)
            loss_out["preds_vision"] = numerator_only_predictions(
                out_net["preds_vision"], noised.vt_target_vision, prefixes["vision"]
            )
            loss_out["preds_action"] = numerator_only_predictions(
                out_net["preds_action"], noised.vt_target_action, prefixes["action"]
            )
        # The unchanged packed object preserves every original denominator and
        # active-sample count. This view is created after the single forward.
        result = super()._compute_whole_losses(
            loss_out, packed, noised, timesteps, is_image_batch
        )
        if self.config.mask_prefix_loss:
            # Preserve the original eight unweighted ALL-future field logs.
            with torch.no_grad():
                _, raw = whole_action_flow_loss(
                    pred=[x.detach() for x in out_net["preds_action"]],
                    target=[x.detach() for x in noised.vt_target_action],
                    condition_mask=packed.action.condition_mask,
                    visibility=self._current_hand_visibility,
                    valid_mask=packed.action.action_valid_mask,
                    mask_out_of_fov=False, collect_field_metrics=True,
                    channel_weights=self.config.action_channel_weights,
                )
                window = self._ar_loss_window
                self._last_visibility_loss_metrics = {
                    name + "_loss": values.sum() * window.world_size * window.microsteps /
                    window.global_counts[1].clamp_min(1)
                    for name, values in raw["field_per_sample_losses"].items()
                }
        return result

