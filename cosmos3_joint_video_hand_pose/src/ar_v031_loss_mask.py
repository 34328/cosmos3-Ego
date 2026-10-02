"""Zero only prefix error numerators; original keep masks and denominators stay intact."""
from __future__ import annotations

import torch

from .ar_v02_layout import ACTION, VIDEO


def prefix_rows(packed, plan):
    """Loss-only row selection; never changes packing, conditions or history."""
    result = {"vision": [], "action": []}
    for index, layout in enumerate(packed.joint_layouts):
        for name, role, metadata in (
            ("vision", VIDEO, layout.video_metadata),
            ("action", ACTION, layout.action_metadata),
        ):
            condition = getattr(packed, name).condition_mask[index]
            roles, chunks, _ = metadata(device=condition.device)
            prefix = torch.zeros_like(roles, dtype=torch.bool) if plan is None else (
                plan.mask[index].to(condition.device)[chunks] & (roles == role)
            )
            result[name].append(prefix)
    return result


def numerator_only_predictions(predictions, targets, prefixes):
    """FP32 target substitution gives zero prefix error, preserving suffix gradients."""
    result = []
    for prediction, target, prefix in zip(predictions, targets, prefixes, strict=True):
        shape = (-1, 1) if prediction.ndim == 2 else (
            *([1] * (prediction.ndim - 3)), -1, 1, 1
        )
        result.append(torch.where(prefix.reshape(shape), target.float(), prediction.float()))
    return result


@torch.no_grad()
def raw_group_moments(out_net, noised, packed, prefixes):
    """Unweighted SSE/count for video/action × prefix/target; U/S and padding excluded."""
    values = []
    for name, key, targets in (
        ("vision", "preds_vision", noised.vt_target_vision),
        ("action", "preds_action", noised.vt_target_action),
    ):
        sums = [out_net[key][0].new_zeros((), dtype=torch.float64) for _ in range(4)]
        for i, (prediction, target, prefix) in enumerate(zip(
            out_net[key], targets, prefixes[name], strict=True
        )):
            condition = getattr(packed, name).condition_mask[i].reshape(-1).bool()
            if name == "vision":
                keep = (~condition).reshape(-1, 1, 1).expand_as(prediction)
                prefix = prefix.reshape(*([1] * (prediction.ndim - 3)), -1, 1, 1)
            else:
                prediction, target = prediction[:, :57], target[:, :57]
                keep = (~condition)[:, None].expand_as(prediction).clone()
                masks = packed.action.action_valid_mask
                valid = None if masks is None else masks[i]
                if valid is not None:
                    if valid.ndim == 1:
                        valid = valid[None]
                    keep &= valid[:, :57].bool()
                prefix = prefix[:, None]
            error = (torch.where(keep, prediction.float(), 0) -
                     torch.where(keep, target.float(), 0)).square()
            for offset, group in ((0, prefix), (2, ~prefix)):
                selected = keep & group
                sums[offset] += torch.where(selected, error, 0).sum(dtype=torch.float64)
                sums[offset + 1] += selected.sum()
        values.extend((torch.stack(sums[:2]), torch.stack(sums[2:])))
    return torch.stack(values)
