"""V0.3.1 native RF/packing/loss contracts and differentiable noisy history on CPU."""

import pytest
import torch

from test_ar_v03_attention import eager_cpu_flex, memory_value
from test_ar_v03_model import data_fixture, model_fixture
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import (
    ACTION, CONDITION_VIDEO, STATE, VIDEO, JointChunkLayout,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence
from cosmos3_joint_video_hand_pose.src.ar_v03_attention import JointDiffusionForcingAttention
from cosmos3_joint_video_hand_pose.src.ar_v03_model import EgoVerseARV03Model
from cosmos3_joint_video_hand_pose.src.ar_v03_sigma import sample_prefix_low_noise
from cosmos3_joint_video_hand_pose.src.ar_v031_model import EgoVerseARV031Model


def prefix_iteration(layouts):
    """Use a real reproducible draw with block 1 as prefix and block 2 as target."""
    for iteration in range(64):
        plan = sample_prefix_low_noise(
            [len(x.boundaries) for x in layouts], seed=42, iteration=iteration, rank=0,
        )
        if plan.metadata[0]["prefix_length"] == 2 and all(
            row["prefix_length"] >= min(2, row["n_chunks"]) for row in plan.metadata
        ):
            return iteration
    raise AssertionError("fixture needs a nonempty actual prefix")


def prepared_case(layouts, *, mask=True):
    torch.manual_seed(107)
    model = model_fixture(layouts, cls=EgoVerseARV031Model)
    # The old fixture's exact-class test selects TF for a subclass. V031 is DF.
    model.config.causal_training_strategy = "diffusion_forcing"
    model.config.prefix_low_noise_enabled = True
    model.config.mask_prefix_loss = mask
    model.rectified_flow_video = RectifiedFlow(
        lambda *args: None, train_time_distribution="waver", shift=5,
    )
    model.rectified_flow_action = RectifiedFlow(
        lambda *args: None, train_time_distribution="logitnormal", shift=5,
    )
    data = data_fixture(layouts)
    iteration = prefix_iteration(layouts)
    times, sigmas = model._get_train_noise_level_vision(
        len(layouts), False, [x.num_video_frames for x in layouts],
        ["480"] * len(layouts), iteration=iteration,
    )
    model._get_train_noise_level_action(len(layouts), iteration=iteration)
    packed = pack_joint_sequence(
        layout=layouts, gen_data_clean=data, text_ids=[[3, 4]] * len(layouts),
        special_tokens=model.llm_special_tokens, timesteps=times,
        latent_patch_size=2, condition_frames=[()] * len(layouts),
    )
    noised = model._add_noise_to_input(data, packed, sigmas, iteration=iteration)
    for i, layout in enumerate(layouts):
        vr, _, _ = layout.video_metadata()
        ar, _, _ = layout.action_metadata()
        for name in ("vision", "action"):
            torch.testing.assert_close(
                getattr(noised, "vt_target_" + name)[i],
                getattr(noised, "epsilon_" + name)[i] - getattr(data, "x0_tokens_" + name)[i],
                atol=0, rtol=0,
            )
            assert not getattr(noised, "vt_target_" + name)[i].requires_grad
        torch.testing.assert_close(
            noised.xt_tokens_vision[i][:, :, vr == CONDITION_VIDEO],
            data.x0_tokens_vision[i][:, :, vr == CONDITION_VIDEO], atol=0, rtol=0,
        )
        torch.testing.assert_close(
            noised.xt_tokens_action[i][ar == STATE], data.x0_tokens_action[i][ar == STATE],
            atol=0, rtol=0,
        )
    return model, packed, noised, times


def predictions(noised, dtype=torch.float32):
    return {
        "preds_" + name: [
            (target + (i + 1) * 1.25).to(dtype).detach().requires_grad_()
            for i, target in enumerate(getattr(noised, "vt_target_" + name))
        ] for name in ("vision", "action")
    }


def objective_reference(out, packed, noised, lengths, *, suppress_prefix):
    """Independent weighted SSE/full-coordinate denominator, then active-sample mean."""
    values, active_counts = {}, {}
    weights = torch.ones(57)
    weights[9:18] = weights[33:42] = 3
    for name in ("vision", "action"):
        losses = []
        for i, layout in enumerate(packed.joint_layouts):
            pred, target = out["preds_" + name][i], getattr(noised, "vt_target_" + name)[i]
            condition = getattr(packed, name).condition_mask[i].reshape(-1).bool()
            if name == "vision":
                roles, chunks, _ = layout.video_metadata()
                keep = (~condition).view(1, 1, -1, 1, 1).expand_as(pred)
                prefix = ((roles == VIDEO) & (chunks < lengths[i])).view(1, 1, -1, 1, 1)
                error = (pred.float() - target.float()).square()
                denominator = keep.sum()
                numerator = torch.where(keep & (~prefix if suppress_prefix else True), error, 0).sum()
            else:
                roles, chunks, _ = layout.action_metadata()
                keep = (~condition)[:, None].expand(-1, 57).clone()
                masks = packed.action.action_valid_mask
                valid = None if masks is None else masks[i]
                if valid is not None:
                    keep &= valid.reshape(1, -1)[:, :57].bool() if valid.ndim == 1 else valid[:, :57].bool()
                prefix = ((roles == ACTION) & (chunks < lengths[i]))[:, None]
                error = (pred[:, :57].float() - target[:, :57].float()).square()
                denominator = (keep * weights).sum()
                numerator = (torch.where(keep & (~prefix if suppress_prefix else True), error, 0) * weights).sum()
            if keep.any():
                losses.append(numerator / denominator)
        active_counts[name] = len(losses)
        values[name] = torch.stack(losses).mean() if losses else out["preds_" + name][0].sum() * 0
    return values["vision"] + values["action"], values, active_counts


@pytest.mark.parametrize("validity", ["absent", "none_entries", "channels", "rows"])
def test_native_compute_suppresses_only_numerators_and_preserves_full_denominators(validity):
    layouts = [JointChunkLayout(10, 1, 4), JointChunkLayout(6, 1, 4)]
    model, packed, noised, times = prepared_case(layouts)
    lengths = [row["prefix_length"] for row in model._ar_step.prefix_low_noise]
    if validity == "absent":
        packed.action.action_valid_mask = None
    elif validity == "none_entries":
        packed.action.action_valid_mask = [None, None]
    elif validity == "channels":
        valid = torch.ones(64, dtype=torch.bool)
        valid[9:12] = False
        packed.action.action_valid_mask = [valid, torch.zeros(64, dtype=torch.bool)]
    else:
        # Sample 0 has valid action coordinates ONLY in the prefix. It must still
        # count as active after its error numerator becomes zero.
        masks = []
        for i, layout in enumerate(layouts):
            _, chunks, _ = layout.action_metadata()
            valid = torch.ones(layout.num_action_rows, 64, dtype=torch.bool)
            if i == 0:
                valid[chunks >= lengths[i]] = False
            else:
                valid[::3, 33:42] = False
            masks.append(valid)
        packed.action.action_valid_mask = masks
    out = predictions(noised)
    original_masks = {name: [x.clone() for x in getattr(packed, name).condition_mask]
                      for name in ("vision", "action")}
    original_targets = {name: [x.clone() for x in getattr(noised, "vt_target_" + name)]
                        for name in ("vision", "action")}
    expected, terms, active = objective_reference(out, packed, noised, lengths, suppress_prefix=True)
    expected_grads = torch.autograd.grad(
        expected, [*out["preds_vision"], *out["preds_action"]],
        allow_unused=True, materialize_grads=True,
    )
    loss, stats = model._compute_whole_losses(out, packed, noised, times, False)
    raw_fields = dict(model._last_visibility_loss_metrics)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(model._ar_backward_loss, expected)
    for name in ("vision", "action"):
        torch.testing.assert_close(stats["flow_matching_loss_" + name], terms[name])
        assert stats["egoverse_global_" + ("video" if name == "vision" else "action") + "_samples"] == active[name]
        for actual, old in zip(getattr(packed, name).condition_mask, original_masks[name]):
            assert torch.equal(actual, old)
        for actual, old in zip(getattr(noised, "vt_target_" + name), original_targets[name]):
            assert torch.equal(actual, old)
    actual_grads = torch.autograd.grad(loss, [*out["preds_vision"], *out["preds_action"]])
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference)
    # Raw eight field losses remain the all-future V03 values, despite masking
    # the optimization numerator. Diagnostics must not retain a graph.
    EgoVerseARV03Model._compute_whole_losses(model, out, packed, noised, times, False)
    assert len(raw_fields) == 8 and raw_fields.keys() == model._last_visibility_loss_metrics.keys()
    for name, value in raw_fields.items():
        assert torch.equal(value, model._last_visibility_loss_metrics[name])
        assert not value.requires_grad
    assert not model._ar_v031_group_mse_moments.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mask_off_is_bitwise_v03_loss_stats_and_prediction_gradients(dtype):
    layouts = [JointChunkLayout(10, 1, 4), JointChunkLayout(6, 1, 4)]
    model, packed, noised, times = prepared_case(layouts, mask=False)
    out = predictions(noised, dtype)
    leaves = [*out["preds_vision"], *out["preds_action"]]
    actual, stats = model._compute_whole_losses(out, packed, noised, times, False)
    actual_fields = dict(model._last_visibility_loss_metrics)
    actual_grad = torch.autograd.grad(actual, leaves)
    reference, reference_stats = EgoVerseARV03Model._compute_whole_losses(
        model, out, packed, noised, times, False,
    )
    assert torch.equal(actual, reference)
    assert stats.keys() == reference_stats.keys()
    for name in stats:
        assert torch.equal(stats[name], reference_stats[name]), name
    for name in actual_fields:
        assert torch.equal(actual_fields[name], model._last_visibility_loss_metrics[name])
    for actual, wanted in zip(actual_grad, torch.autograd.grad(reference, leaves)):
        assert torch.equal(actual, wanted)


def test_native_rf_target_gives_zero_loss_in_both_versions():
    model, packed, noised, times = prepared_case([JointChunkLayout(10, 1, 4)])
    out = {"preds_" + name: [x.clone().requires_grad_() for x in getattr(noised, "vt_target_" + name)]
           for name in ("vision", "action")}
    for compute in (model._compute_whole_losses,
                    lambda *args: EgoVerseARV03Model._compute_whole_losses(model, *args)):
        loss, stats = compute(out, packed, noised, times, False)
        assert loss == 0 and stats["flow_matching_loss_vision"] == stats["flow_matching_loss_action"] == 0
        for grad in torch.autograd.grad(loss, [*out["preds_vision"], *out["preds_action"]]):
            assert torch.count_nonzero(grad) == 0


def test_prefix_prediction_gradient_is_zero_but_noisy_history_inputs_receive_suffix_gradient(eager_cpu_flex):
    # Native eager FlexAttention + real noised inputs and V031 loss. Full Cosmos
    # modality encode/decode is exercised by the separately scheduled GPU test.
    layouts = [JointChunkLayout(10, 1, 4), JointChunkLayout(2, 1, 4)]
    model, packed, noised, times = prepared_case(layouts)
    video = [x.detach().requires_grad_() for x in noised.xt_tokens_vision]
    action = [x.detach().requires_grad_() for x in noised.xt_tokens_action]
    gen, metadata = [], []
    for layout, v, a in zip(layouts, video, action):
        vrows = torch.nn.functional.pad(v.permute(0, 2, 1, 3, 4).reshape(layout.num_video_frames, 16), (0, 48))
        gen.extend(vrows[start:start + count] if role in (VIDEO, CONDITION_VIDEO) else a[start:start + count]
                   for role, _, start, count, _ in layout.spans())
        metadata.append(layout.metadata())
    hidden = torch.cat(gen)
    text_lengths = list(packed.joint_text_lengths)
    text_count = sum(text_lengths)
    attention = JointDiffusionForcingAttention(layouts, "cpu", text_lengths=text_lengths)
    text_k, text_v = [torch.randn(1, text_count, 1, 32) for _ in range(2)]
    for _ in range(2):
        projections = [torch.randn(width, 64) / 8 for width in (64, 32, 32)]
        q, k, v = [torch.nn.functional.linear(hidden, weight).reshape(1, 1, len(hidden), heads, 32)
                   for weight, heads in zip(projections, (2, 1, 1))]
        hidden = hidden + .2 * attention(q, k, v, text_k, text_v, memory_value("cpu", text_count)).reshape(-1, 64)
    out = {"preds_vision": [], "preds_action": []}
    offset = 0
    for layout, (r, _, _) in zip(layouts, metadata):
        h = hidden[offset:offset + layout.num_tokens]
        out["preds_vision"].append(h[(r == VIDEO) | (r == CONDITION_VIDEO), :16].reshape(
            layout.num_video_frames, 4, 2, 2).permute(1, 0, 2, 3).unsqueeze(0))
        out["preds_action"].append(h[(r == ACTION) | (r == STATE)])
        offset += layout.num_tokens
    selected = {"preds_vision": [], "preds_action": []}
    for i, layout in enumerate(layouts):
        vr, vc, _ = layout.video_metadata()
        ar, ac, _ = layout.action_metadata()
        # Keep real block-1 predictions here: V031 itself must remove their
        # direct error, while block 2 must still backpropagate through history.
        selected["preds_vision"].append(torch.where(
            ((vr == VIDEO) & (vc <= 2) & (i == 0)).view(1, 1, -1, 1, 1),
            out["preds_vision"][i], noised.vt_target_vision[i]))
        selected["preds_action"].append(torch.where(
            ((ar == ACTION) & (ac <= 2) & (i == 0))[:, None],
            out["preds_action"][i], noised.vt_target_action[i]))
    loss, _ = model._compute_whole_losses(selected, packed, noised, times, False)
    leaves = [*video, *action, *out["preds_vision"], *out["preds_action"]]
    grads = torch.autograd.grad(loss, leaves)
    vg, ag, pg_v, pg_a = (grads[i:i+2] for i in range(0, 8, 2))
    assert torch.isfinite(loss) and loss > 0
    layout = layouts[0]
    vr, vc, _ = layout.video_metadata()
    ar, ac, _ = layout.action_metadata()
    for name, grad, pred_grad, role, chunk in (
        ("video", vg[0], pg_v[0], vr, vc), ("action", ag[0], pg_a[0], ar, ac),
    ):
        prefix = (role == (VIDEO if name == "video" else ACTION)) & (chunk == 1)
        source = grad[:, :, prefix] if name == "video" else grad[prefix, :57]
        output = pred_grad[:, :, prefix] if name == "video" else pred_grad[prefix]
        assert torch.isfinite(source).all() and source.norm() > 0
        assert torch.count_nonzero(output) == 0
        suffix = (role == (VIDEO if name == "video" else ACTION)) & (chunk == 2)
        suffix_output = pred_grad[:, :, suffix] if name == "video" else pred_grad[suffix, :57]
        assert torch.isfinite(suffix_output).all() and suffix_output.norm() > 0
        future = grad[:, :, chunk > 2] if name == "video" else grad[chunk > 2]
        assert torch.count_nonzero(future) == 0
    assert torch.count_nonzero(vg[1]) == torch.count_nonzero(ag[1]) == 0
    assert torch.count_nonzero(pg_v[0][:, :, vr == CONDITION_VIDEO]) == 0
    assert torch.count_nonzero(pg_a[0][ar == STATE]) == torch.count_nonzero(pg_a[0][:, 57:]) == 0
