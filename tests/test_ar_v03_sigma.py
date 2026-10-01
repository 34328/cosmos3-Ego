"""Prefix routing, private RNG/resume, and actual official-RF histograms."""

import json
import os
from pathlib import Path
import pytest
import torch
from test_ar_v03_model import model_fixture, data_fixture
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, CONDITION_VIDEO, STATE, VIDEO, JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence
from cosmos3_joint_video_hand_pose.src.ar_v03_sigma import continuous_rf_timesteps, sample_prefix_low_noise


def official_model(layouts, enabled, cls=None):
    from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
    model = model_fixture(layouts) if cls is None else model_fixture(layouts, cls=cls)
    model.config.prefix_low_noise_enabled = enabled
    model.rectified_flow_video = RectifiedFlow(lambda *args: None, train_time_distribution="waver", shift=5)
    model.rectified_flow_action = RectifiedFlow(lambda *args: None, train_time_distribution="logitnormal", shift=5)
    return model


def draw(model, iteration):
    counts = [x.num_video_frames for x in model._joint_layouts]
    vt, vs = model._get_train_noise_level_vision(len(counts), False, counts, ["480"] * len(counts), iteration=iteration)
    at, acts = model._get_train_noise_level_action(len(counts), iteration=iteration)
    return vt, vs, at, acts


def test_off_switch_and_prefix_do_not_change_original_rf_or_later_rng():
    layouts = [JointChunkLayout(69, 1, 4), JointChunkLayout(18, 1, 4), JointChunkLayout(2, 1, 4)]
    observed = []
    for enabled in (False, True):
        model = official_model(layouts, enabled)
        torch.manual_seed(153)
        outputs = draw(model, 37)
        observed.append((outputs, model._ar_step.video_chunk_sigmas.clone(), model._ar_step.action_sigmas.clone(),
                         torch.get_rng_state(), torch.rand(97), model))
    old, new = observed
    torch.testing.assert_close(old[3], new[3], atol=0, rtol=0)
    torch.testing.assert_close(old[4], new[4], atol=0, rtol=0)
    plan = new[-1]._ar_step.prefix_low_noise_plan
    for a, b in zip(old[1:3], new[1:3]):
        torch.testing.assert_close(a[~plan.mask], b[~plan.mask], atol=0, rtol=0)
    torch.testing.assert_close(new[1][plan.mask], new[2][plan.mask], atol=0, rtol=0)
    assert old[-1]._ar_step.prefix_low_noise == [] and old[-1]._ar_step.prefix_low_noise_plan is None
    assert not plan.mask[2].any()
    assert [row["n_chunks"] for row in plan.metadata] == [17, 5, 1]
    from cosmos3_joint_video_hand_pose.src.ar_v02_model import EgoVerseARV02Model
    control = official_model(layouts, False, cls=EgoVerseARV02Model)
    torch.manual_seed(153)
    repeated = draw(control, 37)
    for a, b in zip(old[0], repeated):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(old[3], torch.get_rng_state(), atol=0, rtol=0)


def test_private_prefix_rng_reproduces_checkpoint_and_separates_iteration_rank():
    torch.manual_seed(42)
    state = torch.get_rng_state()
    first = sample_prefix_low_noise([17, 5, 1], seed=42, iteration=101, rank=3)
    torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
    torch.rand(891)
    replay = sample_prefix_low_noise([17, 5, 1], seed=42, iteration=101, rank=3)
    torch.testing.assert_close(first.mask, replay.mask, atol=0, rtol=0)
    torch.testing.assert_close(first.sigmas, replay.sigmas, atol=0, rtol=0)
    assert first.metadata == replay.metadata
    for iteration, rank in ((102, 3), (101, 4)):
        different = sample_prefix_low_noise([17, 5, 1], seed=42, iteration=iteration, rank=rank)
        assert not torch.equal(first.sigmas, different.sigmas)
    layouts = [JointChunkLayout(69, 1, 4), JointChunkLayout(18, 1, 4)]
    torch.manual_seed(1007)
    checkpoint_rng = torch.get_rng_state()
    uninterrupted = official_model(layouts, True)
    outputs = draw(uninterrupted, 102)
    tail_rng = torch.get_rng_state()
    torch.rand(333)
    restored = official_model(layouts, True)
    torch.set_rng_state(checkpoint_rng)
    replayed = draw(restored, 102)
    for a, b in zip(outputs, replayed):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert uninterrupted._ar_step.prefix_low_noise == restored._ar_step.prefix_low_noise
    torch.testing.assert_close(tail_rng, torch.get_rng_state(), atol=0, rtol=0)


def test_shared_prefix_routes_actual_noised_sigmas_and_official_timesteps_with_clean_us():
    layouts = [JointChunkLayout(69, 1, 4), JointChunkLayout(18, 1, 4)]
    model = official_model(layouts, True)
    data = data_fixture(layouts)
    vt, vs, _, _ = draw(model, 11)
    packed = pack_joint_sequence(layout=layouts, gen_data_clean=data, text_ids=[[3], [4, 5]],
        special_tokens=model.llm_special_tokens, timesteps=vt, latent_patch_size=2, condition_frames=[(), ()])
    noised = model._add_noise_to_input(data, packed, vs, iteration=11)
    plan = model._ar_step.prefix_low_noise_plan
    assert plan.mask.any()
    assert model._ar_step.noised_video_sigmas is noised.sigmas_vision
    assert model._ar_step.noised_action_sigmas is noised.sigmas_action
    for i, layout in enumerate(layouts):
        vr, vc, _ = layout.video_metadata()
        ar, ac, _ = layout.action_metadata()
        torch.testing.assert_close(noised.xt_tokens_vision[i][:, :, vr == CONDITION_VIDEO],
            data.x0_tokens_vision[i][:, :, vr == CONDITION_VIDEO], atol=0, rtol=0)
        torch.testing.assert_close(noised.xt_tokens_action[i][ar == STATE], data.x0_tokens_action[i][ar == STATE], atol=0, rtol=0)
        assert torch.count_nonzero(vs[i, :len(vr)][vr == CONDITION_VIDEO]) == 0
        assert torch.count_nonzero(vt[i, :len(vr)][vr == CONDITION_VIDEO]) == 0
        assert torch.count_nonzero(noised.sigmas_action[i][ar == STATE]) == 0
        assert torch.count_nonzero(model._ar_step.action_timesteps[i][ar == STATE]) == 0
        for chunk in range(1, len(layout.boundaries) + 1):
            arange, vrange = (ar == ACTION) & (ac == chunk), (vr == VIDEO) & (vc == chunk)
            v, a = vs[i, :len(vr)][vrange], noised.sigmas_action[i][arange].flatten()
            assert v.unique().numel() == a.unique().numel() == 1
            if plan.mask[i, chunk]:
                torch.testing.assert_close(v[0], a[0], atol=0, rtol=0)
                assert 0 <= float(v[0]) < .1
            torch.testing.assert_close(vt[i, :len(vr)][vrange], continuous_rf_timesteps(v, model.rectified_flow_video), atol=0, rtol=0)
            torch.testing.assert_close(model._ar_step.action_timesteps[i][arange], continuous_rf_timesteps(a, model.rectified_flow_action), atol=0, rtol=0)
        assert torch.count_nonzero(noised.xt_tokens_action[i][:, 57:]) == 0


@pytest.mark.parametrize("maximum", [0, -.1, float("nan"), float("inf"), 1.1])
def test_rejects_invalid_history_maximum(maximum):
    with pytest.raises(ValueError, match="sigma_hist_max"):
        sample_prefix_low_noise([17], seed=42, iteration=1, rank=0, sigma_hist_max=maximum)


def sigma_summary(values):
    values = values.flatten().float()
    edges = torch.tensor([0, .02, .05, .1, .2, .3, .5, .7, .9, 1.0])
    quantiles = torch.tensor([0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.0])
    return dict(count=values.numel(), mean=float(values.mean()),
                quantiles={str(float(q)): float(x) for q, x in zip(quantiles, torch.quantile(values, quantiles))},
                bins=dict(edges=edges.tolist(), counts=torch.histogram(values, edges).hist.long().tolist()),
                fractions_below={str(threshold): float((values < threshold).float().mean()) for threshold in (.02, .05, .1)})


def test_10000_official_rf_draws_histogram_and_true_history_denominators():
    from cosmos3_joint_video_hand_pose.src import ar_v03_config
    from cosmos3_joint_video_hand_pose.src.config import COSMOS_REPO_ROOT
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    config = load_experiment_from_toml(COSMOS_REPO_ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_3.toml", [])
    rf = config.model.config.rectified_flow_training_config
    assert rf.train_time_video_distribution == "waver"
    assert rf.train_time_action_distribution == "logitnormal"
    assert rf.shift["480"] == 5 and rf.shift_action == 5
    count, chunks = 10000, 17
    layout = JointChunkLayout(1 + 4 * chunks, 1, 4)
    model = official_model([layout] * count, True)
    torch.manual_seed(42)
    _, routed_video, _, _ = draw(model, 12)
    plan = model._ar_step.prefix_low_noise_plan
    lengths = torch.tensor([row["prefix_length"] for row in plan.metadata])
    histogram = torch.bincount(lengths, minlength=chunks + 1)[1:]
    assert int(histogram.sum()) == count and (histogram > 0).all()
    assert (histogram.float() - count / chunks).abs().max() < count * .02
    assert plan.sigmas[plan.mask].max() < .1 and torch.all(plan.sigmas[plan.mask] >= 0)
    assert abs(float(plan.sigmas[plan.mask].mean()) - .05) < .001
    torch.testing.assert_close(model._ar_step.video_chunk_sigmas[plan.mask], model._ar_step.action_sigmas[plan.mask], atol=0, rtol=0)
    roles, video_chunks, _ = layout.video_metadata()
    assert torch.count_nonzero(routed_video[:, roles == CONDITION_VIDEO]) == 0
    for chunk in range(1, chunks + 1):
        rows = (roles == VIDEO) & (video_chunks == chunk)
        torch.testing.assert_close(routed_video[:, rows], model._ar_step.video_chunk_sigmas[:, chunk:chunk + 1].expand(-1, int(rows.sum())), atol=0, rtol=0)
    pair_weights = torch.minimum(torch.arange(chunks - 1, 0, -1), torch.tensor(15))
    previous_fraction = float(plan.mask[:, 1:-1].float().mean())
    pair_fraction = float((plan.mask[:, 1:-1] * pair_weights).sum() / (count * pair_weights.sum()))
    expected_pairs = float((torch.arange(chunks - 1, 0, -1) / chunks * pair_weights).sum() / pair_weights.sum())
    assert abs(previous_fraction - .5) < .015
    assert abs(pair_fraction - expected_pairs) < .015
    report = dict(schema="ar_v03_uniform_prefix_sigma_histogram_v1", n_samples=count, n_chunks=chunks,
        history_chunks=15, sigma_hist_max=.1, seed=42, iteration=12, rank=0,
        prefix_formula="L~UniformInt(1,N); k<L shared VA sigma~Uniform[0,.1); k>=L original RF draw",
        official_rf=dict(video_distribution="waver", action_distribution="logitnormal", shift=5,
                         configured_resolution_shifts=dict(rf.shift), selected_resolution="480"),
        prefix_length_counts=histogram.tolist(),
        history_denominators=dict(immediate_previous_history="Every query block 2..17 contributes its immediately previous block once; blocks 1..16, excludes last block",
                                  attention_pairs_appendix="All H15 key-query pairs; early keys counted repeatedly"),
        theoretical_prefix_fractions=dict(immediate_previous_history=.5, attention_pairs_appendix=expected_pairs),
        observed_prefix_fractions=dict(immediate_previous_history=previous_fraction, attention_pairs_appendix=pair_fraction),
        modalities={})
    for name, matrix in (("video", model._ar_step.video_chunk_sigmas), ("action", model._ar_step.action_sigmas)):
        history = matrix[:, 1:-1]
        report["modalities"][name] = dict(
            prefix_conditional=sigma_summary(matrix[plan.mask]),
            per_chunk=[dict(chunk=chunk, prefix_fraction=float(plan.mask[:, chunk].float().mean()), all_draws=sigma_summary(matrix[:, chunk])) for chunk in range(1, chunks + 1)],
            immediate_previous_history=sigma_summary(history),
            attention_pairs_appendix=sigma_summary(torch.repeat_interleave(history, pair_weights, dim=1)))
        assert abs(report["modalities"][name]["immediate_previous_history"]["fractions_below"]["0.1"] - .5) < .02
    destination = os.environ.get("AR_V03_SIGMA_REPORT_PATH")
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"prefix_length_counts": histogram.tolist(), "prefix_fractions": report["observed_prefix_fractions"],
        "below_0.1": {name: {scope: report["modalities"][name][scope]["fractions_below"]["0.1"] for scope in
          ("prefix_conditional", "immediate_previous_history", "attention_pairs_appendix")} for name in report["modalities"]}}), flush=True)
