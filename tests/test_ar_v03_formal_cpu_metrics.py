"""CPU-only tests for formal aggregation; no Nano weights, sampler, or GPU jobs."""
import copy
import json
import math
from pathlib import Path
import sys

import pytest

from cosmos3_joint_video_hand_pose.scripts import ar_v03_formal_cpu_metrics as metrics
from cosmos3_joint_video_hand_pose.scripts import ar_v03_sigma_scan as scan


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def formal(tmp_path, monkeypatch):
    root = tmp_path / "scan"
    plan = scan.prepare(root, "formal")
    run_root = tmp_path / "formal_training"
    binding = dict(checkpoint=str(run_root / "job/checkpoints/iter_000000500/model"),
                   prefix_low_noise_enabled=True, sigma_hist_max=0.1, contract_matches_snapshot=True)
    monkeypatch.setattr(scan, "load_binding", lambda _: binding)
    for task in plan["tasks"]:
        validation = [
            dict(archive=archive, sha256="validated", history=history,
                 sigma_small=task["sigma_small"], chunks=17,
                 parent_manifest_sha256=task["parent_manifest_sha256"],
                 fragment_manifest_sha256=task["fragment_manifest_sha256"])
            for history, archive in zip(task["histories"], task["expected_archives"])
        ]
        dump(root / "states" / (task["job_id"] + ".json"),
             dict(job_id=task["job_id"], status="success", validation=validation))
    return root, plan, run_root, binding


def test_formal_inventory_and_pilot_are_exact(formal):
    root, plan, run_root, _ = formal
    _, _, _, jobs = metrics.load_jobs(root, step=500, run_root=run_root)
    assert len(jobs) == 96
    assert len({j["archive"] for j in jobs}) == 96
    assert sum(j["task"]["split"] == "train" for j in jobs) == 48
    _, _, _, pilot = metrics.load_jobs(root, step=500, run_root=run_root, pilot=True)
    assert [(j["task"]["split"], j["history"]) for j in pilot] == [
        ("heldout", "gt"), ("heldout", "generated"), ("train", "gt")]
    assert all(j["task"]["sigma_small"] == 0.02 and j["task"]["parent_window_index"] == 0 for j in pilot)


@pytest.mark.parametrize("step", [20, 1500, 2500])
def test_short_and_unrequested_milestones_are_rejected(formal, step):
    root, _, run_root, _ = formal
    with pytest.raises(ValueError, match="formal step"):
        metrics.load_jobs(root, step=step, run_root=run_root)


@pytest.mark.parametrize("change", ["wrong_step", "other_run", "duplicate_task", "pending", "archive_hash_binding"])
def test_incorrect_binding_or_scan_inventory_is_rejected(formal, change):
    root, plan, run_root, binding = formal
    if change == "wrong_step":
        binding["checkpoint"] = str(run_root / "job/checkpoints/iter_000001000/model")
    elif change == "other_run":
        run_root = root / "different_run"
    elif change == "duplicate_task":
        plan["tasks"][-1] = copy.deepcopy(plan["tasks"][0])
        dump(root / "plan.json", plan)
    else:
        task = plan["tasks"][0]
        state_path = root / "states" / (task["job_id"] + ".json")
        state = metrics.read_json(state_path)
        if change == "pending":
            state["status"] = "pending"
        else:
            state["validation"][0]["fragment_manifest_sha256"] = "other_fragment"
        dump(state_path, state)
    with pytest.raises(ValueError):
        metrics.load_jobs(root, step=500, run_root=run_root)


def row(chunk=1, value=10.0, actions=32, frames=16, mse=100.0):
    return dict(chunk=chunk, source_start=(chunk - 1) * 32, source_stop=chunk * 32,
                action_count=actions, video_sampled_future_frames=frames,
                video_mse_uint8=mse, **{field: value for field in metrics.FIELDS})


def test_endpoints_shape_and_psnr_use_distinct_correct_weights():
    result = metrics.action_summary([row(value=10, actions=1, frames=1, mse=100),
                                     row(value=20, actions=3, frames=3, mse=900)])
    assert result["local_left_wrist_end_mm"] == 15
    assert result["local_right_wrist_end_degrees"] == 15
    assert result["local_right_wrist_local_shape_mpjpe_mm"] == 17.5
    assert result["video_mse_uint8"] == 700
    assert result["video_psnr_db"] == 10 * math.log10(255**2 / 700)
    assert result["video_psnr_db"] != (10 * math.log10(255**2 / 100) + 3 * 10 * math.log10(255**2 / 900)) / 4
    assert result["action_frames"] == 4 and result["future_rgb_frames"] == 4


def test_zero_rgb_error_has_explicit_exact_match_psnr_and_invalid_metrics_fail():
    assert metrics.action_summary([row(mse=0)])["video_psnr_db"] is None
    for rows in ([], [row(value=float("nan"))], [row(actions=0)], [row(mse=-1)]):
        with pytest.raises(ValueError):
            metrics.action_summary(rows)


def synthetic_sample(split, history, sigma, index):
    identity = f"{split}_{index}"
    flow = [dict(chunk=k, pixels=320 * 180, pairs=1, pred_magnitude_sum=float(index + 1),
                 gt_magnitude_sum=1.0, dot_sum=1.0, pred_squared_sum=float(index + 1), gt_squared_sum=1.0)
            for k in range(1, 18)]
    shifts = [dict(chunk=k, curve=[dict(gt_shift_frames=shift, squared_error_sum=float(abs(shift)),
                                       values=10) for shift in (-4, -2, 0, 2, 4)])
              for k in range(1, 18)]
    video = dict(sample_id=identity, source_offset=index, checkpoint="/synthetic/iter/model",
                 history=history, history_video_sigma=sigma, seed=42, steps=30, source_fps=60,
                 input=f"/synthetic/{split}/{sigma}/{history}/{index}.npz", archive_sha256="synthetic",
                 flow=dict(chunks=flow), shift=dict(chunks=shifts))
    action = dict(metadata=dict(sample_id=identity, source_offset=index),
                  metrics=dict(chunks=[row(k, value=float(index + 1)) for k in range(1, 18)]))
    return dict(split=split, window_index=index, sigma_small=sigma, history=history,
                action=action, endpoint_video=video)


@pytest.fixture
def samples():
    return [synthetic_sample(split, history, sigma, index) for sigma in scan.SIGMAS
            for split, history, count in (("heldout", "gt", 8), ("heldout", "generated", 8), ("train", "gt", 16))
            for index in range(count)]


@pytest.mark.parametrize("step", metrics.MILESTONES)
def test_nine_groups_use_real_official_pooling_and_all_seventeen_curves(samples, step):
    records = metrics.make_records(list(reversed(samples)), step)
    assert len(records) == 9 and len({r["key"] for r in records}) == 9
    for record in records:
        count = 8 if record["split"] == "heldout" else 16
        assert record["step"] == step and record["label"] == f"V0.3 uniform-prefix step{step}"
        assert [r["chunk"] for r in record["chunks"]] == list(range(1, 18))
        assert record["all"]["chunks"] == count * 17
        assert record["chunk17plus"]["chunks"] == count
        assert record["video"]["flow"]["pairs"] == count * 17
        assert record["video"]["flow"]["pixels"] == count * 17 * 320 * 180
        assert record["video"]["flow"]["direction_cosine"] == pytest.approx(1 / math.sqrt((count + 1) / 2))
        assert record["video"]["flow"]["direction_cosine"] != pytest.approx(
            sum(1 / math.sqrt(i + 1) for i in range(count)) / count)
        assert record["video"]["shift"]["best_gt_shift_frames"] == 0


def test_group_missing_or_duplicated_window_cannot_be_hidden_by_total_count(samples):
    broken = copy.deepcopy(samples)
    broken[1] = copy.deepcopy(broken[0])
    with pytest.raises(ValueError, match="exactly once"):
        metrics.make_records(broken, 500)
    with pytest.raises(ValueError, match="96 archives"):
        metrics.make_records(samples[:-1], 500)


def test_all_five_real_baseline_identities_and_pooled_support_match(formal):
    _, plan, _, _ = formal
    records = metrics.load_baselines(metrics.BASELINE, plan)
    assert [r["key"] for r in records] == list(metrics.BASELINE_KEYS)
    assert records[-1]["split"] == "train" and records[-1]["windows"] == 16


@pytest.mark.parametrize("change", ["missing", "duplicate", "identity", "pooled_support", "definition"])
def test_baseline_tampering_is_rejected(formal, tmp_path, change):
    _, plan, _, _ = formal
    report = metrics.read_json(metrics.BASELINE)
    target = next(r for r in report["records"] if r["key"] == "lr1000_train16_gt")
    if change == "missing":
        report["records"].remove(target)
    elif change == "duplicate":
        report["records"].append(copy.deepcopy(target))
    elif change == "identity":
        target["samples"][0]["source_offset"] += 1
    elif change == "pooled_support":
        target["video"]["flow"]["pairs"] -= 1
    else:
        report["flow_geometry"] = [640, 360]
    path = tmp_path / "edited_baseline.json"
    dump(path, report)
    with pytest.raises(ValueError):
        metrics.load_baselines(path, plan)


def complete_sample(job):
    binding = job["binding"]
    binding.update(action_representation="fixed_camera_wrist_local_delta_latent_v1",
                   frozen_artifacts={key: dict(sha256=key) for key in
                                     ("state_normalizer", "future_normalizer", "right_codec", "left_codec")})
    task = job["task"]
    meta = dict(model_version="ar_v0.3.0", schema="ar_v02_rollout_v1", layout_version="joint_chunk_cond_v1",
                history=job["history"], checkpoint=binding["checkpoint"], sample_id=task["window"]["sample_id"],
                source_offset=task["window"]["start"], seed=42, steps=30, chunk_size=4, num_frames=69,
                selected_windows=1, frozen_windows=1, sigma_small=0.02, history_video_sigma=0.02,
                history_action_sigma=0.02, video_shift=5.0, action_shift=5.0, video_guidance=1.0,
                action_guidance=1.0, action_representation=binding["action_representation"],
                eval_windows_sha256=task["fragment_manifest_sha256"], raw_gt_disabled_diagnostic=False,
                state_normalizer=dict(sha256="state_normalizer"),
                future_normalizer=dict(sha256="future_normalizer"),
                hand_codecs=dict(right=dict(sha256="right_codec"), left=dict(sha256="left_codec")))
    action = dict(metadata=meta, hand_metric_scope="all_finite_raw_coordinates_no_visibility_mask",
                  metrics=dict(chunks=[row(k) for k in range(1, 18)]))
    video = dict(input=str(Path(job["archive"]).resolve()), sample_id=meta["sample_id"],
                 source_offset=meta["source_offset"], checkpoint=meta["checkpoint"], history=job["history"],
                 history_video_sigma=0.02, seed=42, steps=30, archive_sha256=job["validated_sha256"],
                 flow=dict(chunks=[dict(chunk=k, pairs=1, pixels=320 * 180) for k in range(1, 18)]))
    return dict(split=task["split"], window_index=task["parent_window_index"],
                sigma_small=0.02, history=job["history"], action=action, endpoint_video=video)


@pytest.mark.parametrize("change", ["decoded_gt", "sigma", "codec", "fragment", "raw_gt", "chunks", "hash", "flow_support"])
def test_archive_metric_binding_raw_gt_and_geometry_are_strict(formal, change):
    root, _, run_root, _ = formal
    _, _, _, jobs = metrics.load_jobs(root, step=500, run_root=run_root, pilot=True)
    sample = complete_sample(jobs[0])
    metrics.validate_sample(sample["action"], sample["endpoint_video"], jobs[0])
    meta, video = sample["action"]["metadata"], sample["endpoint_video"]
    if change == "decoded_gt":
        sample["action"]["hand_metric_scope"] = "all_finite_decoded_coordinates_no_visibility_mask"
    elif change == "sigma":
        meta["history_action_sigma"] = 0.1
    elif change == "codec":
        meta["hand_codecs"]["left"]["sha256"] = "changed"
    elif change == "fragment":
        meta["eval_windows_sha256"] = "other"
    elif change == "raw_gt":
        meta["raw_gt_disabled_diagnostic"] = True
    elif change == "chunks":
        sample["action"]["metrics"]["chunks"].pop()
    elif change == "hash":
        video["archive_sha256"] = "changed"
    else:
        video["flow"]["chunks"][0]["pixels"] -= 1
    with pytest.raises(ValueError):
        metrics.validate_sample(sample["action"], video, jobs[0])


def test_pilot_cache_requires_same_tool_source_three_archives_and_current_hash(formal, tmp_path, monkeypatch):
    root, _, run_root, _ = formal
    _, _, _, jobs = metrics.load_jobs(root, step=500, run_root=run_root, pilot=True)
    samples = [complete_sample(j) for j in jobs]
    source = dict(plan_sha256="plan", binding_sha256="binding", tool_sha256="tool", step=500)
    pilot = dict(schema=metrics.SCHEMA, mode="pilot", source=source, exact_equal=True,
                 serial_workers=1, parallel_workers=3, samples=samples)
    receipt = tmp_path / "pilot.json"
    dump(receipt, pilot)
    monkeypatch.setattr(metrics, "sha256", lambda _: "validated")
    assert metrics.cached_pilot(receipt, source, jobs) == samples
    for changed in (
        dict(source=dict(source, tool_sha256="other")),
        dict(exact_equal=False), dict(parallel_workers=2), dict(samples=samples[:2]),
    ):
        dump(receipt, dict(pilot, **changed))
        with pytest.raises(ValueError):
            metrics.cached_pilot(receipt, source, jobs)
    dump(receipt, pilot)
    monkeypatch.setattr(metrics, "sha256", lambda _: "changed")
    with pytest.raises(ValueError, match="NPZ changed"):
        metrics.cached_pilot(receipt, source, jobs)


def test_pilot_cli_executes_workers_one_and_three_and_saves_exact_cache(tmp_path, monkeypatch):
    output = tmp_path / "pilot.json"
    jobs, samples, calls = [object(), object(), object()], [dict(result=i) for i in range(3)], []
    monkeypatch.setattr(sys, "argv", ["metrics", "pilot", "--scan-root", "/scan", "--run-root", "/run",
                                    "--step", "500", "--output", str(output)])
    monkeypatch.setattr(metrics, "resource_snapshot", lambda workers: dict(workers=workers))
    monkeypatch.setattr(metrics, "load_jobs", lambda *a, **kw: (Path("/scan"), {}, {}, jobs))
    source = dict(plan_sha256="plan", binding_sha256="binding")
    monkeypatch.setattr(metrics, "source_record", lambda *args: source)
    def calculate(input_jobs, workers):
        assert input_jobs is jobs
        calls.append(workers)
        return copy.deepcopy(samples), float(workers)
    monkeypatch.setattr(metrics, "calculate", calculate)
    metrics.main()
    report = metrics.read_json(output)
    assert calls == [1, 3] and report["exact_equal"] is True
    assert report["samples"] == samples and report["source"] == source and report["step"] == 500
    assert report["hardware"]["workers"] == 3
    assert metrics.os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert metrics.os.environ["OMP_NUM_THREADS"] == "1"


def test_aggregate_requires_pilot_before_resource_or_archive_work(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["metrics", "aggregate", "--scan-root", "/scan", "--run-root", "/run",
                                    "--step", "500", "--output", str(tmp_path / "report.json")])
    with pytest.raises(ValueError, match="pilot-preflight"):
        metrics.main()


def test_aggregate_reuses_three_cache_results_and_labels_controller_schema(tmp_path, monkeypatch, samples):
    output = tmp_path / "aggregate.json"
    cached = [x for x in samples if x["sigma_small"] == 0.02 and x["window_index"] == 0]
    jobs = [dict(archive=x["endpoint_video"]["input"]) for x in samples]
    pending_samples = [x for x in samples if x not in cached]
    source = dict(plan_sha256="plan", binding_sha256="binding")
    baselines = [dict(key=key) for key in metrics.BASELINE_KEYS]
    monkeypatch.setattr(sys, "argv", ["metrics", "aggregate", "--scan-root", "/scan", "--run-root", "/run",
                                    "--step", "2000", "--output", str(output), "--pilot-preflight", "/pilot"])
    monkeypatch.setattr(metrics, "resource_snapshot", lambda workers: dict(workers=workers))
    monkeypatch.setattr(metrics, "load_jobs", lambda *a, **kw: (Path("/scan"), dict(frozen_parents={}), {}, jobs))
    monkeypatch.setattr(metrics, "source_record", lambda *args: source)
    monkeypatch.setattr(metrics, "load_baselines", lambda *args: baselines)
    monkeypatch.setattr(metrics, "cached_pilot", lambda *args: cached)
    monkeypatch.setattr(metrics, "sha256", lambda _: "test_hash")
    monkeypatch.setattr(metrics, "baseline_provenance", lambda: dict(note="historical evidence"))
    def calculate(input_jobs, workers):
        assert len(input_jobs) == 93 and workers == 24
        assert not {x["endpoint_video"]["input"] for x in cached} & {x["archive"] for x in input_jobs}
        return copy.deepcopy(pending_samples), 1.0
    monkeypatch.setattr(metrics, "calculate", calculate)
    metrics.main()
    report = metrics.read_json(output)
    assert report["step"] == 2000 and report["archive_count"] == 96 and report["groups"] == 9
    assert report["recomputed_archives"] == 93 and report["pilot_receipt"]["cached_archives"] == 3
    assert report["source"] == source and report["baselines"] == baselines
    assert len(report["records"]) == 9 and len(report["samples"]) == 96


def test_historical_parameter_evidence_is_explicit_and_frozen():
    result = metrics.baseline_provenance()
    assert result["applies_to_keys"] == ["baseline_gt", "baseline_generated"]
    assert result["sha256"] == metrics.BASELINE_AUDIT_SHA256
    assert "do not explicitly bind" in result["limitation"]
    assert result["absent_archive_metadata_fields"] == ["video_shift", "video_guidance", "history_video_sigma"]


def test_parser_and_exclusive_write_preserve_history(tmp_path):
    args = metrics.parser().parse_args(["aggregate", "--scan-root", "/scan", "--run-root", "/run",
                                       "--step", "3000", "--output", "/new", "--pilot-preflight", "/pilot"])
    assert args.workers == 24 and args.step == 3000
    path = tmp_path / "report.json"
    metrics.write_new(path, dict(value="original"))
    with pytest.raises(FileExistsError):
        metrics.write_new(path, dict(value="replacement"))
    assert metrics.read_json(path) == dict(value="original")
