"""CPU-only scan planning/worker tests; no GPU, model or sampler process starts."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos3_joint_video_hand_pose.scripts import ar_v03_sigma_scan as scan


@pytest.mark.parametrize("scope,jobs,results,fragments", [
    ("short", 24, 48, 8), ("formal", 72, 96, 24),
])
def test_preparation_preserves_frozen_parent_identity(tmp_path, scope, jobs, results, fragments):
    root = tmp_path / "scan"
    plan = scan.prepare(root, scope)
    assert (plan["jobs"], plan["expected_results"]) == (jobs, results)
    assert len(list((root / "manifests").glob("*.json"))) == fragments
    assert not (root / "binding.json").exists()
    assert not (root / "eval").exists()
    assert len([x for x in plan["tasks"] if x["stage"] == "pilot"]) == 3
    _, validated = scan.load_plan(root)
    assert validated == plan
    for task in plan["tasks"]:
        parent = scan.read_json(task["parent_manifest"])
        assert task["window"] == parent[task["parent_window_index"]]
        assert task["window"]["seed"] == 42 and task["window"]["frames"] == 273
    for sigma in scan.SIGMAS:
        group = [x for x in plan["tasks"] if x["sigma_small"] == sigma]
        assert sum(len(x["histories"]) for x in group) == results // 3


def test_fragment_and_parent_position_tampering_are_rejected(tmp_path):
    root = tmp_path / "scan"
    plan = scan.prepare(root, "short")
    first = plan["tasks"][0]
    fragment = Path(first["fragment_manifest"])
    original = fragment.read_text()
    fragment.write_text("[]")
    with pytest.raises(ValueError, match="fragment changed"):
        scan.load_plan(root)
    fragment.write_text(original)
    plan["tasks"][0]["parent_window_index"] = 1
    scan.replace_json(root / "plan.json", plan)
    with pytest.raises(ValueError, match="parent position"):
        scan.load_plan(root)


def test_event_and_progress_share_timestamp_without_duplicate_keyword(tmp_path):
    root = tmp_path / "scan"
    plan = scan.prepare(root, "short")
    stream = io.StringIO()
    scan.event(stream, "test", **scan.progress(root, plan))
    result = json.loads(stream.getvalue())
    assert result["event"] == "test" and result["completed_results"] == 0
    assert result["total_results"] == 48


@pytest.mark.parametrize("change", [
    {"processes": [{"pid": "42"}]}, {"utilization_percent": 1}, {"memory_used_mib": 257},
])
def test_idle_gpu_check_rejects_occupied_card(change):
    record = dict(processes=[], utilization_percent=0, memory_used_mib=0)
    record.update(change)
    with pytest.raises(RuntimeError, match="occupied"):
        scan.require_idle({0: record}, [0])


def test_sampler_command_keeps_single_window_both_modes_and_exact_sigma(tmp_path):
    plan = scan.prepare(tmp_path / "scan", "short")
    for task in plan["tasks"]:
        command = scan.sampler_command(plan, {"checkpoint": "/frozen/checkpoint/model"}, task)
        assert command[command.index("--sigma-small") + 1] == str(task["sigma_small"])
        assert command[command.index("--eval-windows") + 1] == task["fragment_manifest"]
        assert command[command.index("--history") + 1:command.index("--chunk-size")] == ["gt", "generated"]
        assert command[command.index("--output") + 1] == task["output"]


def mocked_worker(monkeypatch, exit_code=0):
    monkeypatch.setattr(scan, "load_binding", lambda root: {"checkpoint": "/frozen/model"})
    monkeypatch.setattr(scan, "gpu_snapshot", lambda: {0: dict(
        processes=[], utilization_percent=0, memory_used_mib=0)})
    monkeypatch.setattr(scan, "validate_archives", lambda task, binding: [
        {"archive": x, "chunks": 17, "sigma_small": task["sigma_small"]} for x in task["expected_archives"]])
    class Process:
        def __init__(self, *args, **kwargs):
            self.returncode = exit_code
            self.pid = 999999
        def poll(self):
            return self.returncode
    monkeypatch.setattr(scan.subprocess, "Popen", Process)


def test_one_worker_runs_next_job_only_after_success_and_pilot_covers_all_sigmas(tmp_path, monkeypatch):
    root = tmp_path / "scan"
    plan = scan.prepare(root, "short")
    mocked_worker(monkeypatch)
    assert scan.worker(root, "pilot", 0, "mock_node", 30) == 0
    progress = scan.progress(root, plan)
    assert progress["completed_jobs"] == 3 and progress["completed_results"] == 6
    assert scan.pilot_complete(root, plan)
    assert scan.read_json(root / "pilot_validation.json")["results"] == 6
    assert not (root / "STOP").exists()
    states = [scan.read_json(root / "states" / (x["job_id"] + ".json")) for x in plan["tasks"]]
    assert sum(x["status"] == "pending" for x in states) == 21
    # Durable claims/success receipts suppress duplicate dispatch.
    assert scan.worker(root, "pilot", 0, "mock_node", 30) == 0
    assert scan.progress(root, plan)["completed_results"] == 6


def test_failed_job_stops_queue_and_records_exit_without_automatic_retry(tmp_path, monkeypatch):
    root = tmp_path / "scan"
    plan = scan.prepare(root, "short")
    mocked_worker(monkeypatch, exit_code=7)
    assert scan.worker(root, "pilot", 0, "mock_node", 30) == 1
    report = scan.progress(root, plan)
    assert len(report["failed_jobs"]) == 1 and report["completed_results"] == 0
    state = scan.read_json(root / "states" / (report["failed_jobs"][0] + ".json"))
    assert state["exit_code"] == 7 and state["status"] == "failed"
    assert (root / "STOP").is_file()
    assert not scan.pilot_complete(root, plan)
    assert scan.worker(root, "pilot", 0, "mock_node", 30) == 0
    assert len(scan.progress(root, plan)["failed_jobs"]) == 1


def test_remaining_stage_requires_accepted_pilot(tmp_path, monkeypatch):
    root = tmp_path / "scan"
    scan.prepare(root, "short")
    mocked_worker(monkeypatch)
    with pytest.raises(RuntimeError, match="successful"):
        scan.worker(root, "remaining", 0, "mock_node", 30)

def archive_fixture(tmp_path, monkeypatch):
    import sys
    root = tmp_path / "scan"
    plan = scan.prepare(root, "short")
    task = plan["tasks"][1]
    output = Path(task["output"])
    output.mkdir(parents=True)
    for path in task["expected_archives"]:
        Path(path).write_bytes(b"CPU archive validator fixture; no sampled data")
    scan.write_new(output / "run.json", {"archives": task["expected_archives"]})
    binding = dict(
        checkpoint="/frozen/checkpoint/model", action_representation="fixed_camera_wrist_local_delta_latent_v1",
        frozen_artifacts={name: {"sha256": "a" * 64} for name in
                          ("state_normalizer", "future_normalizer", "right_codec", "left_codec")},
    )
    meta = dict(
        model_version="ar_v0.3.0", schema="ar_v02_rollout_v1", layout_version="joint_chunk_cond_v1",
        sample_id=task["window"]["sample_id"], source_offset=task["window"]["start"],
        seed=42, chunk_size=4, num_frames=69, steps=30, sigma_small=.05,
        history_video_sigma=.05, history_action_sigma=.05,
        eval_windows_sha256=task["fragment_manifest_sha256"], selected_windows=1, frozen_windows=1,
        checkpoint=binding["checkpoint"], video_shift=5.0, action_shift=5.0,
        video_guidance=1.0, action_guidance=1.0, action_representation=binding["action_representation"],
        state_normalizer={"sha256": "a" * 64}, future_normalizer={"sha256": "a" * 64},
        hand_codecs={side: {"sha256": "a" * 64} for side in ("right", "left")},
        chunk_reports=[dict(
            chunk=i, denoise_steps=30, forward_calls=32, noisy_calls=30, condition_prefill_calls=1,
            clean_refresh_calls=0, noisy_refresh_calls=1, cache_mode="persistent",
            includes_reference_checks=False, action_count=32, end_to_end_seconds=1.0,
            peak_memory_bytes=123,
        ) for i in range(1, 18)],
        condition_reports=[dict(chunk=i, episode_boundary_source_index=task["window"]["start"] + 32 * (i - 1))
                           for i in range(1, 18)],
    )
    arrays = {name: None for name in ("raw_gt_keypoints", "raw_gt_camera_poses", "raw_gt_source_indexes",
                                     "gt_rgb", "generated_rgb", "generated_offsets", "intrinsics")}
    def native_validator_fixture(path):
        item = dict(meta, history="generated" if str(path).endswith("_generated.npz") else "gt")
        return SimpleNamespace(boundaries=list(range(17))), item, arrays
    monkeypatch.setitem(sys.modules, "cosmos3_joint_video_hand_pose.src.ar_v02_eval",
                        SimpleNamespace(load_rollout=native_validator_fixture))
    return task, binding, meta, arrays


def test_archive_gate_accepts_exact_two_mode_three_sigma_identity(tmp_path, monkeypatch):
    task, binding, meta, arrays = archive_fixture(tmp_path, monkeypatch)
    reports = scan.validate_archives(task, binding)
    assert len(reports) == 2
    assert {x["history"] for x in reports} == {"gt", "generated"}
    assert all(x["sigma_small"] == .05 and x["chunks"] == 17 for x in reports)
    assert all(x["sampler_seconds"] == 17 for x in reports)


@pytest.mark.parametrize("invalid", [
    "sigma_small", "history_action_sigma", "fragment_hash", "chunk_count",
    "denoise_steps", "condition_boundary", "raw_gt_payload",
])
def test_archive_gate_rejects_sigma_partial_chunk_or_window_mismatch(tmp_path, monkeypatch, invalid):
    task, binding, meta, arrays = archive_fixture(tmp_path, monkeypatch)
    if invalid in ("sigma_small", "history_action_sigma"):
        meta[invalid] = .02
    elif invalid == "fragment_hash":
        meta["eval_windows_sha256"] = "b" * 64
    elif invalid == "chunk_count":
        meta["chunk_reports"] = meta["chunk_reports"][:-1]
    elif invalid == "denoise_steps":
        meta["chunk_reports"][3]["denoise_steps"] = 29
    elif invalid == "condition_boundary":
        meta["condition_reports"][0]["episode_boundary_source_index"] += 1
    else:
        arrays.pop("raw_gt_keypoints")
    with pytest.raises(ValueError):
        scan.validate_archives(task, binding)
