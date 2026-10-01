"""CPU-only followup tests; no formal checkpoint is bound or GPU launched."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ar_v03_eval_followup.py"
spec = importlib.util.spec_from_file_location("ar_v03_followup_test", SCRIPT)
followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    scan = followup.dispatcher()
    monkeypatch.setattr(followup, "dispatcher", lambda: scan)
    root, training = tmp_path / "eval", tmp_path / "training_output_not_created"
    config = followup.init(root, training, ["Tdebug4", "Tdebug2"])
    return scan, root, training, config


def mark_binding(scan, root, step):
    target = root / f"step{step:06d}"
    scan.write_new(target / "binding.json", {"test_fixture_only": True})
    return target


def completed_jobs(scan, target):
    _, plan = scan.load_plan(target)
    for task in plan["tasks"]:
        state = target / "states" / (task["job_id"] + ".json")
        scan.replace_json(state, dict(job_id=task["job_id"], status="success",
                                      expected_results=len(task["histories"]), validation=[]))
    return plan


def test_init_four_independent_unbound_plans_and_training_nodes_excluded(campaign):
    scan, root, training, config = campaign
    assert not training.exists()
    assert config["excluded_training_nodes"] == ["Tdebug4", "Tdebug2"]
    assert config["permitted_evaluation_nodes"] == ["Tdebug1", "Tdebug3", "Tdebug5", "Tdebug6"]
    assert [x["step"] for x in config["campaigns"]] == [500, 1000, 2000, 3000]
    for item in config["campaigns"]:
        target, plan = scan.load_plan(item["scan_root"])
        assert plan["jobs"] == 72 and plan["expected_results"] == 96
        assert not (target / "binding.json").exists() and not (target / "eval").exists()


def test_inspect_is_readonly_and_waits_for_nonexistent_checkpoint(campaign):
    scan, root, training, config = campaign
    before = {str(x): scan.digest(x) for x in root.rglob("*.json")}
    result = followup.inspect(root)
    assert result["next_campaign"] is None
    assert all(x["phase"] == "waiting_checkpoint" for x in result["campaigns"])
    assert {str(x): scan.digest(x) for x in root.rglob("*.json")} == before
    assert not result["automatic_GPU_or_SSH_actions_performed"]


def test_snapshot_discovery_uses_actual_official_job_dir_and_rejects_ambiguity(tmp_path):
    training = tmp_path / "formal_output"
    actual = training / "project/group/name"
    actual.mkdir(parents=True)
    (actual / "config.yaml").write_text("model: fixture")
    path, reason = followup.discover_run_dir(training)
    assert path == actual.resolve() and reason is None
    another = training / "other"
    another.mkdir()
    (another / "config.yaml").write_text("fixture")
    with pytest.raises(ValueError, match="multiple"):
        followup.discover_run_dir(training)


def create_marker_fixture(tmp_path, latest="iter_000001000", metadata=True):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text("CPU fixture only")
    (run / "checkpoints").mkdir()
    (run / "checkpoints/latest_checkpoint.txt").write_text(latest + "\n")
    directory = run / "checkpoints/iter_000000500"
    if metadata:
        for name in ("model", "trainer", "optim", "scheduler"):
            (directory / name).mkdir(parents=True)
            (directory / name / ".metadata").write_bytes(b"mock metadata, never DCP-loaded")
    return run


@pytest.mark.parametrize("latest", ["iter_000000499", "iter_0000000", "partial_write"])
def test_readiness_requires_official_marker_at_or_beyond_exact_step(tmp_path, latest):
    run = create_marker_fixture(tmp_path, latest)
    def forbidden(path):
        raise AssertionError("storage must not be read before complete marker")
    assert not followup.checkpoint_ready(run, 500, storage_validator=forbidden)["ready"]


def test_readiness_requires_all_four_metadata_groups_not_just_model(tmp_path):
    run = create_marker_fixture(tmp_path)
    (run / "checkpoints/iter_000000500/trainer/.metadata").unlink()
    result = followup.checkpoint_ready(run, 500, storage_validator=lambda _: {"unused": True})
    assert not result["ready"] and "trainer/.metadata" in result["missing"][0]


def test_completed_marker_then_checks_all_storage_bounds_without_guessed_shard_count(tmp_path):
    run = create_marker_fixture(tmp_path)
    read = []
    def validation(path):
        read.append(Path(path).name)
        return {"metadata_keys": 1, "storage_files": ["fixture"]}
    result = followup.checkpoint_ready(run, 500, storage_validator=validation)
    assert result["ready"] and read == ["model", "trainer", "optim", "scheduler"]
    assert result["checkpoint"].endswith("/iter_000000500/model")


def test_storage_bounds_reads_native_small_CPU_DCP_and_rejects_truncated_shard(tmp_path):
    import torch
    import torch.distributed.checkpoint as dcp
    root = tmp_path / "tiny_dcp_fixture"
    dcp.save({"CPU_fixture_tensor": torch.arange(8, dtype=torch.float32)}, checkpoint_id=root)
    report = followup.storage_bounds(root)
    assert report["metadata_keys"] == 1 and len(report["storage_files"]) >= 1
    shard = Path(report["storage_files"][0]["path"])
    shard.write_bytes(b"")
    with pytest.raises(ValueError, match="storage incomplete"):
        followup.storage_bounds(root)


@pytest.mark.parametrize("node", ["Tdebug4", "Tdebug2"])
def test_dispatch_rejects_formal_training_nodes_before_any_binding_or_process(campaign, node):
    scan, root, training, config = campaign
    with pytest.raises(ValueError, match="excluded"):
        followup.dispatch_command(root, 500, "pilot", node, [0, 1, 2])


def test_generated_MCP_command_uses_only_requested_nontraining_node_and_no_launch(campaign, monkeypatch):
    scan, root, training, config = campaign
    target = mark_binding(scan, root, 500)
    monkeypatch.setattr(scan, "load_binding", lambda _: {})
    monkeypatch.setattr(followup.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    result = followup.dispatch_command(root, 500, "pilot", "Tdebug3", [0, 1, 2])
    assert result["hostAlias"] == "Tdebug3" and result["requires_fresh_resource_check"]
    assert "--gpus 0,1,2" in result["command"] and "--stage pilot" in result["command"]
    assert not result["GPU_processes_started_by_this_controller"]
    with pytest.raises(ValueError, match="pilot"):
        followup.dispatch_command(root, 500, "remaining", "Tdebug3", [3])


def test_active_milestone_suppresses_new_GPU_campaign(campaign, monkeypatch):
    scan, root, training, config = campaign
    target = mark_binding(scan, root, 500)
    monkeypatch.setattr(scan, "load_binding", lambda _: {})
    _, plan = scan.load_plan(target)
    task = plan["tasks"][0]
    scan.replace_json(target / "states" / (task["job_id"] + ".json"),
                      dict(job_id=task["job_id"], status="running", expected_results=2))
    next_target = mark_binding(scan, root, 1000)
    with pytest.raises(ValueError, match="another milestone"):
        followup.dispatch_command(root, 1000, "pilot", "Tdebug5", [0])
    result = followup.inspect(root)
    assert result["campaigns"][0]["phase"] == "GPU_running"
    assert result["next_campaign"] is None


def test_complete_scan_requests_CPU_only_stage_and_claim_suppresses_duplicate(campaign, monkeypatch):
    scan, root, training, config = campaign
    target = mark_binding(scan, root, 500)
    monkeypatch.setattr(scan, "load_binding", lambda _: {})
    completed_jobs(scan, target)
    result = followup.inspect(root)
    report = result["campaigns"][0]
    assert report["phase"] == "CPU_pilot_pending"
    assert report["next_action"]["choose_exactly_one_node"]
    commands = report["next_action"]["commands_by_node"]
    assert set(commands) == {"Tdebug1", "Tdebug3", "Tdebug5", "Tdebug6"}
    assert all("run-metrics" in value["command"] and "CUDA_VISIBLE_DEVICES=" in value["command"]
               for value in commands.values())
    claim = followup.metric_claim(target, "pilot")
    claim.mkdir(parents=True)
    scan.write_new(claim / "state.json", {"status": "running", "PID": 123})
    assert followup.inspect(root)["campaigns"][0]["phase"] == "CPU_metrics_running"


def test_failed_metric_claim_requires_attention_without_retry(campaign, monkeypatch):
    scan, root, training, config = campaign
    target = mark_binding(scan, root, 500)
    monkeypatch.setattr(scan, "load_binding", lambda _: {})
    completed_jobs(scan, target)
    claim = followup.metric_claim(target, "pilot")
    claim.mkdir(parents=True)
    scan.write_new(claim / "state.json", {"status": "failed", "exit_code": 5})
    result = followup.inspect(root)["campaigns"][0]
    assert result["phase"] == "attention_required" and result["metrics_error"]["exit_code"] == 5


def test_bind_ready_cannot_bind_missing_milestone(campaign, monkeypatch):
    scan, root, training, config = campaign
    monkeypatch.setattr(scan, "bind", lambda *a: pytest.fail("must not bind"))
    with pytest.raises(ValueError, match="not fully saved"):
        followup.bind_ready(root, 500)


def test_no_duplicate_dispatch_for_a_completed_pilot(campaign, monkeypatch):
    scan, root, training, config = campaign
    target = mark_binding(scan, root, 500)
    monkeypatch.setattr(scan, "load_binding", lambda _: {})
    _, plan = scan.load_plan(target)
    for task in plan["tasks"]:
        if task["stage"] == "pilot":
            scan.replace_json(target / "states" / (task["job_id"] + ".json"),
                              dict(job_id=task["job_id"], status="success", expected_results=2))
    with pytest.raises(ValueError, match="no pending"):
        followup.dispatch_command(root, 500, "pilot", "Tdebug3", [0])


def mock_metrics_ready(campaign, monkeypatch):
    scan, root, training, config = campaign
    target = mark_binding(scan, root, 500)
    monkeypatch.setattr(scan, "load_binding", lambda _: {})
    completed_jobs(scan, target)
    monkeypatch.setattr(scan, "cpu_resources", lambda: {
        "effective_cpus": 120, "memory": {"MemAvailable": 120 << 30}, "loadavg": [0, 0, 0],
    })
    return scan, root, target, training


def test_CPU_stage_claim_logs_exact_helper_args_and_blocks_duplicate(campaign, monkeypatch):
    scan, root, target, training = mock_metrics_ready(campaign, monkeypatch)
    launched = []
    def fake_process(arguments, **kwargs):
        launched.append((arguments, kwargs))
        output = Path(arguments[arguments.index("--output") + 1])
        def wait():
            output.write_text('{}\n')
            return 0
        return SimpleNamespace(pid=456, wait=wait)
    monkeypatch.setattr(followup.subprocess, "Popen", fake_process)
    result = followup.run_metrics(root, 500, "pilot", "Tdebug1", 24)
    assert result["status"] == "success" and result["exit_code"] == 0
    assert result["child_PID"] == 456 and not result["GPU_used"]
    arguments, kwargs = launched[0]
    assert arguments[:3] == [str(scan.PYTHON), str(followup.METRICS), "pilot"]
    assert arguments[arguments.index("--run-root") + 1] == str(training)
    assert arguments[arguments.index("--scan-root") + 1] == str(target)
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "" and kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert (target / "metrics/pilot_cpu.log").is_file()
    assert scan.read_json(followup.metric_claim(target, "pilot") / "state.json")["status"] == "success"
    with pytest.raises(FileExistsError, match="already exists"):
        followup.run_metrics(root, 500, "pilot", "Tdebug1", 24)
    assert len(launched) == 1


def test_CPU_subprocess_error_is_durable_and_never_automatically_retried(campaign, monkeypatch):
    scan, root, target, training = mock_metrics_ready(campaign, monkeypatch)
    def forbidden_process(*args, **kwargs):
        raise RuntimeError("fixture launch failure")
    monkeypatch.setattr(followup.subprocess, "Popen", forbidden_process)
    result = followup.run_metrics(root, 500, "pilot", "Tdebug3", 24)
    assert result["status"] == "failed" and "fixture launch failure" in result["error"]
    assert scan.read_json(followup.metric_claim(target, "pilot") / "state.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        followup.run_metrics(root, 500, "pilot", "Tdebug3", 24)
    assert followup.inspect(root)["campaigns"][0]["phase"] == "attention_required"


def test_CPU_aggregate_rejects_preflight_from_another_binding_before_claim(campaign, monkeypatch):
    scan, root, target, training = mock_metrics_ready(campaign, monkeypatch)
    scan.write_new(target / "metrics/pilot_cpu_consistency.json", {
        "exact_equal": True, "source": {"plan_sha256": scan.digest(target / "plan.json"),
                                         "binding_sha256": "another checkpoint"},
    })
    monkeypatch.setattr(followup.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="preflight"):
        followup.run_metrics(root, 500, "aggregate", "Tdebug1", 24)
    assert not followup.metric_claim(target, "aggregate").exists()


@pytest.mark.parametrize("wrapper_alive,child_alive,expected_phase", [
    (True, True, "running"), (False, False, "attention_required"),
    (False, True, "attention_required"),
])
def test_CPU_probe_detects_dead_wrapper_and_live_orphan_without_mutation(
        campaign, monkeypatch, wrapper_alive, child_alive, expected_phase):
    scan, root, training, config = campaign
    target = root / "step000500"
    claim = followup.metric_claim(target, "pilot")
    claim.mkdir(parents=True)
    state = dict(status="running", node_label="Tdebug3", hostname="fixture_node",
                 PID=111, child_PID=222, PID_start_ticks=1, child_PID_start_ticks=2)
    scan.write_new(claim / "state.json", state)
    before = scan.digest(claim / "state.json")
    monkeypatch.setattr(followup.socket, "gethostname", lambda: "fixture_node")
    identities = {111: dict(state="S", start_ticks=1) if wrapper_alive else None,
                  222: dict(state="S", start_ticks=2) if child_alive else None}
    monkeypatch.setattr(followup, "process_identity", lambda pid: identities.get(pid))
    result = followup.probe_process(root, 500, "pilot", "Tdebug3")
    assert result["phase"] == expected_phase and result["read_only"]
    assert result["live_processes"] == {"PID": wrapper_alive, "child_PID": child_alive}
    assert scan.digest(claim / "state.json") == before


def test_CPU_probe_rejects_wrong_node_and_reused_PID(campaign, monkeypatch):
    scan, root, training, config = campaign
    claim = followup.metric_claim(root / "step000500", "pilot")
    claim.mkdir(parents=True)
    scan.write_new(claim / "state.json", dict(status="running", node_label="Tdebug3",
                   hostname="fixture_node", PID=111, PID_start_ticks=1))
    monkeypatch.setattr(followup.socket, "gethostname", lambda: "another_node")
    with pytest.raises(ValueError, match="recorded node/hostname"):
        followup.probe_process(root, 500, "pilot", "Tdebug3")
    monkeypatch.setattr(followup.socket, "gethostname", lambda: "fixture_node")
    monkeypatch.setattr(followup, "process_identity", lambda pid: {"state": "S", "start_ticks": 999})
    result = followup.probe_process(root, 500, "pilot", "Tdebug3")
    assert result["phase"] == "attention_required" and not result["live_processes"]["PID"]


def test_native_process_identity_reports_this_CPU_process_only():
    import os
    identity = followup.process_identity(os.getpid())
    assert identity["start_ticks"] > 0 and identity["state"] != "Z"
    assert followup.process_identity(None) is None
    assert followup.process_identity(999999999) is None
