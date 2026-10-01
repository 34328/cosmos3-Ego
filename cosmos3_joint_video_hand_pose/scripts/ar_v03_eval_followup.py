#!/usr/bin/env python3
"""CPU-only milestone readiness and commands for the current-thread heartbeat.

This controller never uses SSH and never launches GPU processes. A heartbeat
uses MCP to check real resources on allowed nodes, then executes the generated
local-node scan command there. Durable claims/progress belong to the existing
sigma-scan dispatcher; these files do not alter training or bound source.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
MILESTONES = (500, 1000, 2000, 3000)
HOSTS = tuple(f"Tdebug{i}" for i in range(1, 7))
SCHEMA = "ar_v03_formal_eval_followup_v1"
METRICS = REPO / "cosmos3_joint_video_hand_pose/scripts/ar_v03_formal_cpu_metrics.py"
DISPATCHER = REPO / "cosmos3_joint_video_hand_pose/scripts/ar_v03_sigma_scan.py"


def dispatcher():
    spec = importlib.util.spec_from_file_location("ar_v03_bound_scan_dispatcher", DISPATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_command(arguments):
    env = ["env", "CUDA_VISIBLE_DEVICES=", "LD_LIBRARY_PATH=",
           f"PYTHONPATH={REPO}:{REPO}/packages/cosmos3", "OMP_NUM_THREADS=1",
           "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"]
    return "cd " + shlex.quote(str(REPO)) + " && " + shlex.join(env + arguments)


def init(root, training_output_root, excluded):
    scan = dispatcher()
    root, training_output_root = Path(root).resolve(), Path(training_output_root).resolve()
    if not excluded or len(set(excluded)) != len(excluded) or any(x not in HOSTS for x in excluded):
        raise ValueError("training-node exclusion must be explicit, unique Tdebug aliases")
    root.mkdir(parents=True, exist_ok=False)
    campaigns = []
    for step in MILESTONES:
        target = root / f"step{step:06d}"
        plan = scan.prepare(target, "formal")
        campaigns.append(dict(step=step, scan_root=str(target), jobs=plan["jobs"],
                              results=plan["expected_results"], plan_sha256=scan.digest(target / "plan.json")))
    config = dict(schema=SCHEMA, created_at=scan.now(), root=str(root),
                  training_output_root=str(training_output_root), excluded_training_nodes=list(excluded),
                  permitted_evaluation_nodes=[x for x in HOSTS if x not in excluded],
                  milestones=list(MILESTONES), campaigns=campaigns, jobs_per_checkpoint=72,
                  results_per_checkpoint=96, total_planned_results=384,
                  heartbeat_interval_minutes=10, dispatcher=str(DISPATCHER), metrics_script=str(METRICS),
                  checkpoint_readiness="latest_checkpoint.txt at or beyond milestone plus all four DCP metadata/storage bounds",
                  GPU_launch_policy="MCP only, fresh CPU/memory/load/nvidia checks, exclude training nodes",
                  launch_status="four_plans_prepared_only_no_checkpoint_binding_or_GPU",
                  wandb_verification="root verifies the training run separately through W&B API")
    scan.write_new(root / "followup.json", config)
    return config


def load(root):
    scan = dispatcher()
    root = Path(root).resolve()
    config = scan.read_json(root / "followup.json")
    if config["schema"] != SCHEMA or config["root"] != str(root) or config["milestones"] != list(MILESTONES):
        raise ValueError("followup identity/milestone mismatch")
    for campaign in config["campaigns"]:
        path = Path(campaign["scan_root"]) / "plan.json"
        if scan.digest(path) != campaign["plan_sha256"]:
            raise ValueError("formal plan changed after initialization")
    return scan, root, config


def discover_run_dir(output_root):
    output_root = Path(output_root)
    if not output_root.exists():
        return None, "training output root not created yet"
    candidates = ([output_root / "config.yaml"] if (output_root / "config.yaml").is_file()
                  else list(output_root.rglob("config.yaml")))
    if not candidates:
        return None, "waiting for official config.yaml snapshot"
    if len(candidates) != 1:
        raise ValueError("multiple config.yaml snapshots under training output root; explicit run provenance required")
    return candidates[0].parent.resolve(), None


def storage_bounds(directory):
    """Read metadata only; never materialize DCP tensors."""
    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(str(directory)).read_metadata()
    requirements = {}
    for info in metadata.storage_data.values():
        relative = Path(info.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("DCP metadata storage path escapes its directory")
        requirements[str(relative)] = max(requirements.get(str(relative), 0), info.offset + info.length)
    if not requirements:
        raise ValueError("DCP metadata has no storage records")
    shards = []
    for relative, minimum in requirements.items():
        path = directory / relative
        if not path.is_file() or path.stat().st_size < minimum:
            raise ValueError(f"DCP storage incomplete: {path}, required size {minimum}")
        shards.append(dict(path=str(path), bytes=path.stat().st_size, required_bytes=minimum))
    return dict(metadata_keys=len(metadata.state_dict_metadata), storage_files=shards)


def checkpoint_ready(run_dir, step, *, storage_validator=storage_bounds):
    if run_dir is None:
        return dict(ready=False, reason="official training snapshot has not appeared")
    run_dir = Path(run_dir)
    latest = run_dir / "checkpoints/latest_checkpoint.txt"
    if not latest.is_file():
        return dict(ready=False, reason="latest_checkpoint.txt has not appeared")
    value = latest.read_text().strip()
    match = re.fullmatch(r"iter_(\d{9})", value)
    if not match or int(match.group(1)) < step:
        return dict(ready=False, reason="latest fully saved checkpoint is below milestone", latest=value)
    directory = run_dir / "checkpoints" / f"iter_{step:09d}"
    names = ("model", "trainer", "optim", "scheduler")
    missing = [str(directory / name / ".metadata") for name in names
               if not (directory / name / ".metadata").is_file()]
    if missing:
        return dict(ready=False, reason="milestone DCP metadata missing", missing=missing, latest=value)
    # Rank zero writes latest_checkpoint.txt only after official save returns.
    # Every referenced storage range is also checked, including replicas' shard
    # layout, without requiring any guessed shard count or reading model tensors.
    details = {name: storage_validator(directory / name) for name in names}
    return dict(ready=True, latest=value, checkpoint=str(directory / "model"),
                snapshot=str(run_dir / "config.yaml"), DCP=details,
                verification_scope="CPU metadata/storage sizes only, no tensor or GPU load")


def states(scan, campaign):
    root, plan = scan.load_plan(campaign["scan_root"])
    values = [scan.read_json(root / "states" / (task["job_id"] + ".json")) for task in plan["tasks"]]
    return root, plan, values


def metrics_command(scan, campaign, mode, config):
    root = Path(campaign["scan_root"])
    output = root / "metrics" / ("pilot_cpu_consistency.json" if mode == "pilot" else "comparison.json")
    arguments = [str(scan.PYTHON), str(Path(__file__).resolve()), "run-metrics",
                 "--root", config["root"], "--step", str(campaign["step"]),
                 "--mode", mode, "--workers", "24"]
    commands = {node: dict(hostAlias=node, command=cpu_command(arguments + ["--node", node]))
                for node in config["permitted_evaluation_nodes"]}
    return dict(commands_by_node=commands, choose_exactly_one_node=True,
                requires_CPU_resource_check=True, GPU_used=False, output=str(output),
                duplicate_launch_guard="exclusive durable metrics stage claim; PID/log/exit receipt")


def metric_claim(scan_root, mode):
    return Path(scan_root) / "metrics" / (mode + "_claim")


def process_identity(pid):
    """Linux PID birth/state; prevents treating a reused PID as our process."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(start_ticks=int(fields[19]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError, TypeError):
        return None


def process_probe_command(scan, campaign, config, mode, state):
    node = state.get("node_label")
    if node not in config["permitted_evaluation_nodes"]:
        return "process receipt lacks an allowed node; inspect its exact claim before acting"
    arguments = [str(scan.PYTHON), str(Path(__file__).resolve()), "probe-process",
                 "--root", config["root"], "--step", str(campaign["step"]),
                 "--mode", mode, "--node", node]
    return dict(hostAlias=node, command=cpu_command(arguments), read_only=True,
                next_action="MCP probe recorded wrapper/child PID on this node; report dead/orphaned processes, never retry automatically")


def probe_process(root, step, mode, node):
    scan, root, config = load(root)
    if node not in config["permitted_evaluation_nodes"]:
        raise ValueError("formal training nodes are excluded from metric process probes")
    campaign = next(x for x in config["campaigns"] if x["step"] == step)
    state = scan.read_json(metric_claim(campaign["scan_root"], mode) / "state.json")
    if state.get("node_label") != node or state.get("hostname") != socket.gethostname():
        raise ValueError("process probe must run through MCP on the recorded node/hostname")
    live = {}
    for field in ("PID", "child_PID"):
        actual = process_identity(state.get(field))
        expected = state.get(field + "_start_ticks")
        live[field] = bool(actual and actual["state"] != "Z"
                           and (expected is None or expected == actual["start_ticks"]))
    failed = state["status"] == "failed" or (state["status"] == "running" and not live["PID"])
    return dict(step=step, mode=mode, node_label=node, receipt_status=state["status"],
                live_processes=live, phase="attention_required" if failed else state["status"],
                log=state.get("log"), output=state.get("output"), read_only=True,
                next_action=("wrapper died; inspect log and any live orphan child, never overwrite/retry or kill automatically"
                             if failed else "read existing log/receipt; do not start duplicate work"))


def inspect(root):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    scan, root, config = load(root)
    run_dir, waiting = discover_run_dir(config["training_output_root"])
    reports = []
    for campaign in config["campaigns"]:
        target, plan, values = states(scan, campaign)
        summary = scan.progress(target, plan)
        failed = [x for x in values if x["status"] == "failed"]
        running = [x for x in values if x["status"] == "running"]
        report = dict(step=campaign["step"], scan_root=str(target), progress=summary,
                      readiness=None, phase=None, next_action=None)
        if failed or (target / "STOP").exists():
            report.update(phase="attention_required", failures=failed,
                          next_action="inspect exact job errors; never overwrite or automatically retry")
        elif (target / "binding.json").exists():
            scan.load_binding(target)
            pilot_ids = {x["job_id"] for x in plan["tasks"] if x["stage"] == "pilot"}
            pilot = [x for x in values if x["job_id"] in pilot_ids]
            pilot_success = all(x["status"] == "success" for x in pilot)
            if running:
                report.update(phase="GPU_running", active_tasks=[
                    {k: x.get(k) for k in ("job_id", "node_label", "hostname", "gpu", "child_pid", "started_at", "log")}
                    for x in running], next_action="read progress; do not duplicate existing workers")
            elif not pilot_success:
                report.update(phase="pilot_pending", next_action="fresh-check one nontraining node and dispatch 3 pilot GPU workers")
            elif summary["completed_results"] != 96:
                report.update(phase="remaining_pending", next_action="fresh-check nontraining nodes and dispatch remaining workers")
            else:
                comparison = target / "metrics/comparison.json"
                preflight = target / "metrics/pilot_cpu_consistency.json"
                mode = "aggregate" if preflight.is_file() else "pilot"
                claim = metric_claim(target, mode)
                if comparison.is_file():
                    data = scan.read_json(comparison)
                    if (data.get("archive_count") != 96 or data.get("groups") != 9
                            or data.get("step") != campaign["step"]
                            or data["source"]["plan_sha256"] != scan.digest(target / "plan.json")
                            or data["source"]["binding_sha256"] != scan.digest(target / "binding.json")):
                        raise ValueError("formal metric report must contain 96 archives and nine groups")
                    report.update(phase="complete", metrics_report=str(comparison),
                                  next_action="record tables/1..17 curves/overlays and this checkpoint's completion")
                elif claim.exists():
                    receipt = claim / "state.json"
                    data = scan.read_json(receipt) if receipt.is_file() else {"status": "claimed"}
                    if data["status"] == "failed":
                        report.update(phase="attention_required", metrics_error=data,
                                      next_action="inspect CPU stage log/exit; never auto-retry or overwrite")
                    else:
                        report.update(phase="CPU_metrics_running", metrics_process=data,
                                      next_action=process_probe_command(scan, campaign, config, mode, data))
                elif preflight.is_file():
                    data = scan.read_json(preflight)
                    if (not data.get("exact_equal")
                            or data["source"]["plan_sha256"] != scan.digest(target / "plan.json")
                            or data["source"]["binding_sha256"] != scan.digest(target / "binding.json")):
                        raise ValueError("CPU metric serial/parallel preflight did not pass")
                    report.update(phase="CPU_aggregate_pending", next_action=metrics_command(scan, campaign, "aggregate", config))
                else:
                    report.update(phase="CPU_pilot_pending", next_action=metrics_command(scan, campaign, "pilot", config))
        else:
            ready = checkpoint_ready(run_dir, campaign["step"])
            report["readiness"] = ready
            if ready["ready"]:
                command = cpu_command([str(scan.PYTHON), str(Path(__file__).resolve()), "bind-ready",
                                       "--root", str(root), "--step", str(campaign["step"])])
                report.update(phase="checkpoint_ready_unbound", next_action=dict(command=command, GPU_used=False))
            else:
                report.update(phase="waiting_checkpoint", next_action="wait quietly until official save completes")
        reports.append(report)
    # One GPU campaign at a time. Completed GPU campaigns may finish CPU metrics
    # while the next saved milestone scans on other nodes.
    active = [x for x in reports if x["phase"] == "GPU_running"]
    actionable = [x for x in reports if x["phase"] not in
                  ("complete", "waiting_checkpoint", "GPU_running", "CPU_metrics_running")]
    gpu_phases = {"checkpoint_ready_unbound", "pilot_pending", "remaining_pending"}
    if active:
        actionable = [x for x in actionable if x["phase"] not in gpu_phases]
    return dict(schema=SCHEMA, inspected_at=scan.now(), training_output_root=config["training_output_root"],
                discovered_run_dir=str(run_dir) if run_dir else None, snapshot_wait_reason=waiting,
                excluded_training_nodes=config["excluded_training_nodes"],
                permitted_evaluation_nodes=config["permitted_evaluation_nodes"], campaigns=reports,
                next_campaign=actionable[0] if actionable else None,
                automatic_GPU_or_SSH_actions_performed=False)


def bind_ready(root, step):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    scan, root, config = load(root)
    campaign = next(x for x in config["campaigns"] if x["step"] == step)
    run_dir, reason = discover_run_dir(config["training_output_root"])
    readiness = checkpoint_ready(run_dir, step)
    if not readiness["ready"]:
        raise ValueError(f"milestone is not fully saved: {readiness}")
    if (Path(campaign["scan_root"]) / "binding.json").exists():
        raise FileExistsError("binding already exists; use inspect, never overwrite")
    return scan.bind(campaign["scan_root"], readiness["checkpoint"], readiness["snapshot"])


def dispatch_command(root, step, stage, node, gpu_indexes):
    scan, root, config = load(root)
    if node not in config["permitted_evaluation_nodes"]:
        raise ValueError("this node is excluded because formal training owns it")
    if not gpu_indexes or len(gpu_indexes) != len(set(gpu_indexes)) or any(x < 0 for x in gpu_indexes):
        raise ValueError("unique nonnegative GPU indexes required")
    campaign = next(x for x in config["campaigns"] if x["step"] == step)
    target, plan, values = states(scan, campaign)
    scan.load_binding(target)
    if (target / "STOP").exists() or any(x["status"] == "failed" for x in values):
        raise ValueError("campaign stopped; manual error review required")
    chosen_ids = {x["job_id"] for x in plan["tasks"] if x["stage"] == stage}
    if not any(x["status"] == "pending" for x in values if x["job_id"] in chosen_ids):
        raise ValueError("this stage has no pending jobs; do not duplicate active/completed workers")
    if stage == "remaining":
        ids = {x["job_id"] for x in plan["tasks"] if x["stage"] == "pilot"}
        if not all(x["status"] == "success" for x in values if x["job_id"] in ids):
            raise ValueError("all three sigma/dual-mode pilot jobs must pass before remaining")
    for other in config["campaigns"]:
        if other["step"] != step:
            _, _, other_states = states(scan, other)
            if any(x["status"] == "running" for x in other_states):
                raise ValueError("another milestone GPU campaign is active")
    arguments = [str(scan.PYTHON), str(DISPATCHER), "supervise", "--root", str(target),
                 "--stage", stage, "--gpus", ",".join(map(str, gpu_indexes)), "--node-label", node]
    # Worker sets physical CUDA_VISIBLE_DEVICES itself; removing it from the
    # CPU command is unnecessary because the dispatcher sets each child value.
    return dict(hostAlias=node, command=cpu_command(arguments), requires_fresh_resource_check=True,
                prohibited_nodes=config["excluded_training_nodes"], stage=stage,
                gpu_indexes=gpu_indexes, GPU_processes_started_by_this_controller=False)


def run_metrics(root, step, mode, node, workers):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    scan, root, config = load(root)
    if node not in config["permitted_evaluation_nodes"]:
        raise ValueError("formal training nodes are excluded from CPU batch dispatch")
    campaign = next(x for x in config["campaigns"] if x["step"] == step)
    target, plan, values = states(scan, campaign)
    scan.load_binding(target)
    if len(values) != 72 or not all(x["status"] == "success" for x in values):
        raise ValueError("all 72 GPU jobs/96 archives must pass before CPU metrics")
    if workers < 1:
        raise ValueError("positive CPU worker count required")
    resource = scan.cpu_resources()
    needed = 3 if mode == "pilot" else workers
    if resource["effective_cpus"] < needed or resource["memory"]["MemAvailable"] < needed * (4 << 30):
        raise ValueError("insufficient real CPU quota or available memory")
    output = target / "metrics" / ("pilot_cpu_consistency.json" if mode == "pilot" else "comparison.json")
    preflight = target / "metrics/pilot_cpu_consistency.json"
    if mode == "aggregate":
        proof = scan.read_json(preflight)
        if (proof.get("exact_equal") is not True
                or proof["source"]["plan_sha256"] != scan.digest(target / "plan.json")
                or proof["source"]["binding_sha256"] != scan.digest(target / "binding.json")):
            raise ValueError("CPU preflight must match this plan/binding")
    if output.exists():
        raise FileExistsError("metric output already exists; inspect instead of repeating")
    claim = metric_claim(target, mode)
    claim.mkdir(parents=True, exist_ok=False)
    log = target / "metrics" / (mode + "_cpu.log")
    command = [str(scan.PYTHON), str(METRICS), mode, "--scan-root", str(target),
               "--step", str(step), "--run-root", config["training_output_root"],
               "--output", str(output), "--workers", str(workers)]
    if mode == "aggregate":
        command += ["--pilot-preflight", str(preflight)]
    state = dict(status="running", node_label=node, hostname=socket.gethostname(), mode=mode,
                 step=step, PID=os.getpid(), started_at=scan.now(), command=command,
                 log=str(log), output=str(output), CPU_resources=resource, GPU_used=False)
    identity = process_identity(state["PID"])
    state["PID_start_ticks"] = identity["start_ticks"] if identity else None
    scan.write_new(claim / "state.json", state)
    began = time.monotonic()
    try:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONPATH=f"{REPO}:{REPO}/packages/cosmos3",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        with log.open("x") as handle:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
            state["child_PID"] = process.pid
            identity = process_identity(process.pid)
            state["child_PID_start_ticks"] = identity["start_ticks"] if identity else None
            scan.replace_json(claim / "state.json", state)
            code = process.wait()
        state.update(status="success" if code == 0 and output.is_file() else "failed", exit_code=code)
    except Exception as exc:
        state.update(status="failed", exit_code=None, error=f"{type(exc).__name__}: {exc}")
    state.update(finished_at=scan.now(), wall_seconds=time.monotonic() - began)
    scan.replace_json(claim / "state.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("init", help="CPU: prepare four unbound formal scans")
    first.add_argument("--root", type=Path, required=True)
    first.add_argument("--training-output-root", type=Path, required=True)
    first.add_argument("--exclude-training-nodes", nargs="+", choices=HOSTS, required=True)
    read = sub.add_parser("inspect", help="CPU: read milestones and produce next commands only")
    read.add_argument("--root", type=Path, required=True)
    bind = sub.add_parser("bind-ready", help="CPU: bind a genuinely saved milestone only")
    bind.add_argument("--root", type=Path, required=True)
    bind.add_argument("--step", type=int, choices=MILESTONES, required=True)
    dispatch = sub.add_parser("dispatch-command", help="print one MCP command; never execute it")
    dispatch.add_argument("--root", type=Path, required=True)
    dispatch.add_argument("--step", type=int, choices=MILESTONES, required=True)
    dispatch.add_argument("--stage", choices=("pilot", "remaining"), required=True)
    dispatch.add_argument("--node", choices=HOSTS, required=True)
    dispatch.add_argument("--gpus", required=True)
    metrics = sub.add_parser("run-metrics", help="CPU: claim and run one metric stage, no GPU")
    metrics.add_argument("--root", type=Path, required=True)
    metrics.add_argument("--step", type=int, choices=MILESTONES, required=True)
    metrics.add_argument("--mode", choices=("pilot", "aggregate"), required=True)
    metrics.add_argument("--node", choices=HOSTS, required=True)
    metrics.add_argument("--workers", type=int, default=24)
    probe = sub.add_parser("probe-process", help="read-only: check CPU stage PIDs on the recorded MCP node")
    probe.add_argument("--root", type=Path, required=True)
    probe.add_argument("--step", type=int, choices=MILESTONES, required=True)
    probe.add_argument("--mode", choices=("pilot", "aggregate"), required=True)
    probe.add_argument("--node", choices=HOSTS, required=True)
    args = parser.parse_args()
    if args.command == "init":
        result = init(args.root, args.training_output_root, args.exclude_training_nodes)
    elif args.command == "inspect":
        result = inspect(args.root)
    elif args.command == "bind-ready":
        result = bind_ready(args.root, args.step)
    elif args.command == "run-metrics":
        result = run_metrics(args.root, args.step, args.mode, args.node, args.workers)
    elif args.command == "probe-process":
        result = probe_process(args.root, args.step, args.mode, args.node)
    else:
        result = dispatch_command(args.root, args.step, args.stage, args.node,
                                  [int(x) for x in args.gpus.split(",")])
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "run-metrics" and result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
