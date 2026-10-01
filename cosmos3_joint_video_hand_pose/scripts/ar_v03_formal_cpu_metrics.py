#!/usr/bin/env python3
"""Aggregate completed formal V0.3 sigma scans on CPU using frozen V0.2 metrics.

pilot compares the same three archives with workers=1 and workers=3.
aggregate reuses that receipt and computes the remaining archives. Neither mode
binds a checkpoint, samples, trains, edits old reports, or loads Nano weights.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

from cosmos3_joint_video_hand_pose.scripts import ar_v03_sigma_scan as scan

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "outputs/maintenance/video_lr_eval_extended_20261001T2229/report/comparison.json"
BASELINE_AUDIT = BASELINE.parents[1] / "baseline_parameter_audit.json"
BASELINE_AUDIT_SHA256 = "b40c15281e4d4b58762142a2f5ec5bcd4b854351d85103b02b175b4f6d262ac6"
SCHEMA = "ar_v03_formal_cpu_metrics_v1"
MILESTONES = (500, 1000, 2000, 3000)
FIELDS = (
    "local_camera_end_mm", "local_left_wrist_end_mm", "local_right_wrist_end_mm",
    "no_motion_camera_end_mm", "no_motion_left_wrist_end_mm", "no_motion_right_wrist_end_mm",
    "local_left_wrist_end_degrees", "local_right_wrist_end_degrees",
    "local_left_wrist_local_shape_mpjpe_mm", "local_right_wrist_local_shape_mpjpe_mm",
)
BASELINE_KEYS = (
    "baseline_gt", "baseline_generated", "lr1000_gt", "lr1000_generated", "lr1000_train16_gt",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def cpu_only():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def resource_snapshot(workers):
    if workers < 1:
        raise ValueError("positive worker count required")
    affinity = len(os.sched_getaffinity(0))
    quota = None
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        maximum, period = cpu_max.read_text().split()
        if maximum != "max":
            quota = int(maximum) / int(period)
    effective = min(affinity, quota) if quota is not None else affinity
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
    available = memory["MemAvailable"]
    memory_max, memory_current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if memory_max.is_file() and memory_max.read_text().strip() != "max":
        available = min(available, int(memory_max.read_text()) - int(memory_current.read_text()))
    load = list(os.getloadavg())
    if workers > effective or available < workers * (4 << 30) or load[0] + workers > effective:
        raise RuntimeError("insufficient unoccupied CPU quota or available memory")
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip()
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True).strip()
    return dict(hostname=socket.gethostname(), affinity_cpus=affinity, quota_cpus=quota,
                effective_cpus=effective, memory_available_bytes=available, loadavg=load,
                gpu_snapshot=gpu, gpu_processes=processes, workers=workers,
                threads_per_worker=1, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def load_jobs(root, *, step, run_root, pilot=False):
    if step not in MILESTONES:
        raise ValueError("formal step must be one of 500/1000/2000/3000")
    root, plan = scan.load_plan(root)
    binding = scan.load_binding(root)
    run_root = Path(run_root).resolve()
    checkpoint = Path(binding["checkpoint"]).resolve()
    if (plan["scope"] != "formal" or plan["expected_results"] != 96 or plan["jobs"] != 72
            or plan["sigmas"] != list(scan.SIGMAS)
            or checkpoint.name != "model" or checkpoint.parent.name != f"iter_{step:09d}"
            or not checkpoint.is_relative_to(run_root)
            or binding.get("prefix_low_noise_enabled") is not True or binding.get("sigma_hist_max") != 0.1
            or binding.get("contract_matches_snapshot") is not True):
        raise ValueError("formal scan/checkpoint/step/run binding mismatch")
    parents = plan["frozen_parents"]
    if set(parents) != set(scan.FROZEN):
        raise ValueError("formal scan requires both frozen parents")
    frozen = {}
    for split, (_, expected, count) in scan.FROZEN.items():
        parent = parents[split]
        if parent["sha256"] != expected or sha256(parent["path"]) != expected:
            raise ValueError(f"frozen {split} parent changed")
        frozen[split] = read_json(parent["path"])
        if parent["windows"] != count or len(frozen[split]) != count:
            raise ValueError(f"incorrect frozen {split} window count")
    expected_tasks = {(split, sigma, index) for split, (_, _, count) in scan.FROZEN.items()
                      for sigma in scan.SIGMAS for index in range(count)}
    identities = [(x["split"], x["sigma_small"], x["parent_window_index"]) for x in plan["tasks"]]
    if len(identities) != 72 or set(identities) != expected_tasks:
        raise ValueError("formal task identities are missing or duplicated")
    jobs = []
    for task in plan["tasks"]:
        split, index = task["split"], task["parent_window_index"]
        histories = ["gt", "generated"] if split == "heldout" else ["gt"]
        if (task["histories"] != histories or len(task["expected_archives"]) != len(histories)
                or task["expected_chunks"] != 17 or task["window"] != frozen[split][index]):
            raise ValueError("formal task history/window/chunk mismatch")
        if pilot and not (task["sigma_small"] == 0.02 and index == 0):
            continue
        state = read_json(root / "states" / (task["job_id"] + ".json"))
        if state["status"] != "success" or state["job_id"] != task["job_id"]:
            raise ValueError(f"not completed: {task['job_id']}")
        reports = state["validation"]
        validated = {x["archive"]: x for x in reports}
        if len(reports) != len(histories) or set(validated) != set(task["expected_archives"]):
            raise ValueError("archive validation inventory mismatch")
        for history, archive in zip(histories, task["expected_archives"]):
            entry = validated[archive]
            if (entry["history"] != history or entry["sigma_small"] != task["sigma_small"]
                    or entry["chunks"] != 17
                    or entry["parent_manifest_sha256"] != task["parent_manifest_sha256"]
                    or entry["fragment_manifest_sha256"] != task["fragment_manifest_sha256"]):
                raise ValueError("archive validation does not match the frozen task")
            jobs.append(dict(task=task, history=history, archive=archive, binding=binding,
                             validated_sha256=entry["sha256"]))
    if len(jobs) != (3 if pilot else 96):
        raise ValueError("formal archive count mismatch")
    return root, plan, binding, jobs


def validate_sample(action, video, job):
    task, history, binding = job["task"], job["history"], job["binding"]
    meta = action["metadata"]
    expected = dict(
        model_version="ar_v0.3.0", schema="ar_v02_rollout_v1", layout_version="joint_chunk_cond_v1",
        history=history, checkpoint=binding["checkpoint"], sample_id=task["window"]["sample_id"],
        source_offset=task["window"]["start"], seed=task["window"]["seed"], steps=30,
        chunk_size=4, num_frames=69, selected_windows=1, frozen_windows=1,
        sigma_small=task["sigma_small"], history_video_sigma=task["sigma_small"],
        history_action_sigma=task["sigma_small"], video_shift=5.0, action_shift=5.0,
        video_guidance=1.0, action_guidance=1.0,
        action_representation=binding["action_representation"],
        eval_windows_sha256=task["fragment_manifest_sha256"], raw_gt_disabled_diagnostic=False,
    )
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(f"{job['archive']}: {field} mismatch")
    for name in ("state_normalizer", "future_normalizer"):
        if meta[name]["sha256"] != binding["frozen_artifacts"][name]["sha256"]:
            raise ValueError("archive normalizer differs from checkpoint")
    for side in ("right", "left"):
        if meta["hand_codecs"][side]["sha256"] != binding["frozen_artifacts"][side + "_codec"]["sha256"]:
            raise ValueError("archive PCA15 codec differs from checkpoint")
    if action["hand_metric_scope"] != "all_finite_raw_coordinates_no_visibility_mask":
        raise ValueError("main action metrics require original raw GT")
    rows = action["metrics"]["chunks"]
    if ([x["chunk"] for x in rows] != list(range(1, 18))
            or any(x["action_count"] != 32 or x["video_sampled_future_frames"] != 16
                   or x["source_start"] != (x["chunk"] - 1) * 32 or x["source_stop"] != x["chunk"] * 32
                   for x in rows)):
        raise ValueError("expected all 17 complete C4/K8 action/RGB chunks")
    if video["archive_sha256"] != job["validated_sha256"]:
        raise ValueError("NPZ changed after GPU archive validation")
    video_expected = dict(input=str(Path(job["archive"]).resolve()), sample_id=meta["sample_id"],
                          source_offset=meta["source_offset"], checkpoint=meta["checkpoint"],
                          history=history, history_video_sigma=task["sigma_small"], seed=42, steps=30)
    for key, value in video_expected.items():
        if video.get(key) != value:
            raise ValueError(f"endpoint video identity mismatch: {key}")
    flow = video["flow"]["chunks"]
    if ([x["chunk"] for x in flow] != list(range(1, 18))
            or any(x["pairs"] != 1 or x["pixels"] != 320 * 180 for x in flow)):
        raise ValueError("per-archive endpoint pooled support mismatch")


def evaluate_job(job):
    cpu_only()
    import torch
    torch.set_num_threads(1)
    from cosmos3_joint_video_hand_pose.src.ar_v02_eval import evaluate_archive
    from cosmos3_joint_video_hand_pose.src.ar_v02_endpoint_video_metrics import evaluate
    if sha256(job["archive"]) != job["validated_sha256"]:
        raise ValueError("NPZ changed before CPU metrics")
    action = evaluate_archive(job["archive"], SimpleNamespace(rigid_only=False))
    video = evaluate(job["archive"])
    validate_sample(action, video, job)
    return dict(split=job["task"]["split"], window_index=job["task"]["parent_window_index"],
                sigma_small=job["task"]["sigma_small"], history=job["history"],
                action=action, endpoint_video=video)


def calculate(jobs, workers):
    if not jobs or workers < 1:
        raise ValueError("nonempty jobs and positive worker count required")
    started = time.monotonic()
    results = []
    if workers == 1:
        iterator = map(evaluate_job, jobs)
        for result in iterator:
            results.append(result)
            print_progress(results, jobs, workers, started)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for result in pool.map(evaluate_job, jobs):
                results.append(result)
                print_progress(results, jobs, workers, started)
    return results, time.monotonic() - started


def print_progress(results, jobs, workers, started):
    elapsed = time.monotonic() - started
    print(json.dumps(dict(completed=len(results), total=len(jobs), workers=workers,
                          elapsed_seconds=elapsed, archives_per_second=len(results) / elapsed)), flush=True)


def action_summary(rows):
    if not rows:
        raise ValueError("empty action scope")
    result = {}
    for field in FIELDS:
        weights = [r["action_count"] if "shape" in field else 1 for r in rows]
        if any(not math.isfinite(r[field]) for r in rows) or min(weights) <= 0:
            raise ValueError("finite action metrics and positive support required")
        result[field] = sum(r[field] * w for r, w in zip(rows, weights)) / sum(weights)
    frames = sum(r["video_sampled_future_frames"] for r in rows)
    if frames <= 0 or any(not math.isfinite(r["video_mse_uint8"]) or r["video_mse_uint8"] < 0 for r in rows):
        raise ValueError("finite nonnegative MSE and positive frame support required")
    mse = sum(r["video_mse_uint8"] * r["video_sampled_future_frames"] for r in rows) / frames
    result.update(video_mse_uint8=mse, video_psnr_db=10 * math.log10(255**2 / mse) if mse else None,
                  chunks=len(rows), action_frames=sum(r["action_count"] for r in rows),
                  future_rgb_frames=frames)
    return result


def make_records(samples, step):
    from cosmos3_joint_video_hand_pose.src.ar_v02_endpoint_video_metrics import summarize
    if step not in MILESTONES or len(samples) != 96:
        raise ValueError("96 archives and an accepted formal milestone required")
    records = []
    for sigma in scan.SIGMAS:
        for split, history, count in (("heldout", "gt", 8), ("heldout", "generated", 8), ("train", "gt", 16)):
            group = sorted((x for x in samples if x["sigma_small"] == sigma and x["split"] == split
                            and x["history"] == history), key=lambda x: x["window_index"])
            if [x["window_index"] for x in group] != list(range(count)):
                raise ValueError("each group requires every frozen window exactly once")
            rows = [r for x in group for r in x["action"]["metrics"]["chunks"]]
            video = summarize([x["endpoint_video"] for x in group])
            if (video["flow"]["pairs"] != count * 17 or video["flow"]["pixels"] != count * 17 * 320 * 180
                    or [x["chunk"] for x in video["chunks"]] != list(range(1, 18))):
                raise ValueError("group endpoint pooled support mismatch")
            records.append(dict(
                key=f"v03_step{step}_sigma{round(sigma * 1000):03d}_{split}_{history}",
                label=f"V0.3 uniform-prefix step{step}", split=split, mode=history, sigma=sigma,
                step=step, windows=count, all=action_summary(rows),
                chunks=[dict(chunk=k, **action_summary([r for r in rows if r["chunk"] == k]))
                        for k in range(1, 18)],
                chunk17plus=action_summary([r for r in rows if r["chunk"] >= 17]), video=video,
                samples=[dict(window_index=x["window_index"], sample_id=x["action"]["metadata"]["sample_id"],
                              source_offset=x["action"]["metadata"]["source_offset"],
                              archive=x["endpoint_video"]["input"],
                              archive_sha256=x["endpoint_video"]["archive_sha256"]) for x in group],
            ))
    return records


def load_baselines(path, plan):
    old = read_json(path)
    if old["flow_geometry"] != [320, 180] or old["flow_cosine"] != "pooled_dot_over_global_norms":
        raise ValueError("baseline report uses a different flow definition")
    records = []
    for key in BASELINE_KEYS:
        selected = [x for x in old["records"] if x["key"] == key]
        if len(selected) != 1:
            raise ValueError(f"missing or duplicated baseline: {key}")
        baseline = selected[0]
        split, count = ("train", 16) if key == "lr1000_train16_gt" else ("heldout", 8)
        history = "generated" if key.endswith("_generated") else "gt"
        frozen = read_json(plan["frozen_parents"][split]["path"])
        samples = sorted(baseline["samples"], key=lambda x: x["window_index"])
        expected = [(x["sample_id"], x["start"]) for x in frozen]
        actual = [(x["sample_id"], x["source_offset"]) for x in samples]
        if (baseline["windows"] != count or baseline["sigma"] != 0 or baseline["split"] != split
                or baseline["mode"] != history or actual != expected
                or [x["window_index"] for x in samples] != list(range(count))
                or baseline["video"]["flow"]["pairs"] != count * 17
                or baseline["video"]["flow"]["pixels"] != count * 17 * 320 * 180
                or [x["chunk"] for x in baseline["chunks"]] != list(range(1, 18))):
            raise ValueError(f"baseline identity/support mismatch: {key}")
        records.append(baseline)
    return records


def source_record(root, binding, step, run_root):
    return dict(tool=str(Path(__file__).resolve()), tool_sha256=sha256(__file__),
                scan_root=str(root), plan_sha256=sha256(root / "plan.json"),
                binding_sha256=sha256(root / "binding.json"), checkpoint=binding["checkpoint"],
                snapshot=binding["snapshot"], snapshot_sha256=binding["snapshot_sha256"],
                step=step, run_root=str(Path(run_root).resolve()))


def baseline_provenance():
    if sha256(BASELINE_AUDIT) != BASELINE_AUDIT_SHA256:
        raise ValueError("historical baseline parameter evidence changed")
    audit = read_json(BASELINE_AUDIT)
    return dict(
        path=str(BASELINE_AUDIT.resolve()), sha256=BASELINE_AUDIT_SHA256,
        applies_to_keys=["baseline_gt", "baseline_generated"],
        absent_archive_metadata_fields=audit["legacy_absent_metadata_fields"],
        source_commit=audit["source_commit"], source_file=audit["source_file"],
        source_sha256=audit["source_sha256"], launch_source=audit["launch_source"],
        note=audit["note"],
        limitation="historical source/launch evidence; these legacy NPZ files do not explicitly bind shift/CFG/history sigma",
        lr1000_groups="explicit sampling parameters in archive metadata; no legacy inference needed",
    )


def cached_pilot(path, source, jobs):
    pilot = read_json(path)
    if (pilot.get("schema") != SCHEMA or pilot.get("mode") != "pilot" or pilot.get("exact_equal") is not True
            or pilot.get("serial_workers") != 1 or pilot.get("parallel_workers") != 3
            or pilot["source"] != source):
        raise ValueError("pilot receipt belongs to another scan/tool/checkpoint")
    cached = pilot["samples"]
    expected = {(split, 0.02, 0, history) for split, history in
                (("heldout", "gt"), ("heldout", "generated"), ("train", "gt"))}
    identities = [(x["split"], x["sigma_small"], x["window_index"], x["history"]) for x in cached]
    if len(cached) != 3 or set(identities) != expected:
        raise ValueError("pilot cache does not contain the three required archives")
    inventory = {str(Path(j["archive"]).resolve()): j for j in jobs}
    for sample in cached:
        archive = sample["endpoint_video"]["input"]
        if archive not in inventory or sha256(archive) != inventory[archive]["validated_sha256"]:
            raise ValueError("pilot NPZ changed or belongs to another scan")
        validate_sample(sample["action"], sample["endpoint_video"], inventory[archive])
    return cached


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("mode", choices=["pilot", "aggregate"])
    value.add_argument("--scan-root", type=Path, required=True)
    value.add_argument("--step", type=int, choices=MILESTONES, required=True)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--workers", type=int, default=24)
    value.add_argument("--pilot-preflight", type=Path)
    value.add_argument("--baseline-report", type=Path, default=BASELINE)
    return value


def main():
    args = parser().parse_args()
    if args.output.exists() or args.workers < 1:
        raise ValueError("new output path and positive worker count required")
    if args.mode == "aggregate" and not args.pilot_preflight:
        raise ValueError("aggregate requires a matching workers1/3 --pilot-preflight receipt")
    cpu_only()
    hardware = resource_snapshot(3 if args.mode == "pilot" else args.workers)
    root, plan, binding, jobs = load_jobs(args.scan_root, step=args.step, run_root=args.run_root,
                                        pilot=args.mode == "pilot")
    source = source_record(root, binding, args.step, args.run_root)
    if args.mode == "pilot":
        serial, serial_seconds = calculate(jobs, 1)
        parallel, parallel_seconds = calculate(jobs, 3)
        if serial != parallel:
            raise ValueError("pilot metric outputs differ between workers1 and workers3")
        report = dict(schema=SCHEMA, mode="pilot", step=args.step, source=source, hardware=hardware, exact_equal=True,
                      serial_workers=1, parallel_workers=3, serial_seconds=serial_seconds,
                      parallel_seconds=parallel_seconds, samples=serial)
    else:
        baselines = load_baselines(args.baseline_report, plan)
        provenance = baseline_provenance()
        cached = cached_pilot(args.pilot_preflight, source, jobs)
        cached_paths = {x["endpoint_video"]["input"] for x in cached}
        pending = [job for job in jobs if str(Path(job["archive"]).resolve()) not in cached_paths]
        samples, seconds = calculate(pending, min(args.workers, len(pending)))
        samples += cached
        samples.sort(key=lambda x: (x["sigma_small"], x["split"], x["history"], x["window_index"]))
        records = make_records(samples, args.step)
        report = dict(
            schema=SCHEMA, mode="aggregate", step=args.step, source=source, hardware=hardware,
            frozen_parents=plan["frozen_parents"], groups=9, archive_count=96,
            recomputed_archives=len(pending), seconds=seconds, samples=samples, records=records, baselines=baselines,
            pilot_receipt=dict(path=str(args.pilot_preflight.resolve()), sha256=sha256(args.pilot_preflight),
                               cached_archives=len(cached)),
            baseline_source=dict(path=str(args.baseline_report.resolve()), sha256=sha256(args.baseline_report),
                                 inference_sigma=0,
                                 legacy_parameter_evidence=provenance,
                                 comparison_scope="same frozen heldout8/train16; reference checkpoint1000; different training duration/inference sigma"),
            flow_geometry=[320, 180], flow_cosine="pooled_dot_over_global_norms",
            flow_scope="chunk_condition_start_to_last_future",
            wrist_aggregation="one endpoint per chunk; original raw GT; generated includes history drift",
            shape_aggregation="weighted by action_count",
            psnr_aggregation="weighted future-frame MSE then dB",
        )
    write_new(args.output, report)
    print(json.dumps(dict(output=str(args.output), mode=args.mode, step=args.step, status="complete")), flush=True)


if __name__ == "__main__":
    main()
