#!/usr/bin/env python3
"""Prepare and supervise fixed-window V0.3.1 sigma scans; never train or use SSH.

prepare/status/bind are CPU commands. supervise is an explicit local-node GPU
launch; invoke it separately through MCP on each allocated node after checking
resources. Each GPU worker claims one job at a time. A job reuses the stable
ar_v031_eval CLI and loads one model for its requested history modes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import signal
import socket
import subprocess
import sys
import time

from cosmos3_joint_video_hand_pose.src.ar_v031_eval_contract import (
    MODEL_TARGET, MODEL_VERSION, PREFIX_LOSS_DENOMINATOR, PREFIX_LOSS_MASK_SCOPE,
    repository, validate_checkpoint_snapshot,
)
REPO = repository()
PACKAGE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/lzh/miniconda3/envs/cosmos3/bin/python")
SCHEMA = "ar_v031_sigma_scan_v1"
SIGMAS = (0.02, 0.05, 0.1)
FROZEN = {
    "heldout": (REPO / "outputs/maintenance/video_lr_eval_extended_20261001T2229/heldout8.json",
                "8ac5c99d0a80f963b03858dd9260e12aa3cccead0e0bf4475427bd863ec1f03d", 8),
    "train": (REPO / "cosmos3_joint_video_hand_pose/configs/overfit16_windows_20261001.json",
              "7e088380b461fce4ea34ec6aeea141d445d1aff6c3acf81f2cbfb401a6bb9ee4", 16),
}
ARTIFACT_FIELDS = {
    "state_normalizer": "chunk_state_normalizer",
    "future_normalizer": "future_normalizer",
    "right_codec": "right_codec",
    "left_codec": "left_codec",
    "valid_windows_manifest": "valid_windows_manifest",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def replace_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    write_new(temporary, value)
    os.replace(temporary, path)


def checked(command):
    return subprocess.check_output(command, text=True, cwd=REPO).strip()


def prepare(root, scope):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    tasks, parents = [], {}
    for split in (["heldout"] if scope == "short" else ["heldout", "train"]):
        parent, expected, count = FROZEN[split]
        if digest(parent) != expected:
            raise ValueError(f"frozen {split} parent hash changed")
        windows = read_json(parent)
        if len(windows) != count:
            raise ValueError(f"expected {count} {split} windows")
        parents[split] = dict(path=str(parent), sha256=expected, windows=count)
        for index, item in enumerate(windows):
            if any(type(item.get(k)) is not int for k in ("start", "frames", "seed")):
                raise ValueError("frozen window must contain integer start/frames/seed")
            if item["frames"] != 273 or item["seed"] != 42 or not item.get("sample_id"):
                raise ValueError("scan requires the unchanged 273-frame, seed42 windows")
            fragment = root / "manifests" / f"{split}_window{index:02d}.json"
            write_new(fragment, [item])
            fragment_hash = digest(fragment)
            for sigma in SIGMAS:
                tag = f"sigma_{round(sigma * 1000):03d}"
                job_id = f"{split}_{tag}_window{index:02d}"
                histories = ["gt", "generated"] if split == "heldout" else ["gt"]
                output = root / "eval" / tag / split / f"window{index:02d}"
                tasks.append(dict(
                    job_id=job_id, split=split, histories=histories, sigma_small=sigma,
                    parent_manifest=str(parent), parent_manifest_sha256=expected,
                    parent_window_index=index, window=item,
                    fragment_manifest=str(fragment), fragment_manifest_sha256=fragment_hash,
                    output=str(output), expected_chunks=17,
                    expected_archives=[str(output / f"0000_{history}.npz") for history in histories],
                    stage="pilot" if split == "heldout" and index == 0 else "remaining",
                ))
    plan = dict(
        schema=SCHEMA, created_at=now(), scope=scope, root=str(root), sigmas=list(SIGMAS),
        repository=str(REPO), toml=str(PACKAGE / "configs/ar_v0_3_1.toml"),
        episodes_manifest=str(REPO / "outputs/data_expansion_20260928/episodes.csv"),
        segments_manifest=str(REPO / "outputs/data_expansion_20260928/segments.csv"),
        frozen_parents=parents, tasks=tasks, jobs=len(tasks),
        expected_results=sum(len(x["histories"]) for x in tasks),
        launch_status="prepared_only_no_GPU_or_sampling_started",
        sampling=dict(chunk_size=4, history_chunks=15, action_tokens_per_latent=8,
                      steps=30, video_shift=5.0, action_shift=5.0,
                      video_guidance=1.0, action_guidance=1.0),
    )
    write_new(root / "plan.json", plan)
    for name in ("states", "claims", "workers", "logs", "gpu_locks"):
        (root / name).mkdir()
    for task in tasks:
        write_new(root / "states" / (task["job_id"] + ".json"),
                  dict(job_id=task["job_id"], status="pending", expected_results=len(task["histories"])))
    return plan


def load_plan(root):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    if plan["schema"] != SCHEMA or plan["root"] != str(root):
        raise ValueError("scan plan identity mismatch")
    for parent in plan["frozen_parents"].values():
        if digest(parent["path"]) != parent["sha256"]:
            raise ValueError("frozen parent changed after preparation")
    for task in plan["tasks"]:
        if digest(task["fragment_manifest"]) != task["fragment_manifest_sha256"]:
            raise ValueError("window fragment changed")
        if read_json(task["fragment_manifest"]) != [task["window"]]:
            raise ValueError("window fragment identity mismatch")
        parent = plan["frozen_parents"][task["split"]]
        if (task["parent_manifest"] != parent["path"]
                or task["parent_manifest_sha256"] != parent["sha256"]
                or read_json(parent["path"])[task["parent_window_index"]] != task["window"]):
            raise ValueError("window fragment does not match its frozen parent position")
    return root, plan


def bind(root, checkpoint, snapshot):
    # Bind only a real completed checkpoint. Read DCP metadata and small contract
    # extra_state, never instantiate/load Nano tensors. YAML comes from Trainer.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root, plan = load_plan(root)
    checkpoint, snapshot = Path(checkpoint).resolve(), Path(snapshot).resolve()
    metadata_path = checkpoint / ".metadata"
    if not metadata_path.is_file() or not snapshot.is_file():
        raise ValueError("completed checkpoint metadata and training snapshot are required")
    if checkpoint.name != "model" or snapshot.parent != checkpoint.parents[2]:
        raise ValueError("--checkpoint must be this snapshot run's iter_*/model directory")
    import yaml
    import torch.distributed.checkpoint as dcp
    from cosmos3_joint_video_hand_pose.src.ar_v02_contract import ARTrainingContract

    config, provenance = validate_checkpoint_snapshot(snapshot, checkpoint)
    dataset = config["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
    artifacts = {name: dict(path=str(Path(dataset[field]).resolve()), sha256=digest(dataset[field]))
                 for name, field in ARTIFACT_FIELDS.items()}
    reader = dcp.FileSystemReader(str(checkpoint))
    keys = list(reader.read_metadata().state_dict_metadata)
    for name in ("action2llm", "llm2action", "action_modality_embed", "action_state_embed", "vision_condition_embed"):
        if not any(name in key for key in keys):
            raise ValueError(f"checkpoint missing {name}")
    contract = ARTrainingContract(
        state_normalizer=dataset["chunk_state_normalizer"], action_normalizer=dataset["future_normalizer"],
        manifest_sha256=artifacts["valid_windows_manifest"]["sha256"],
        representation=dataset["action_representation"], right_hand_codec=dataset["right_codec"],
        left_hand_codec=dataset["left_codec"],
    )
    extra = "net.ar_training_contract._extra_state"
    binding = {extra: contract.get_extra_state()}
    dcp.load(binding, storage_reader=reader, no_dist=True)
    contract.set_extra_state(binding[extra])
    source_paths = sorted((REPO / "cosmos3_joint_video_hand_pose/src").glob("ar_v03_*.py"))
    source_paths += sorted((PACKAGE / "src").glob("ar_v031_*.py"))
    source_paths += sorted((PACKAGE / "scripts").glob("ar_v031_*.py"))
    source_paths += [REPO / "cosmos3_joint_video_hand_pose/src/ar_v02_eval.py",
                     REPO / "cosmos3_joint_video_hand_pose/src/ar_v02_endpoint_video_metrics.py",
                     Path(plan["toml"])]
    value = dict(
        schema=SCHEMA, bound_at=now(), plan_sha256=digest(root / "plan.json"),
        checkpoint=str(checkpoint), checkpoint_metadata_sha256=digest(metadata_path),
        checkpoint_metadata_keys=len(keys), snapshot=str(snapshot), snapshot_sha256=digest(snapshot),
        model_target=config["model"]["_target_"], model_version=MODEL_VERSION, mask_prefix_loss=True,
        prefix_loss_denominator=PREFIX_LOSS_DENOMINATOR, prefix_loss_mask_scope=PREFIX_LOSS_MASK_SCOPE,
        recipe_reference_snapshot_sha256=provenance["recipe_reference_snapshot_sha256"],
        checkpoint_readiness=provenance["checkpoint_readiness"],
        prefix_low_noise_enabled=True, sigma_hist_max=0.1,
        action_representation=dataset["action_representation"], frozen_artifacts=artifacts,
        contract_schema=binding[extra]["schema"], contract_matches_snapshot=True,
        source_commit=checked(["git", "rev-parse", "HEAD"]),
        source_files={str(path): digest(path) for path in source_paths},
        verification_scope="CPU metadata and contract only; no Nano weight load or GPU",
    )
    write_new(root / "binding.json", value)
    return value


def load_binding(root):
    binding = read_json(root / "binding.json")
    if digest(root / "plan.json") != binding["plan_sha256"]:
        raise ValueError("plan changed after checkpoint binding")
    for path, expected in binding["source_files"].items():
        if digest(path) != expected:
            raise ValueError(f"bound source/config changed: {path}")
    for path, expected in [
        (binding["snapshot"], binding["snapshot_sha256"]),
        (Path(binding["checkpoint"]) / ".metadata", binding["checkpoint_metadata_sha256"]),
    ]:
        if digest(path) != expected:
            raise ValueError(f"checkpoint/snapshot changed: {path}")
    if (binding.get("schema") != SCHEMA or binding.get("model_target") != MODEL_TARGET
            or binding.get("model_version") != MODEL_VERSION or binding.get("mask_prefix_loss") is not True
            or binding.get("prefix_loss_denominator") != PREFIX_LOSS_DENOMINATOR
            or binding.get("prefix_loss_mask_scope") != PREFIX_LOSS_MASK_SCOPE):
        raise ValueError("binding is not a real V0.3.1 masked-prefix run")
    validate_checkpoint_snapshot(binding["snapshot"], binding["checkpoint"])
    return binding


def progress(root, plan):
    states = [read_json(root / "states" / (task["job_id"] + ".json")) for task in plan["tasks"]]
    completed = [x for x in states if x["status"] == "success"]
    result_count = sum(x["expected_results"] for x in completed)
    starts = [x["started_epoch"] for x in states if "started_epoch" in x]
    elapsed = max(0.0, time.time() - min(starts)) if starts else 0.0
    return dict(
        time=now(), completed_jobs=len(completed), total_jobs=plan["jobs"],
        completed_results=result_count, total_results=plan["expected_results"],
        running_jobs=sum(x["status"] == "running" for x in states),
        failed_jobs=[x["job_id"] for x in states if x["status"] == "failed"],
        elapsed_seconds=elapsed, results_per_minute=result_count * 60 / elapsed if elapsed else 0.0,
    )


def pilot_complete(root, plan):
    ids = [x["job_id"] for x in plan["tasks"] if x["stage"] == "pilot"]
    states = [read_json(root / "states" / (job_id + ".json")) for job_id in ids]
    if not all(x["status"] == "success" for x in states):
        return False
    value = dict(
        schema=SCHEMA, checked_at=now(), status="passed", jobs=ids,
        results=sum(x["expected_results"] for x in states),
        acceptance="three sigmas and both history modes; full native archive validation, metadata and 17 chunks",
        validations={x["job_id"]: x["validation"] for x in states},
        throughput=progress(root, plan),
    )
    try:
        write_new(root / "pilot_validation.json", value)
    except FileExistsError:
        pass
    return True


def gpu_snapshot():
    records = checked(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                       "--format=csv,noheader,nounits"])
    gpus = {}
    for line in records.splitlines():
        index, uuid, used, total, utilization = [x.strip() for x in line.split(",")]
        gpus[int(index)] = dict(uuid=uuid, memory_used_mib=int(used), memory_total_mib=int(total),
                               utilization_percent=int(utilization), processes=[])
    processes = checked(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                         "--format=csv,noheader,nounits"])
    for line in processes.splitlines():
        if not line.strip() or line.startswith("No running"):
            continue
        uuid, pid, memory = [x.strip() for x in line.split(",")]
        for record in gpus.values():
            if record["uuid"] == uuid:
                record["processes"].append(dict(pid=pid, used_gpu_memory=memory))
    return gpus


def require_idle(gpus, indexes):
    for index in indexes:
        if index not in gpus:
            raise ValueError(f"GPU {index} is absent")
        value = gpus[index]
        if value["processes"] or value["utilization_percent"] > 0 or value["memory_used_mib"] > 256:
            raise RuntimeError(f"GPU {index} is occupied: {value}")


def cpu_resources():
    available = len(os.sched_getaffinity(0))
    quota = None
    path = Path("/sys/fs/cgroup/cpu.max")
    if path.is_file():
        limit, period = path.read_text().split()
        if limit != "max":
            quota = int(limit) / int(period)
            available = min(available, quota)
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemAvailable", "MemTotal"):
            memory[key] = int(value.split()[0]) * 1024
    return dict(affinity_cpus=len(os.sched_getaffinity(0)), quota_cpus=quota,
                effective_cpus=available, memory=memory, load_average=list(os.getloadavg()))


def sampler_command(plan, binding, task):
    return [
        str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=1",
        "-m", "cosmos3_joint_video_hand_pose.src.ar_v031_eval", "sample",
        "--ckpt", binding["checkpoint"], "--toml", plan["toml"],
        "--training-snapshot", binding["snapshot"],
        "--episodes-manifest", plan["episodes_manifest"], "--segments-manifest", plan["segments_manifest"],
        "--eval-windows", task["fragment_manifest"], "--split", task["split"],
        "--history", *task["histories"], "--chunk-size", "4", "--video-shift", "5",
        "--sigma-small", str(task["sigma_small"]), "--output", task["output"],
    ]


def stop_sampler(process):
    """Stop only the sampler process group created by this worker."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def validate_archives(task, binding):
    # Reuse the stable archive validator, including raw GT, finite 57D values,
    # padding zeros, dense RGB coverage and projection transforms.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from cosmos3_joint_video_hand_pose.src.ar_v02_eval import load_rollout

    output = Path(task["output"])
    run = read_json(output / "run.json")
    if set(run["archives"]) != set(task["expected_archives"]):
        raise ValueError("archive list does not match this task's modes")
    reports = []
    for history, path in zip(task["histories"], task["expected_archives"]):
        layout, meta, arrays = load_rollout(path)
        expected = dict(
            model_version=MODEL_VERSION, model_target=MODEL_TARGET,
            mask_prefix_loss=True, prefix_loss_denominator=PREFIX_LOSS_DENOMINATOR,
            prefix_loss_mask_scope=PREFIX_LOSS_MASK_SCOPE,
            training_snapshot_sha256=binding["snapshot_sha256"],
            checkpoint_metadata_sha256=binding["checkpoint_metadata_sha256"],
            recipe_reference_snapshot_sha256=binding["recipe_reference_snapshot_sha256"], schema="ar_v02_rollout_v1", layout_version="joint_chunk_cond_v1",
            history=history, sample_id=task["window"]["sample_id"], source_offset=task["window"]["start"],
            seed=task["window"]["seed"], chunk_size=4, num_frames=69, steps=30,
            sigma_small=task["sigma_small"], history_video_sigma=task["sigma_small"],
            history_action_sigma=task["sigma_small"], eval_windows_sha256=task["fragment_manifest_sha256"],
            selected_windows=1, frozen_windows=1, checkpoint=binding["checkpoint"],
            video_shift=5.0, action_shift=5.0, video_guidance=1.0, action_guidance=1.0,
            action_representation=binding["action_representation"],
        )
        for name, value in expected.items():
            if meta.get(name) != value:
                raise ValueError(f"archive {path}: {name}={meta.get(name)!r}, expected {value!r}")
        for name in ("state_normalizer", "future_normalizer"):
            if meta[name]["sha256"] != binding["frozen_artifacts"][name]["sha256"]:
                raise ValueError("archive normalizer does not match the bound checkpoint")
        for side in ("right", "left"):
            if meta["hand_codecs"][side]["sha256"] != binding["frozen_artifacts"][side + "_codec"]["sha256"]:
                raise ValueError("archive codec does not match the bound checkpoint")
        chunks, conditions = meta["chunk_reports"], meta["condition_reports"]
        if len(layout.boundaries) != 17 or len(chunks) != 17 or len(conditions) != 17:
            raise ValueError("archive must contain all 17 chunks and conditions")
        for index, (chunk, condition) in enumerate(zip(chunks, conditions), 1):
            for name, value in dict(chunk=index, denoise_steps=30, forward_calls=32, noisy_calls=30,
                                    condition_prefill_calls=1, clean_refresh_calls=0,
                                    noisy_refresh_calls=1, cache_mode="persistent",
                                    includes_reference_checks=False, action_count=32).items():
                if chunk.get(name) != value:
                    raise ValueError(f"archive chunk {index}: invalid {name}")
            if condition["chunk"] != index or condition["episode_boundary_source_index"] != task["window"]["start"] + 32 * (index - 1):
                raise ValueError("condition boundary does not match the frozen window")
            if not math.isfinite(chunk["end_to_end_seconds"]) or chunk["end_to_end_seconds"] <= 0:
                raise ValueError("invalid chunk runtime")
        required = {"raw_gt_keypoints", "raw_gt_camera_poses", "raw_gt_source_indexes", "gt_rgb",
                    "generated_rgb", "generated_offsets", "intrinsics"}
        if not required <= arrays.keys():
            raise ValueError("archive is missing raw GT or video payload")
        reports.append(dict(
            archive=path, sha256=digest(path), bytes=Path(path).stat().st_size, history=history,
            sigma_small=task["sigma_small"], chunks=17, source_offset=meta["source_offset"],
            sampler_seconds=sum(x["end_to_end_seconds"] for x in chunks),
            peak_memory_bytes=max(x["peak_memory_bytes"] for x in chunks),
            parent_manifest_sha256=task["parent_manifest_sha256"],
            fragment_manifest_sha256=task["fragment_manifest_sha256"],
        ))
        del arrays
    return reports


def event(handle, name, **fields):
    value = {"time": now(), "event": name, **fields}
    handle.write(json.dumps(value, allow_nan=False) + "\n")
    handle.flush()
    print(json.dumps(value, allow_nan=False), flush=True)


def worker(root, stage, gpu, node_label, timeout_seconds):
    if pwd.getpwuid(os.getuid()).pw_name != "lzh":
        raise RuntimeError("GPU workers must use lzh")
    root, plan = load_plan(root)
    binding = load_binding(root)
    if stage == "remaining" and not pilot_complete(root, plan):
        raise RuntimeError("remaining jobs require a successful three-sigma dual-mode pilot")
    worker_id = f"{node_label}_gpu{gpu}_{os.getpid()}_{time.time_ns()}"
    with (root / "workers" / (worker_id + ".jsonl")).open("x") as events:
        for task in plan["tasks"]:
            if task["stage"] != stage or (root / "STOP").exists():
                continue
            state_path = root / "states" / (task["job_id"] + ".json")
            if read_json(state_path)["status"] != "pending":
                continue
            claim = root / "claims" / task["job_id"]
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            state = dict(
                job_id=task["job_id"], status="running", expected_results=len(task["histories"]),
                started_at=now(), started_epoch=time.time(), worker=worker_id,
                node_label=node_label, hostname=socket.gethostname(), gpu=gpu,
                log=str(root / "logs" / (task["job_id"] + ".log")),
            )
            replace_json(state_path, state)
            event(events, "job_started", job_id=task["job_id"], **progress(root, plan))
            process = None
            try:
                require_idle(gpu_snapshot(), [gpu])
                if Path(task["output"]).exists():
                    raise ValueError("task output already exists; refusing to overwrite/retry")
                command = sampler_command(plan, binding, task)
                env = os.environ.copy()
                for name in list(env):
                    if name in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK"} or name.startswith("TORCHELASTIC_"):
                        env.pop(name)
                env.pop("PYTHONSAFEPATH", None)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), LD_LIBRARY_PATH="", PYTHONPATH=f"{REPO}:{REPO}/packages/cosmos3",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                state["command"] = command
                replace_json(state_path, state)
                with Path(state["log"]).open("x") as log:
                    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    state["child_pid"] = process.pid
                    replace_json(state_path, state)
                    last_report = time.monotonic()
                    while process.poll() is None:
                        time.sleep(1)
                        elapsed = time.time() - state["started_epoch"]
                        if elapsed > timeout_seconds:
                            stop_sampler(process)
                            raise TimeoutError("sampler exceeded explicit task timeout; no automatic retry")
                        if time.monotonic() - last_report >= 30:
                            event(events, "job_heartbeat", job_id=task["job_id"], job_elapsed_seconds=elapsed,
                                  gpu_usage=gpu_snapshot().get(gpu), **progress(root, plan))
                            last_report = time.monotonic()
                state["exit_code"] = process.returncode
                if process.returncode != 0:
                    raise RuntimeError(f"sampler exited {process.returncode}; inspect {state['log']}")
                state["validation"] = validate_archives(task, binding)
                state["status"] = "success"
            except (Exception, KeyboardInterrupt) as exc:
                if process is not None:
                    stop_sampler(process)
                state["status"] = "failed"
                state["error"] = f"{type(exc).__name__}: {exc}"
                state.setdefault("exit_code", process.returncode if process is not None else None)
                try:
                    write_new(root / "STOP", dict(time=now(), job_id=task["job_id"], error=state["error"]))
                except FileExistsError:
                    pass
            state["finished_at"] = now()
            state["wall_seconds"] = time.time() - state["started_epoch"]
            replace_json(state_path, state)
            event(events, "job_finished", job_id=task["job_id"], status=state["status"],
                  exit_code=state["exit_code"], wall_seconds=state["wall_seconds"], **progress(root, plan))
            if state["status"] != "success":
                return 1
        event(events, "worker_finished", worker=worker_id, **progress(root, plan))
    return 0


def supervise(root, stage, indexes, node_label, timeout_seconds):
    if pwd.getpwuid(os.getuid()).pw_name != "lzh":
        raise RuntimeError("node supervisor must use lzh")
    root, plan = load_plan(root)
    load_binding(root)
    if stage == "remaining" and not pilot_complete(root, plan):
        raise RuntimeError("pilot has not passed")
    if not indexes or len(set(indexes)) != len(indexes):
        raise ValueError("GPU indexes must be nonempty and unique")
    resources, gpus = cpu_resources(), gpu_snapshot()
    require_idle(gpus, indexes)
    if resources["effective_cpus"] < len(indexes) or resources["memory"]["MemAvailable"] < len(indexes) * (8 << 30):
        raise RuntimeError("insufficient CPU quota or available memory for the requested workers")
    locks, children = [], []
    receipt = dict(time=now(), stage=stage, node_label=node_label, hostname=socket.gethostname(),
                   gpu_indexes=indexes, cpu=resources, gpu=gpus, action="explicit_local_node_GPU_launch")
    write_new(root / "workers" / f"{node_label}_{stage}_{os.getpid()}_{time.time_ns()}_launch.json", receipt)
    try:
        for gpu in indexes:
            lock = root / "gpu_locks" / f"{node_label}_gpu{gpu}"
            lock.mkdir()
            locks.append(lock)
        for gpu in indexes:
            command = [str(PYTHON), str(Path(__file__).resolve()), "worker", "--root", str(root),
                       "--stage", stage, "--gpu", str(gpu), "--node-label", node_label,
                       "--timeout-seconds", str(timeout_seconds)]
            children.append(subprocess.Popen(command, cwd=REPO))
        codes = [child.wait() for child in children]
        print(json.dumps(dict(node_label=node_label, worker_exit_codes=codes, **progress(root, plan))), flush=True)
        if stage == "pilot":
            pilot_complete(root, plan)
        return 0 if all(code == 0 for code in codes) else 1
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                child.wait()
        # Normal exit releases only this supervisor's own local-node leases.
        # Per-job claims remain durable; failed/stale jobs are never auto-retried.
        for lock in locks:
            lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="CPU: freeze task fragments, create no GPU processes")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--scope", choices=("short", "formal"), default="short")
    bind_parser = sub.add_parser("bind", help="CPU: pin a completed prefix-enabled checkpoint and contract")
    bind_parser.add_argument("--root", type=Path, required=True)
    bind_parser.add_argument("--checkpoint", type=Path, required=True)
    bind_parser.add_argument("--snapshot", type=Path, required=True)
    status_parser = sub.add_parser("status", help="CPU: report progress and pilot acceptance")
    status_parser.add_argument("--root", type=Path, required=True)
    for name in ("supervise", "worker"):
        run = sub.add_parser(name, help="explicit GPU launch on this node only; no cross-node SSH")
        run.add_argument("--root", type=Path, required=True)
        run.add_argument("--stage", choices=("pilot", "remaining"), required=True)
        run.add_argument("--node-label", required=True)
        run.add_argument("--timeout-seconds", type=int, default=3600)
        run.add_argument("--gpus" if name == "supervise" else "--gpu", required=True)
    args = parser.parse_args()
    if args.command in ("worker", "supervise"):
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"scan interrupted by signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
    if args.command == "prepare":
        plan = prepare(args.root, args.scope)
        print(json.dumps(dict(root=plan["root"], jobs=plan["jobs"], results=plan["expected_results"],
                              status=plan["launch_status"])))
    elif args.command == "bind":
        print(json.dumps(bind(args.root, args.checkpoint, args.snapshot), indent=2))
    elif args.command == "status":
        root, plan = load_plan(args.root)
        print(json.dumps(dict(pilot_passed=pilot_complete(root, plan), **progress(root, plan)), indent=2))
    elif args.command == "worker":
        return worker(args.root, args.stage, int(args.gpu), args.node_label, args.timeout_seconds)
    else:
        return supervise(args.root, args.stage, [int(x) for x in args.gpus.split(",")],
                         args.node_label, args.timeout_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
