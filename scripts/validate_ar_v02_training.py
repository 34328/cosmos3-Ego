#!/usr/bin/env python3
"""Isolated fixed-input Nano/FSDP comparison or authorized compact 50+1 acceptance."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TOML = REPO / "cosmos3_joint_video_hand_pose/configs/ar_v0_2_c.toml"


def gpucheck(run, phase):
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    (run / (phase + ".gpus.csv")).write_text(gpu)
    (run / (phase + ".processes.csv")).write_text(processes)
    rows = [line.split(",") for line in gpu.strip().splitlines()]
    assert {int(r[0]) for r in rows} == set(range(8)), "eight GPUs required"
    for index, uuid, used, total, utilization in rows:
        assert int(used) <= 1024 and int(utilization) <= 5, f"GPU {index} busy"
        assert int(total) - int(used) >= 65000, f"GPU {index} insufficient memory"
    assert not processes.strip(), "GPU compute processes are present"


def source_manifest():
    paths = list((REPO / "cosmos3_joint_video_hand_pose/src").glob("*.py"))
    paths += list((REPO / "packages/cosmos3/cosmos_framework/model/generator/mot").glob("*.py"))
    paths += list((REPO / "packages/cosmos3/cosmos_framework/data/generator/sequence_packing").glob("*.py"))
    paths += [TOML, REPO / "packages/cosmos3/cosmos_framework/trainer/__init__.py"]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def verify_result(path, loaded, updates):
    data = json.loads(path.read_text())
    assert data["status"] == "success" and data["world_size"] == 8
    assert data["loaded_iteration"] == loaded
    assert data["optimizer_updates"] == updates and data["microsteps"] == updates
    assert data["final_iteration"] == loaded + updates
    assert {r["rank"] for r in data["ranks"]} == set(range(8))
    for rank in data["ranks"]:
        assert len(rank["steps"]) == updates
        for step in rank["steps"]:
            assert step["finite"] and step["raw_gradients_finite"]
            assert step["gradient_tensors_checked_global"] > 0 and step["grad_accum_position"] == 0
            assert math.isfinite(step["loss"])
    return data


def compare_fixed(left, right, *, rtol=0.005, atol=1e-7):
    errors, records = [], []
    for a, b in zip(
        sorted(left["ranks"], key=lambda r: r["rank"]), sorted(right["ranks"], key=lambda r: r["rank"]), strict=True
    ):
        rank = a["rank"]
        if not a["initial_parameters_sha256"] or a["initial_parameters_sha256"] != b["initial_parameters_sha256"]:
            errors.append(f"rank {rank}: initial parameters differ")
        for step, (x, y) in enumerate(zip(a["steps"], b["steps"], strict=True)):
            for key in ("input_audit", "denoise_input_audit", "rng_sha256"):
                if x[key] != y[key]:
                    errors.append(f"rank {rank} step {step}: {key} mismatch")
            for key in ("loss", "video_loss", "action_loss"):
                if not math.isclose(x[key], y[key], rel_tol=rtol, abs_tol=atol):
                    errors.append(f"rank {rank} step {step}: {key} mismatch")
            gx = json.loads(Path(x["gradient_audit_path"]).read_text())
            gy = json.loads(Path(y["gradient_audit_path"]).read_text())
            assert gx["parameters"].keys() == gy["parameters"].keys()
            mismatches, max_relative = [], 0.0
            for name, p in gx["parameters"].items():
                q = gy["parameters"][name]
                if p["present"] != q["present"]:
                    errors.append(f"rank {rank} step {step}: gradient presence {name}")
                    continue
                if not p["present"]:
                    continue
                if p["sha256"] != q["sha256"]:
                    mismatches.append(name)
                delta = abs(p["global_l2"] - q["global_l2"])
                max_relative = max(max_relative, delta / max(p["global_l2"], q["global_l2"], atol))
                if not math.isclose(p["global_l2"], q["global_l2"], rel_tol=rtol, abs_tol=atol):
                    errors.append(f"rank {rank} step {step}: gradient norm {name}")
            for name, norm in gx["layers"].items():
                if not math.isclose(norm, gy["layers"][name], rel_tol=rtol, abs_tol=atol):
                    errors.append(f"rank {rank} step {step}: layer norm {name}")
            if not math.isclose(gx["global_l2"], gy["global_l2"], rel_tol=rtol, abs_tol=atol):
                errors.append(f"rank {rank} step {step}: global norm mismatch")
            records.append(
                dict(
                    rank=rank,
                    step=step,
                    norm_a=gx["global_l2"],
                    norm_b=gy["global_l2"],
                    max_parameter_norm_relative_difference=max_relative,
                    differing_gradient_digests=mismatches,
                )
            )
    return dict(
        status="failed" if errors else "success",
        rtol=rtol,
        atol=atol,
        exact_gradient_digests=not any(r["differing_gradient_digests"] for r in records),
        errors=errors,
        records=records,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fixed", "resume50"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--compact-ready", action="store_true", help="Set only after main controller confirms compact pass wiring ready"
    )
    args = parser.parse_args()
    if args.mode == "resume50" and not args.compact_ready:
        parser.error("50 updates require the main controller's compact-ready confirmation")
    assert os.uname().nodename == "nb-1678910729611554048-cr786mpgj9xc", "Tdebug5 only"
    run = args.run_dir.resolve()
    run.mkdir(parents=False, exist_ok=False)
    env = dict(
        os.environ,
        IMAGINAIRE_OUTPUT_ROOT=str(run / "training"),
        BASE_CHECKPOINT_PATH="/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp",
        LD_LIBRARY_PATH="",
        PYTHONPATH=".:packages/cosmos3",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    (run / "recipe_snapshot.toml").write_bytes(TOML.read_bytes())
    baseline = source_manifest()
    (run / "source_manifest.json").write_text(json.dumps(baseline, indent=2))
    subprocess.run(["git", "diff", "--binary"], cwd=REPO, stdout=(run / "workspace.diff").open("w"), check=True)
    subprocess.run(["git", "status", "--short"], cwd=REPO, stdout=(run / "workspace.status").open("w"), check=True)

    def stage(label, job, steps, loaded, *extra):
        assert source_manifest() == baseline, "implementation changed between stages; use a fresh run"
        gpucheck(run, "before_" + label)
        (run / "status").write_text(label + "\n")
        output = run / (label + ".json")
        command = [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=8",
            "--max-restarts=0",
            "-m",
            "cosmos3_joint_video_hand_pose.src.smoke_train",
            "--toml",
            str(TOML),
            "--wandb-mode",
            "disabled",
            "--job-name",
            job,
            "--steps",
            str(steps),
            "--expect-resume-iteration",
            str(loaded),
            "--output",
            str(output),
            *map(str, extra),
        ]
        (run / (label + ".command.json")).write_text(json.dumps(command))
        started = time.time()
        with (run / (label + ".log")).open("x") as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        (run / (label + ".elapsed_seconds")).write_text(str(time.time() - started))
        assert source_manifest() == baseline, "implementation changed during stage; cannot compare"
        return verify_result(output, loaded, steps)

    try:
        if args.mode == "fixed":
            a = stage(
                "capture", "fixed_capture", 2, 0, "--capture-fixed-inputs", run / "fixed_inputs", "--audit-gradients"
            )
            b = stage(
                "replay", "fixed_replay", 2, 0, "--replay-fixed-inputs", run / "fixed_inputs", "--audit-gradients"
            )
            summary = compare_fixed(a, b)
            (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            assert summary["status"] == "success", f"fixed comparison failed: {summary['errors'][:5]}"
        else:
            a = stage("stage1", "resume_contract_50plus1", 50, 0, "--save-final")
            job = run / "training/joint_video_hand_pose/ar_v0_2/resume_contract_50plus1"
            checkpoint = job / "checkpoints/iter_000000050"
            assert a["checkpoint_saved"]
            assert (job / "checkpoints/latest_checkpoint.txt").read_text().strip() == "iter_000000050"
            for part in ("model", "optim", "scheduler", "trainer"):
                assert (checkpoint / part / ".metadata").is_file(), part
            for rank in range(8):
                assert (checkpoint / "dataloader" / f"rank_{rank}.pkl").is_file()
            b = stage("stage2", "resume_contract_50plus1", 1, 50)
            for rank in range(8):
                trace = job / "dataloader_trace" / f"rank_{rank:05d}.jsonl"
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                assert [r["iteration"] for r in rows] == list(range(51))
            summary = dict(
                status="success",
                first_process_exited=True,
                optimizer_updates_before_save=50,
                loaded_iteration=50,
                final_iteration=51,
                checkpoint=str(checkpoint),
                world_size=8,
            )
            (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (run / "status").write_text("SUCCESS\n")
    except BaseException as error:
        (run / "failure.txt").write_text(repr(error) + "\n")
        (run / "status").write_text("FAILED\n")
        raise


if __name__ == "__main__":
    main()
