#!/usr/bin/env bash
# Run only after the user releases all eight Tdebug5 GPUs for this check.
# Two separate torchrun processes: 2 updates/save/exit, then restore + 1 update.
set -Eeuo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AR_V02_PYTHON:-/home/lzh/miniconda3/envs/cosmos3/bin/python}"
TOML="$REPO/cosmos3_joint_video_hand_pose/configs/ar_v0_2_c.toml"
BASE="$REPO/outputs/joint_video_hand_pose/ar_v0_2"
JOB=resume_contract_2plus1

if [[ "${1:-}" != --run-authorized ]]; then
    echo "Prepared only. After explicit GPU release:"
    echo "bash $REPO/scripts/check_ar_v02_resume.sh --run-authorized <unique-run-id>"
    exit 0
fi
RUN_ID="${2:-resume_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid run id" >&2; exit 2; }
[[ "$(hostname)" == nb-1678910729611554048-cr786mpgj9xc ]] || {
    echo "This script is pinned to the verified Tdebug5 host" >&2; exit 2;
}
cd "$REPO"
[[ -x "$PYTHON" && -f "$TOML" ]] || exit 2
RUN_DIR="$BASE/$RUN_ID"
# Atomic reservation: never reuse or overwrite an earlier attempt.
mkdir "$RUN_DIR"
export IMAGINAIRE_OUTPUT_ROOT="$RUN_DIR/training"
export BASE_CHECKPOINT_PATH=/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp
export LD_LIBRARY_PATH=''
export PYTHONPATH=.:packages/cosmos3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
JOB_DIR="$IMAGINAIRE_OUTPUT_ROOT/joint_video_hand_pose/ar_v0_2/$JOB"
STAGE=preflight
trap 'code=$?; if (( code != 0 )); then printf "FAILED %s exit=%s\n" "$STAGE" "$code" > "$RUN_DIR/status"; fi' EXIT

gpucheck() {
    local phase="$1"
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > "$RUN_DIR/$phase.gpus.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$RUN_DIR/$phase.processes.csv"
    "$PYTHON" - "$RUN_DIR/$phase.gpus.csv" "$RUN_DIR/$phase.processes.csv" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
assert {int(r[0]) for r in rows} == set(range(8)), "expected eight GPUs"
for index, uuid, used, total, utilization in rows:
    assert int(used) <= 1024 and int(utilization) <= 5, f"GPU {index} busy: memory={used}, utilization={utilization}"
    assert int(total) - int(used) >= 65000, f"GPU {index} has insufficient free memory"
processes = open(sys.argv[2]).read().strip()
assert not processes, f"GPU compute processes still present:\n{processes}"
print("GPUCHECK_OK: all eight GPUs idle")
PY
}

verify_result() {
    "$PYTHON" - "$RUN_DIR/$1.json" "$2" "$3" "$4" <<'PY'
import json, math, sys
p, loaded, updates, final = sys.argv[1], *map(int, sys.argv[2:])
data = json.load(open(p))
assert data["status"] == "success" and data["world_size"] == 8
assert data["loaded_iteration"] == loaded
assert data["optimizer_updates"] == updates and data["microsteps"] == updates
assert data["final_iteration"] == final
assert len(data["ranks"]) == 8
assert {r["rank"] for r in data["ranks"]} == set(range(8))
for rank in data["ranks"]:
    assert len(rank["steps"]) == updates
    for step in rank["steps"]:
        assert step["finite"] and step["raw_gradients_finite"]
        assert step["gradient_tensors_checked_global"] > 0
        assert step["grad_accum_position"] == 0
        assert math.isfinite(step["loss"])
print(f"RESULT_OK {p}: {loaded} -> {final}")
PY
}

COMMON=(--standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0
        -m cosmos3_joint_video_hand_pose.src.smoke_train
        --toml "$TOML" --wandb-mode disabled --job-name "$JOB")
printf '%s\n' "run=$RUN_ID" "job=$JOB" "output_root=$IMAGINAIRE_OUTPUT_ROOT"     "toml=$TOML" "gpus=$CUDA_VISIBLE_DEVICES" > "$RUN_DIR/run_manifest.txt"
cp "$TOML" "$RUN_DIR/recipe_snapshot.toml"

gpucheck before_stage1
STAGE=stage1_save
printf '%s\n' "$STAGE" > "$RUN_DIR/status"
"$PYTHON" -m torch.distributed.run "${COMMON[@]}"     --steps 2 --save-final --expect-resume-iteration 0     --output "$RUN_DIR/stage1.json" > "$RUN_DIR/stage1.log" 2>&1
# Reaching here proves the whole first torchrun process exited successfully.
verify_result stage1 0 2 2
"$PYTHON" - "$JOB_DIR" "$RUN_DIR/stage1.json" <<'PY'
import json, sys
from pathlib import Path
job = Path(sys.argv[1])
assert json.load(open(sys.argv[2]))["checkpoint_saved"] is True
root = job / "checkpoints"
assert (root / "latest_checkpoint.txt").read_text().strip() == "iter_000000002"
saved = root / "iter_000000002"
for component in ("model", "optim", "scheduler", "trainer"):
    assert (saved / component / ".metadata").is_file(), f"missing {component} DCP metadata"
for rank in range(8):
    assert (saved / "dataloader" / f"rank_{rank}.pkl").is_file(), f"missing rank {rank} loader state"
print(f"CHECKPOINT_COMPONENTS_OK {saved}")
PY

# No auto-wait, takeover or process termination: fail if GPUs are occupied.
gpucheck before_stage2
STAGE=stage2_resume
printf '%s\n' "$STAGE" > "$RUN_DIR/status"
"$PYTHON" -m torch.distributed.run "${COMMON[@]}"     --steps 1 --expect-resume-iteration 2     --output "$RUN_DIR/stage2.json" > "$RUN_DIR/stage2.log" 2>&1
verify_result stage2 2 1 3
"$PYTHON" - "$RUN_DIR" "$JOB_DIR" <<'PY'
import json, sys
from pathlib import Path
run, job = map(Path, sys.argv[1:])
for rank in range(8):
    records = [json.loads(s) for s in (job / "dataloader_trace" / f"rank_{rank:05d}.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in records] == [0, 1, 2], f"rank {rank}: inconsistent resumed iterations"
summary = dict(status="success", first_process_exited=True, loaded_iteration=2,
               final_iteration=3, world_size=8, checkpoint=str(job / "checkpoints/iter_000000002"),
               stage1=str(run / "stage1.json"), stage2=str(run / "stage2.json"))
(run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
PY
printf '%s\n' SUCCESS > "$RUN_DIR/status"
