#!/usr/bin/env bash
# Run one side of the same-node 20-step comparison through the official CLI.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="${1:?expected v03 or v02}"
: "${COMPARE_ROOT:?set the same absolute independent COMPARE_ROOT for both runs}"
[[ "$COMPARE_ROOT" = "$PROJECT_ROOT/outputs/"* ]] || exit 2
case "$MODE" in
    v03) LAUNCHER="$SCRIPT_DIR/launch_ar_v0_3.sh"; EXPECTED=1; RECIPE=ar_v0_3.toml ;;
    v02) LAUNCHER="$SCRIPT_DIR/launch_ar_v0_2.sh"; EXPECTED=2; RECIPE=ar_v0_2_video_lr1e4.toml ;;
    *) exit 2 ;;
esac
mkdir -p "$COMPARE_ROOT/$MODE"
[[ ! -e "$COMPARE_ROOT/$MODE/started.txt" ]] || { echo 'refusing duplicate run'; exit 2; }
nvidia-smi > "$COMPARE_ROOT/$MODE/nvidia-smi-before.txt"
ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$ACTIVE" ]] || { echo 'GPU compute processes present; select another idle node'; exit 3; }
date -u +%FT%TZ > "$COMPARE_ROOT/$MODE/started.txt"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$COMPARE_ROOT/$MODE/exit.txt"' EXIT
export PATH="/home/lzh/miniconda3/envs/cosmos3/bin:$PATH"
export LD_LIBRARY_PATH='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1 NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export WORKDIR="$PROJECT_ROOT" TOML_FILE="$PROJECT_ROOT/cosmos3_joint_video_hand_pose/configs/$RECIPE"
export NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=51503
export OUTPUT_ROOT="$COMPARE_ROOT/$MODE" IMAGINAIRE_OUTPUT_ROOT="$COMPARE_ROOT"
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=20 trainer.grad_accum_iter=1 checkpoint.save_iter=20 model.config.parallelism.data_parallel_replicate_degree=1 job.wandb_mode=disabled job.group=ar_v0_3_short_compare job.name=$MODE scheduler.warm_up_steps=[100] scheduler.cycle_lengths=[3000] scheduler.f_start=[0.0] scheduler.f_max=[1.0] scheduler.f_min=[0.3] +trainer.callbacks.ar_v03_smoke={_target_:cosmos3_joint_video_hand_pose.src.ar_v03_smoke_monitor.ARV03SmokeMonitor,run_label:$MODE,expected_forwards:$EXPECTED}"
if [[ "$MODE" = v02 ]]; then
    EXTRA_TAIL_OVERRIDES="$EXTRA_TAIL_OVERRIDES +trainer.callbacks.ar_v03_lr_receipt={_target_:cosmos3_joint_video_hand_pose.src.ar_v03_config.ARV03LearningRateReceiptCallback}"
fi
bash "$LAUNCHER"
date -u +%FT%TZ > "$COMPARE_ROOT/$MODE/completed.txt"
