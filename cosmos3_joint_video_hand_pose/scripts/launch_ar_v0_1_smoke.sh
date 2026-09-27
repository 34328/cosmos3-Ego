#!/usr/bin/env bash
# AR v0.1 smoke: a few real packed training steps on 8 GPUs, no checkpoint writes.
# Usage: launch_ar_v0_1_smoke.sh [recipe.toml]   (default: configs/ar_v0_1.toml)
set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/cosmos3"
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly CONFIG="${1:-$REPO_ROOT/cosmos3_joint_video_hand_pose/configs/ar_v0_1.toml}"
readonly RUN_NAME="smoke_$(basename "$CONFIG" .toml)_$(date +%Y%m%d_%H%M%S)"
readonly OUT="$REPO_ROOT/outputs/joint_video_hand_pose/ar/smoke/$RUN_NAME"

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export IMAGINAIRE_OUTPUT_ROOT="$OUT/imaginaire"
export LD_LIBRARY_PATH=''

"$TORCHRUN" --nproc-per-node="${NPROC:-8}" --master-port="${MASTER_PORT:-29581}" \
  -m cosmos3_joint_video_hand_pose.src.smoke_train \
  --toml "$CONFIG" --steps "${STEPS:-4}" --wandb-mode disabled --job-name "$RUN_NAME" \
  --output "$OUT/smoke_result.json" 2>&1 | tee "$OUT/smoke.log"
