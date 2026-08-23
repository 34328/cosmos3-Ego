#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly ROOT=/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose
readonly JOB_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/overfit/overfit_v0.2_lr_balanced
readonly BASE_CHECKPOINT=/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
readonly WAN_VAE=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
readonly CONFIG="$ROOT/configs/overfit_v0_2_lr_balanced.toml"

export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT"
export WAN_VAE_PATH="$WAN_VAE"
export IMAGINAIRE_OUTPUT_ROOT=/mnt/lzh/cosmos/outputs

latest=0
for candidate in "$JOB_ROOT"/checkpoints/iter_*; do
  [[ -d "$candidate" ]] || continue
  complete=true
  for component in model optim scheduler trainer; do
    [[ -f "$candidate/$component/.metadata" ]] || complete=false
  done
  if [[ "$complete" == true ]]; then
    iteration=$((10#${candidate##*_}))
    ((iteration > latest)) && latest=$iteration
  fi
done
((latest > 0)) || { echo "no complete checkpoint found under $JOB_ROOT/checkpoints" >&2; exit 2; }

previous="$JOB_ROOT/checkpoints/iter_$(printf '%09d' "$latest")"
"$ROOT/scripts/run_overfit_v0_2_lr_balanced_replays.sh" "$(printf '%09d' "$latest")"
for target in 600 900 1200 1500 1800 2000; do
  ((target > latest)) || continue
  echo "=== overfit_v0.2_lr_balanced resume: target=$target, init=$previous ==="
  "$TORCHRUN" --nproc-per-node=8 --master-port="$((29556 + target / 300))" \
    -m cosmos3_joint_video_hand_pose.src.train --sft-toml "$CONFIG" \
    "trainer.max_iter=$target" "scheduler.cycle_lengths=[2000]" \
    "checkpoint.load_path=$previous" "checkpoint.load_training_state=true"
  checkpoint="$JOB_ROOT/checkpoints/iter_$(printf '%09d' "$target")"
  [[ -f "$checkpoint/model/.metadata" ]] || { echo "missing complete checkpoint: $checkpoint" >&2; exit 3; }
  "$ROOT/scripts/run_overfit_v0_2_lr_balanced_replays.sh" "$(printf '%09d' "$target")"
  previous="$checkpoint"
done
