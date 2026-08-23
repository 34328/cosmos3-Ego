#!/usr/bin/env bash
set -euo pipefail

# Native Cosmos has no safe in-process inference callback: training owns all
# GPUs. Run each native training interval, release the ranks, infer, then resume
# from the complete DCP checkpoint.
cd /mnt/lzh/cosmos
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly ROOT=/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose
readonly OUTPUT_ROOT=/mnt/lzh/cosmos/outputs
readonly TRAIN_ROOT="$OUTPUT_ROOT/joint_video_hand_pose/train/overfit_v0.0"
readonly BASE_CHECKPOINT=/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
readonly WAN_VAE=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
readonly CONFIG="$ROOT/configs/overfit_v0_0.toml"

[[ ! -e "$TRAIN_ROOT" ]] || { echo "refusing to overwrite $TRAIN_ROOT" >&2; exit 2; }
PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3 "$PYTHON" \
  "$ROOT/artifacts/cosmos3_action_contract/v2/validate_manifest.py"

export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WAN_VAE_PATH="$WAN_VAE"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"

previous="$BASE_CHECKPOINT"
for target in 300 600 900 1200 1500 1800 2000; do
  echo "=== overfit_v0.0 native training stage: target=$target, init=$previous ==="
  if [[ "$previous" == "$BASE_CHECKPOINT" ]]; then
    load_args=("checkpoint.load_path=$BASE_CHECKPOINT" "checkpoint.load_training_state=false")
  else
    load_args=("checkpoint.load_path=$previous" "checkpoint.load_training_state=true")
  fi
  "$TORCHRUN" --nproc-per-node=8 --master-port="$((29556 + target / 300))" \
    -m cosmos3_joint_video_hand_pose.src.train --sft-toml "$CONFIG" \
    "trainer.max_iter=$target" "scheduler.cycle_lengths=[2000]" "${load_args[@]}"
  checkpoint="$TRAIN_ROOT/checkpoints/iter_$(printf '%09d' "$target")"
  [[ -f "$checkpoint/model/.metadata" ]] || { echo "missing complete checkpoint: $checkpoint" >&2; exit 3; }
  "$ROOT/scripts/run_overfit_v0_0_replays.sh" "$(printf '%09d' "$target")"
  previous="$checkpoint"
done
