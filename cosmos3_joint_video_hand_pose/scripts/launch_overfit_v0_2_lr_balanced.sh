#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly ROOT=/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose
readonly OUTPUT_ROOT=/mnt/lzh/cosmos/outputs
readonly JOB_ROOT="$OUTPUT_ROOT/joint_video_hand_pose/overfit/overfit_v0.2_lr_balanced"
readonly BASE_CHECKPOINT=/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
readonly WAN_VAE=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
readonly CONFIG="$ROOT/configs/overfit_v0_2_lr_balanced.toml"

[[ ! -e "$JOB_ROOT" ]] || { echo "refusing to overwrite $JOB_ROOT" >&2; exit 2; }
PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3 "$PYTHON" \
  "$ROOT/artifacts/cosmos3_action_contract/v2/validate_manifest.py"

export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT"
export WAN_VAE_PATH="$WAN_VAE"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"

if [[ "${COSMOS_OVERFIT_DIRECT:-0}" == "1" ]]; then
  exec "$TORCHRUN" --nproc-per-node=8 --master-port="${MASTER_PORT:-29556}" \
    -m cosmos3_joint_video_hand_pose.src.train --sft-toml "$CONFIG"
fi

previous="$BASE_CHECKPOINT"
for target in 300 600 900 1200 1500 1800 2000; do
  echo "=== overfit_v0.2_lr_balanced: target=$target, init=$previous ==="
  if [[ "$previous" == "$BASE_CHECKPOINT" ]]; then
    load_args=("checkpoint.load_path=$BASE_CHECKPOINT" "checkpoint.load_training_state=false")
  else
    load_args=("checkpoint.load_path=$previous" "checkpoint.load_training_state=true")
  fi
  "$TORCHRUN" --nproc-per-node=8 --master-port="$((29556 + target / 300))" \
    -m cosmos3_joint_video_hand_pose.src.train --sft-toml "$CONFIG" \
    "trainer.max_iter=$target" "scheduler.cycle_lengths=[2000]" "${load_args[@]}"
  checkpoint="$JOB_ROOT/checkpoints/iter_$(printf '%09d' "$target")"
  [[ -f "$checkpoint/model/.metadata" ]] || { echo "missing complete checkpoint: $checkpoint" >&2; exit 3; }
  "$ROOT/scripts/run_overfit_v0_2_lr_balanced_replays.sh" "$(printf '%09d' "$target")"
  previous="$checkpoint"
done
