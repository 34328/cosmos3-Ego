#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly ROOT=/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose
readonly OUTPUT_ROOT=/mnt/lzh/cosmos/outputs
readonly JOB_NAME=overfit_v0.4_video_first_causal_mask
readonly JOB_ROOT="$OUTPUT_ROOT/joint_video_hand_pose/overfit/$JOB_NAME"
readonly BASE_CHECKPOINT=/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
readonly WAN_VAE=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
readonly CONFIG="$ROOT/configs/overfit_v0_4_video_first_causal_mask.toml"

[[ ! -e "$JOB_ROOT" ]] || { echo "refusing to overwrite $JOB_ROOT" >&2; exit 2; }
"$PYTHON" "$ROOT/artifacts/cosmos3_action_contract/v2/validate_manifest.py"

export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT"
export WAN_VAE_PATH="$WAN_VAE"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"

echo "=== $JOB_NAME: v0.3 baseline + video-first C/V/A mask, CP1/FSDP-8/75K ==="
exec "$TORCHRUN" --nproc-per-node=8 --master-port="${MASTER_PORT:-29562}" \
  -m cosmos3_joint_video_hand_pose.src.train --sft-toml "$CONFIG" "$@"
