#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly ROOT=/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose
readonly OUTPUT_ROOT=/mnt/lzh/cosmos/outputs
readonly JOB_NAME=overfit_v0.5_frame_delta_b3
readonly JOB_ROOT="$OUTPUT_ROOT/joint_video_hand_pose/overfit/$JOB_NAME"
readonly BASE_CHECKPOINT=/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
readonly WAN_VAE=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
readonly CONFIG="$ROOT/configs/overfit_v0_5_frame_delta_b3.toml"
readonly FRAME_NORMALIZER="$ROOT/artifacts/cosmos3_action_contract/v3_frame_delta/normalizers/future_frame_delta_normalizer.json"

[[ ! -e "$JOB_ROOT" ]] || { echo "refusing to overwrite $JOB_ROOT" >&2; exit 2; }
[[ -f "$FRAME_NORMALIZER" ]] || { echo "missing B3 normalizer: $FRAME_NORMALIZER" >&2; exit 3; }
"$PYTHON" -c 'from cosmos3_joint_video_hand_pose.src.normalization import PiecewiseAsinhNormalizer; import sys; PiecewiseAsinhNormalizer(sys.argv[1])' "$FRAME_NORMALIZER"

export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT"
export WAN_VAE_PATH="$WAN_VAE"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"

echo "=== $JOB_NAME: v0.4 + camera/wrist frame deltas, CP1/FSDP-8/75K, cosine 1200 ==="
exec "$TORCHRUN" --nproc-per-node=8 --master-port="${MASTER_PORT:-29563}" \
  -m cosmos3_joint_video_hand_pose.src.train --sft-toml "$CONFIG" "$@"
