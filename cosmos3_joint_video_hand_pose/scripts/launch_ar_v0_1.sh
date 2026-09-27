#!/usr/bin/env bash
# AR v0.1 overfit training (8 GPUs, CP1/FSDP-8).
# Usage: launch_ar_v0_1.sh [recipe.toml] [extra overrides...]   (default: configs/ar_v0_1.toml = R2 from Nano)
set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/cosmos3"
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly ROOT="${REPO_ROOT}/cosmos3_joint_video_hand_pose"
readonly OUTPUT_ROOT="${REPO_ROOT}/outputs"
readonly CONFIG="${1:-$ROOT/configs/ar_v0_1.toml}"
shift $(( $# > 0 ? 1 : 0 ))
readonly JOB_NAME="$(sed -n 's/^name *= *"\(.*\)"/\1/p' "$CONFIG" | head -1)"
readonly JOB_ROOT="$OUTPUT_ROOT/joint_video_hand_pose/ar/$JOB_NAME"
readonly FRAME_NORMALIZER="$ROOT/artifacts/cosmos3_action_contract/v4_frame_delta_30hz/normalizers/future_frame_delta_normalizer.json"

[[ -n "$JOB_NAME" ]] || { echo "cannot read job name from $CONFIG" >&2; exit 2; }
[[ ! -e "$JOB_ROOT" ]] || { echo "refusing to overwrite $JOB_ROOT" >&2; exit 2; }
[[ -f "$FRAME_NORMALIZER" ]] || { echo "missing v4 normalizer: $FRAME_NORMALIZER" >&2; exit 3; }
"$PYTHON" -c 'from cosmos3_joint_video_hand_pose.src.normalization import PiecewiseAsinhNormalizer; import sys; PiecewiseAsinhNormalizer(sys.argv[1])' "$FRAME_NORMALIZER"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"

echo "=== $JOB_NAME: AR v0.1 lingbot TF, stride 2, K=8, C~{1..4}, window~[4,64], CP1/FSDP-8 ==="
exec "$TORCHRUN" --nproc-per-node=8 --master-port="${MASTER_PORT:-29580}" \
  -m cosmos3_joint_video_hand_pose.src.train --sft-toml "$CONFIG" "$@"
