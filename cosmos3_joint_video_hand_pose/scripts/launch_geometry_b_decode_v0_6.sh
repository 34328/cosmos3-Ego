#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/cosmos3"
readonly PYTHON="${PYTHON_BIN:-/home/lzh/miniconda3/envs/cosmos3/bin/python}"
readonly TORCHRUN="${TORCHRUN_BIN:-/home/lzh/miniconda3/envs/cosmos3/bin/torchrun}"
readonly ROOT="${REPO_ROOT}/cosmos3_joint_video_hand_pose"
readonly OUTPUT_ROOT="${REPO_ROOT}/outputs"
readonly JOB_ROOT="$OUTPUT_ROOT/joint_video_hand_pose/geometry/geometry_b_decode_v0_6"
readonly CONFIG="$ROOT/configs/geometry_b_decode_v0_6.toml"
readonly FRAME_NORMALIZER="$ROOT/artifacts/cosmos3_action_contract/v3_frame_delta/normalizers/future_frame_delta_normalizer.json"
readonly CODEC_ROOT="$ROOT/artifacts/cosmos3_hand_codecs/v2_4/option_b_mlp15"

[[ ! -e "$JOB_ROOT" ]] || { echo "refusing to overwrite $JOB_ROOT" >&2; exit 2; }
[[ -x "$PYTHON" && -x "$TORCHRUN" ]] || { echo "missing Python or torchrun runtime" >&2; exit 3; }
[[ -f "$FRAME_NORMALIZER" ]] || { echo "missing B3 normalizer: $FRAME_NORMALIZER" >&2; exit 3; }
[[ -f "$CODEC_ROOT/right_mlp15_primary.pt" && -f "$CODEC_ROOT/left_mlp15_primary.pt" ]] || {
  echo "missing frozen hand codecs in $CODEC_ROOT" >&2; exit 3;
}
"$PYTHON" -c 'from cosmos3_joint_video_hand_pose.src.normalization import PiecewiseAsinhNormalizer; import sys; PiecewiseAsinhNormalizer(sys.argv[1])' "$FRAME_NORMALIZER"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"

exec "$TORCHRUN" --nproc-per-node=8 --master-port="${MASTER_PORT:-29571}" \
  -m cosmos3_joint_video_hand_pose.src.train --sft-toml "$CONFIG" "$@"
