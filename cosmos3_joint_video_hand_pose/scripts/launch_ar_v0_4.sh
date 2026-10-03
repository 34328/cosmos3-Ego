#!/usr/bin/env bash

# Versioned AR V0.4 registration, with the unchanged official Cosmos launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WORKDIR="${WORKDIR:-$PROJECT_ROOT}"
TRAINING_MODULE="${TRAINING_MODULE:-cosmos3_joint_video_hand_pose.src.train_ar_v04}"
TRAINING_PYTHONPATH="${TRAINING_PYTHONPATH:-$PROJECT_ROOT:$PROJECT_ROOT/packages/cosmos3}"
TOML_FILE="${TOML_FILE:-$PROJECT_ROOT/cosmos3_joint_video_hand_pose/configs/ar_v0_4.toml}"

: "${BASE_CHECKPOINT_PATH:=/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp}"
: "${WAN_VAE_PATH:=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${TEXT_TOKENIZER_PATH:=/mnt/checkpoints/Cosmos3-Nano/text_tokenizer}"
export BASE_CHECKPOINT_PATH WAN_VAE_PATH TEXT_TOKENIZER_PATH

EXTRA_DATASET_CHECK='[[ -d "$TEXT_TOKENIZER_PATH" ]] || { echo "ERROR: TEXT_TOKENIZER_PATH not found: $TEXT_TOKENIZER_PATH" >&2; exit 1; }; [[ -f "$TEXT_TOKENIZER_PATH/tokenizer_config.json" ]] || { echo "ERROR: missing tokenizer_config.json under $TEXT_TOKENIZER_PATH" >&2; exit 1; }; [[ -f "$TEXT_TOKENIZER_PATH/tokenizer.json" || -f "$TEXT_TOKENIZER_PATH/tokenizer.model" || -f "$TEXT_TOKENIZER_PATH/vocab.json" ]] || { echo "ERROR: missing tokenizer vocabulary under $TEXT_TOKENIZER_PATH" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$PROJECT_ROOT/packages/cosmos3/examples/_sft_launcher_common.sh"
