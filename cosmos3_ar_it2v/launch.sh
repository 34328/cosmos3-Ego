#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$PROJECT_ROOT"
TRAINING_MODULE=cosmos3_ar_it2v.train
TRAINING_PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/packages/cosmos3"
TOML_FILE="${TOML_FILE:-$PROJECT_ROOT/cosmos3_ar_it2v/configs/ego100h.toml}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
export TEXT_TOKENIZER_PATH="${TEXT_TOKENIZER_PATH:-/mnt/checkpoints/Cosmos3-Nano/text_tokenizer}"
TAIL_OVERRIDES=(${EXTRA_TAIL_OVERRIDES:-})
source "$PROJECT_ROOT/packages/cosmos3/examples/_sft_launcher_common.sh"
