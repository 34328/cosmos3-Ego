#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
readonly ROOT="${REPO_ROOT}/cosmos3_egoverse_it2v"
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/cosmos3"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WAN_VAE_PATH=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export BASE_CHECKPOINT_PATH=/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
export IMAGINAIRE_OUTPUT_ROOT="${REPO_ROOT}/outputs"

exec "$TORCHRUN" --nproc-per-node=8 --master-port="${MASTER_PORT:-29610}" \
  -m cosmos3_egoverse_it2v.src.train --sft-toml "$ROOT/configs/train_v2.toml" "$@"
