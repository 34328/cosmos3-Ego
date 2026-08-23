#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly ROOT=/mnt/lzh/cosmos/cosmos3_egoverse_it2v
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun

export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WAN_VAE_PATH=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export BASE_CHECKPOINT_PATH=/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
export IMAGINAIRE_OUTPUT_ROOT=/mnt/lzh/cosmos/outputs

exec "$TORCHRUN" --nproc-per-node=8 --master-port="${MASTER_PORT:-29610}" \
  -m cosmos3_egoverse_it2v.src.train --sft-toml "$ROOT/configs/train_v2.toml" "$@"
