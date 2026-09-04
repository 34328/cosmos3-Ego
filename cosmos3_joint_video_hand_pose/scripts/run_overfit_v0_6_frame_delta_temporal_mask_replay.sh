#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly TRAIN_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/overfit/overfit_v0.6_frame_delta_temporal_mask
readonly INFERENCE_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/inference/overfit_v0.6_frame_delta_temporal_mask
# Reuse the exact v0.5 B3 selection rather than merely repeating its seed.
readonly INPUT_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/inference/overfit_v0.5_frame_delta_b3/monitor_inputs
readonly ITERATION="${1:-000001200}"

export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WAN_VAE_PATH=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth

checkpoint="$TRAIN_ROOT/checkpoints/iter_$ITERATION/model"
output="$INFERENCE_ROOT/iter_$ITERATION"
[[ -f "$checkpoint/.metadata" ]] || { echo "checkpoint incomplete: $checkpoint" >&2; exit 3; }
[[ -f "$INPUT_ROOT/manifest.json" ]] || { echo "missing v0.5 B3 replay inputs: $INPUT_ROOT" >&2; exit 3; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite replay output: $output" >&2; exit 4; }

# The v0.5 metadata is authoritative for B3 decoding and contains
# rigid_pose_frame_delta=true plus the v2/v3 normalizer contract.
mkdir -p "$output"
cp -a "$INPUT_ROOT/inputs" "$output/inputs"
cp -a "$INPUT_ROOT/inference_inputs" "$output/inference_inputs"
cp -a "$INPUT_ROOT/manifest.json" "$output/manifest.json"

"$TORCHRUN" --nproc-per-node=8 --master-port=29606 \
  -m cosmos3_joint_video_hand_pose.src.inference --parallelism-preset=throughput \
  --no-use-torch-compile \
  --no-guardrails --no-use-ema-weights --sampler=unipc --num-steps=30 --shift=5 \
  -i "$output/inference_inputs/*.json" -o "$output/generated" \
  --checkpoint-path "$checkpoint" \
  --experiment egoverse_joint_video_hand_pose_overfit_v0_6_frame_delta_temporal_mask --seed 0

for metadata in "$output"/inputs/*/metadata.json; do
  name=$(basename "$(dirname "$metadata")")
  "$PYTHON" -m cosmos3_joint_video_hand_pose.src.monitoring render \
    --video "$output/generated/$name/vision.mp4" \
    --sample-outputs "$output/generated/$name/sample_outputs.json" \
    --metadata "$metadata" --output "$output/replays/$name.mp4"
done
touch "$output/REPLAY_COMPLETE"
echo "v0.6 temporal-mask B3 replay complete: $output"
