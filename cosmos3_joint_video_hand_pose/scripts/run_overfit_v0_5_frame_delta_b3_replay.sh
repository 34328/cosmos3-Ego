#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly ROOT=/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose
readonly TRAIN_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/overfit/overfit_v0.5_frame_delta_b3
readonly INFERENCE_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/inference/overfit_v0.5_frame_delta_b3
readonly INPUT_ROOT="$INFERENCE_ROOT/monitor_inputs"
readonly STATE_NORMALIZER="$ROOT/artifacts/cosmos3_action_contract/v2/normalizers/state_normalizer.json"
readonly FUTURE_NORMALIZER="$ROOT/artifacts/cosmos3_action_contract/v3_frame_delta/normalizers/future_frame_delta_normalizer.json"
readonly ITERATION="${1:-000001200}"

export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WAN_VAE_PATH=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth

checkpoint="$TRAIN_ROOT/checkpoints/iter_$ITERATION/model"
output="$INFERENCE_ROOT/iter_$ITERATION"
[[ -f "$checkpoint/.metadata" ]] || { echo "checkpoint incomplete: $checkpoint" >&2; exit 3; }
[[ -f "$STATE_NORMALIZER" ]] || { echo "missing state normalizer: $STATE_NORMALIZER" >&2; exit 3; }
[[ -f "$FUTURE_NORMALIZER" ]] || { echo "missing B3 future normalizer: $FUTURE_NORMALIZER" >&2; exit 3; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite replay output: $output" >&2; exit 4; }

if [[ ! -e "$INPUT_ROOT/manifest.json" ]]; then
  "$PYTHON" -m cosmos3_joint_video_hand_pose.src.monitoring prepare \
    --output "$INPUT_ROOT" \
    --train-episodes "$ROOT/artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/episodes.csv" \
    --train-segments "$ROOT/artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/segments.csv" \
    --train-only --count 4 --min-frames 201 --max-frames 401 --selection-seed 42 \
    --prompt-mode episode_context_and_segment \
    --state-normalizer "$STATE_NORMALIZER" \
    --future-normalizer "$FUTURE_NORMALIZER" \
    --rigid-pose-frame-delta
fi

mkdir -p "$output"
cp -a "$INPUT_ROOT/inputs" "$output/inputs"
cp -a "$INPUT_ROOT/inference_inputs" "$output/inference_inputs"
cp -a "$INPUT_ROOT/manifest.json" "$output/manifest.json"
"$TORCHRUN" --nproc-per-node=8 --master-port=29605 \
  -m cosmos3_joint_video_hand_pose.src.inference --parallelism-preset=throughput \
  --no-guardrails --no-use-ema-weights --sampler=unipc --num-steps=30 --shift=5 \
  -i "$output/inference_inputs/*.json" -o "$output/generated" \
  --checkpoint-path "$checkpoint" \
  --experiment egoverse_joint_video_hand_pose_overfit_v0_5_frame_delta_b3 --seed 0

for metadata in "$output"/inputs/*/metadata.json; do
  name=$(basename "$(dirname "$metadata")")
  "$PYTHON" -m cosmos3_joint_video_hand_pose.src.monitoring render \
    --video "$output/generated/$name/vision.mp4" \
    --sample-outputs "$output/generated/$name/sample_outputs.json" \
    --metadata "$metadata" --output "$output/replays/$name.mp4"
done
touch "$output/REPLAY_COMPLETE"
echo "v0.5 B3 replay complete: $output"
