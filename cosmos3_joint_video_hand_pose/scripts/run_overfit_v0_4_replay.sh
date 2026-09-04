#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly ROOT=/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose
readonly TRAIN_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/overfit/overfit_v0.4_video_first_causal_mask
readonly INFERENCE_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/inference/overfit_v0.4_video_first_causal_mask
readonly INPUT_ROOT="$INFERENCE_ROOT/monitor_inputs"
readonly V2="$ROOT/artifacts/cosmos3_action_contract/v2/normalizers"
readonly ITERATION="${1:-000001200}"

export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WAN_VAE_PATH=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
"$PYTHON" "$ROOT/artifacts/cosmos3_action_contract/v2/validate_manifest.py"

checkpoint="$TRAIN_ROOT/checkpoints/iter_$ITERATION/model"
output="$INFERENCE_ROOT/iter_$ITERATION"
[[ -f "$checkpoint/.metadata" ]] || { echo "checkpoint incomplete: $checkpoint" >&2; exit 3; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite replay output: $output" >&2; exit 4; }

if [[ ! -e "$INPUT_ROOT/manifest.json" ]]; then
  "$PYTHON" -m cosmos3_joint_video_hand_pose.src.monitoring prepare \
    --output "$INPUT_ROOT" \
    --train-episodes "$ROOT/artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/episodes.csv" \
    --train-segments "$ROOT/artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/segments.csv" \
    --train-only --count 4 --min-frames 201 --max-frames 401 --selection-seed 42 \
    --prompt-mode episode_context_and_segment \
    --state-normalizer "$V2/state_normalizer.json" \
    --future-normalizer "$V2/future_delta_normalizer.json"
fi

mkdir -p "$output"
cp -a "$INPUT_ROOT/inputs" "$output/inputs"
cp -a "$INPUT_ROOT/inference_inputs" "$output/inference_inputs"
cp -a "$INPUT_ROOT/manifest.json" "$output/manifest.json"
"$TORCHRUN" --nproc-per-node=8 --master-port=29604 \
  -m cosmos3_joint_video_hand_pose.src.inference --parallelism-preset=throughput \
  --no-guardrails --no-use-ema-weights --sampler=unipc --num-steps=30 --shift=5 \
  -i "$output/inference_inputs/*.json" -o "$output/generated" \
  --checkpoint-path "$checkpoint" \
  --experiment egoverse_joint_video_hand_pose_overfit_v0_4_video_first_causal_mask --seed 0

for metadata in "$output"/inputs/*/metadata.json; do
  name=$(basename "$(dirname "$metadata")")
  "$PYTHON" -m cosmos3_joint_video_hand_pose.src.monitoring render \
    --video "$output/generated/$name/vision.mp4" \
    --sample-outputs "$output/generated/$name/sample_outputs.json" \
    --metadata "$metadata" --output "$output/replays/$name.mp4"
done
touch "$output/REPLAY_COMPLETE"
echo "v0.4 replay complete: $output"
