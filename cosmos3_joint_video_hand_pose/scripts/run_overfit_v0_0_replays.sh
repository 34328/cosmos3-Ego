#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun
readonly ROOT=/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose
readonly TRAIN_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/train/overfit_v0.0
readonly INFERENCE_ROOT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/inference/overfit_v0.0
readonly INPUT_ROOT="$INFERENCE_ROOT/monitor_inputs"
readonly LEGACY_ROOT="$INFERENCE_ROOT/legacy_plain_prompt"
readonly V2="$ROOT/artifacts/cosmos3_action_contract/v2/normalizers"

PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3 "$PYTHON" "$ROOT/artifacts/cosmos3_action_contract/v2/validate_manifest.py"
export PYTHONPATH=/mnt/lzh/cosmos:/mnt/lzh/cosmos/packages/cosmos3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WAN_VAE_PATH=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth

if [[ ! -e "$INPUT_ROOT/manifest.json" ]]; then
  "$PYTHON" -m cosmos3_joint_video_hand_pose.src.monitoring prepare \
    --output "$INPUT_ROOT" \
    --train-episodes "$ROOT/artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/episodes.csv" \
    --train-segments "$ROOT/artifacts/cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1/segments.csv" \
    --train-only --count 4 --min-frames 201 --max-frames 401 --selection-seed 42 \
    --prompt-mode episode_context_and_segment \
    --state-normalizer "$V2/state_normalizer.json" --future-normalizer "$V2/future_delta_normalizer.json"
fi

if (($#)); then
  ITERATIONS=("$@")
else
  mapfile -t ITERATIONS < <(
    find "$TRAIN_ROOT/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' -printf '%f\n' \
      | sed 's/^iter_//' | sort -n | xargs -r -n1 printf '%09d\n'
  )
fi
[[ ${#ITERATIONS[@]} -gt 0 ]] || { echo "no checkpoint iterations found under $TRAIN_ROOT/checkpoints" >&2; exit 3; }

for ITERATION in "${ITERATIONS[@]}"; do
  CHECKPOINT="$TRAIN_ROOT/checkpoints/iter_${ITERATION}/model"
  OUTPUT="$INFERENCE_ROOT/iter_${ITERATION}"
  [[ -f "$CHECKPOINT/.metadata" ]] || { echo "checkpoint incomplete: $CHECKPOINT" >&2; exit 3; }
  if [[ "${COSMOS_OVERFIT_FORCE_REPLAY:-0}" == "1" && -e "$OUTPUT" ]]; then
    mkdir -p "$LEGACY_ROOT"
    LEGACY_OUTPUT="$LEGACY_ROOT/iter_${ITERATION}"
    [[ ! -e "$LEGACY_OUTPUT" ]] || {
      echo "legacy backup already exists: $LEGACY_OUTPUT" >&2
      exit 4
    }
    mv "$OUTPUT" "$LEGACY_OUTPUT"
  fi
  if [[ -e "$OUTPUT/replays" ]] && compgen -G "$OUTPUT/replays/*.mp4" >/dev/null; then
    echo "replay already exists for iter_${ITERATION}; skipping"
    continue
  fi
  if [[ -e "$OUTPUT" ]]; then
    echo "removing incomplete replay output: $OUTPUT"
    rm -rf "$OUTPUT"
  fi
  mkdir -p "$OUTPUT"
  cp -a "$INPUT_ROOT/inputs" "$OUTPUT/inputs"
  cp -a "$INPUT_ROOT/inference_inputs" "$OUTPUT/inference_inputs"
  cp -a "$INPUT_ROOT/manifest.json" "$OUTPUT/manifest.json"
  "$TORCHRUN" --nproc-per-node=8 --master-port="$((29600 + 10#$ITERATION / 600))" \
    -m cosmos3_joint_video_hand_pose.src.inference \
    --parallelism-preset=throughput --no-guardrails --no-use-ema-weights \
    --sampler=unipc --num-steps=30 --shift=5 \
    -i "$OUTPUT/inference_inputs/*.json" -o "$OUTPUT/generated" \
    --checkpoint-path "$CHECKPOINT" --experiment egoverse_joint_video_hand_pose_overfit_v0_0 --seed 0
  for METADATA in "$OUTPUT"/inputs/*/metadata.json; do
    NAME=$(basename "$(dirname "$METADATA")")
    "$PYTHON" -m cosmos3_joint_video_hand_pose.src.monitoring render \
      --video "$OUTPUT/generated/$NAME/vision.mp4" \
      --sample-outputs "$OUTPUT/generated/$NAME/sample_outputs.json" \
      --metadata "$METADATA" --output "$OUTPUT/replays/$NAME.mp4"
  done
done
