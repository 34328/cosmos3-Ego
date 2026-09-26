#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
readonly ROOT="${REPO_ROOT}/cosmos3_egoverse_it2v"
readonly VERSION="${EGOVERSE_IT2V_VERSION:-v1}"
readonly TRAIN_ROOT="${REPO_ROOT}/outputs/egoverse_it2v/train/$VERSION"
readonly OUTPUT_ROOT="${REPO_ROOT}/outputs/egoverse_it2v/inference/$VERSION"
readonly MONITOR_ROOT="${EGOVERSE_MONITOR_ROOT:-${REPO_ROOT}/outputs/joint_video_hand_pose/inference/overfit_v0.5_frame_delta_b3/monitor_inputs}"
readonly INPUT_ROOT="$OUTPUT_ROOT/inference_inputs"
readonly PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
readonly TORCHRUN=/home/lzh/miniconda3/envs/cosmos3/bin/torchrun

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/cosmos3"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WAN_VAE_PATH=/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth

[[ -f "$MONITOR_ROOT/manifest.json" ]] || { echo "missing replay selection: $MONITOR_ROOT" >&2; exit 3; }

"$PYTHON" -m cosmos3_egoverse_it2v.src.replay prepare \
  --source "$MONITOR_ROOT/inference_inputs" --output "$INPUT_ROOT"

ITERATIONS=("${@:-300 600}")
for ITERATION in ${ITERATIONS[*]}; do
  PADDED=$(printf '%09d' "$ITERATION")
  CHECKPOINT="$TRAIN_ROOT/checkpoints/iter_$PADDED/model"
  OUTPUT="$OUTPUT_ROOT/iter_$PADDED"
  GENERATED="$OUTPUT/.generated"
  [[ -s "$CHECKPOINT/.metadata" ]] || { echo "incomplete checkpoint: $CHECKPOINT" >&2; exit 3; }
  if compgen -G "$OUTPUT/replays/*.mp4" >/dev/null; then
    echo "replays already complete for iter_$PADDED; skipping"
    continue
  fi
  [[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite incomplete replay: $OUTPUT" >&2; exit 4; }
  mkdir -p "$OUTPUT"
  "$TORCHRUN" --nproc-per-node=8 --master-port="$((29720 + ITERATION / 300))" \
    -m cosmos3_egoverse_it2v.src.inference \
    --parallelism-preset=throughput --no-guardrails --no-use-ema-weights \
    --sampler=unipc --num-steps=30 --shift=5 \
    -i "$INPUT_ROOT/*.json" -o "$GENERATED" \
    --checkpoint-path "$CHECKPOINT" --experiment egoverse_it2v_v1 --seed 0 \
    >"$OUTPUT/inference.log" 2>&1
  for INPUT in "$INPUT_ROOT"/*.json; do
    NAME=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$INPUT")
    PROMPT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prompt"])' "$INPUT")
    "$PYTHON" -m cosmos3_egoverse_it2v.src.replay render \
      --generated "$GENERATED/$NAME/vision.mp4" \
      --reference "$MONITOR_ROOT/inputs/$NAME/reference_video.mp4" \
      --prompt "$PROMPT" --output "$OUTPUT/replays/$NAME.mp4"
  done
  rm -rf "$GENERATED"
done
