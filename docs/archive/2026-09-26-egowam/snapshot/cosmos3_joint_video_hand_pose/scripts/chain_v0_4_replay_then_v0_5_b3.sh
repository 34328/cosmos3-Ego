#!/usr/bin/env bash
set -euo pipefail

cd /mnt/lzh/cosmos
readonly ROOT=/mnt/lzh/cosmos/cosmos3_joint_video_hand_pose
readonly TRAIN_PID="${1:?usage: chain_v0_4_replay_then_v0_5_b3.sh TRAIN_PID}"
readonly V04_CHECKPOINT=/mnt/lzh/cosmos/outputs/joint_video_hand_pose/overfit/overfit_v0.4_video_first_causal_mask/checkpoints/iter_000001200/model/.metadata

echo "waiting for v0.4 training pid $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
done

[[ -f "$V04_CHECKPOINT" ]] || {
  echo "v0.4 exited without a complete iter_000001200 checkpoint; refusing replay/B3 launch" >&2
  exit 5
}

echo "v0.4 training complete; starting step-1200 replay"
"$ROOT/scripts/run_overfit_v0_4_replay.sh" 000001200

echo "v0.4 replay complete; starting fresh B3 training"
exec env MASTER_PORT=29563 "$ROOT/scripts/launch_overfit_v0_5_frame_delta_b3.sh"
