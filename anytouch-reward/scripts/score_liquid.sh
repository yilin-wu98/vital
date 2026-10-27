#!/usr/bin/env bash
set -euo pipefail
: "${ANYTOUCH_CHECKPOINT:?Set ANYTOUCH_CHECKPOINT to checkpoint-4frames.pth}"
video="${1:?Usage: bash scripts/score_liquid.sh VIDEO [additional options]}"
shift
exec "${PYTHON:-python}" -m anytouch_reward.video \
  --video_path "${video}" \
  --checkpoint "${ANYTOUCH_CHECKPOINT}" \
  --text "grasp lightly" --text "grasp heavily" \
  --sensor gelsight_mini --num_frames 4 --stride 2 --frame_stride 1 \
  --aggregate mean_patch --video_fps 15 \
  --output_dir "${REWARD_OUTPUT_DIR:-outputs/liquid_reward}" \
  "$@"
