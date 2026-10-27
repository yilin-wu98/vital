#!/usr/bin/env bash
set -euo pipefail
: "${TACTILE_DATASET:?Set TACTILE_DATASET}"
: "${TACTILE_NORM_STATS:?Set TACTILE_NORM_STATS}"
: "${WM_CHECKPOINT:?Set WM_CHECKPOINT}"
: "${TACTILE_DECODERS:?Set TACTILE_DECODERS}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" -m dino_wm.eval_dino_wm_tactile_k_step \
  --config "${root}/dino_wm/configs/liquid.yaml" \
  --hdf5_path "${TACTILE_DATASET}" \
  --norm_stats_json "${TACTILE_NORM_STATS}" \
  --wm_checkpoint "${WM_CHECKPOINT}" \
  --decoder_dir "${TACTILE_DECODERS}" \
  --output_dir "${WM_EVAL_DIR:-outputs/liquid_evaluation}" \
  "$@"
