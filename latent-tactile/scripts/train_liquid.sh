#!/usr/bin/env bash
set -euo pipefail
: "${TACTILE_DATASET:?Set TACTILE_DATASET to the HDF5 file or episode directory}"
: "${TACTILE_NORM_STATS:?Set TACTILE_NORM_STATS to the matching normalization JSON}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
args=(
  -m dino_wm.train_dino_wm_tactile_k_step
  --config "${root}/dino_wm/configs/liquid.yaml"
  --hdf5_path "${TACTILE_DATASET}"
  --norm_stats_json "${TACTILE_NORM_STATS}"
  --output_dir "${WM_OUTPUT_DIR:-outputs/liquid_world_model}"
)
if [[ -n "${TACTILE_DECODERS:-}" ]]; then
  args+=(--decoder_dir "${TACTILE_DECODERS}")
fi
exec "${PYTHON:-python}" "${args[@]}" "$@"
