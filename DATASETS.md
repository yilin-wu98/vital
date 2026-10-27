# ViTaL datasets

[Hugging Face](https://huggingface.co/datasets/yilin-wu/ViTaL-datasets) ·
[Paper](https://arxiv.org/abs/2606.14981) ·
[Project website](https://yilin-wu98.github.io/vital_website/)

**The initial upload is in progress.** All local exports have passed integrity
checks; the table lists the complete intended release.

| Task | Episodes | Train / test | Action frames | Compressed GB | Recorded WM run |
| --- | ---: | ---: | ---: | ---: | --- |
| Pipette / liquid transfer | 320 | 270 / 50 | 153,498 | 361.02 | `xrjdyidg` |
| Insertion | 330 | 280 / 50 | 85,395 | 200.75 | `i8cafzcp` |
| Wiping | 330 | 280 / 50 | 63,738 | 154.74 | `bph9xvcw` |

The release contains 980 episodes and 302,631 action frames.
Episode files total 716,509,764,047 bytes (716.51 decimal GB).
The largest episode is 2.23 GB.

## Download and use

From the code repository root:

```bash
python -m pip install "huggingface_hub>=0.36,<1"
hf download yilin-wu/ViTaL-datasets --repo-type dataset \
  --include "pipette/*" --local-dir datasets/vital
```

Replace `pipette/*` with `insertion/*` or `wiping/*`, or omit `--include` for all
three tasks. Each task has this layout:

```text
pipette/
├── episodes/trajectory_*.hdf5
├── normalization.json
├── splits.json
├── manifest.jsonl
└── metadata.json
```

For pipette training:

```bash
export TACTILE_DATASET="$PWD/datasets/vital/pipette/episodes"
export TACTILE_NORM_STATS="$PWD/datasets/vital/pipette/normalization.json"
bash latent-tactile/scripts/train_liquid.sh --no_consolidated
```

The same episode directory and normalization file work with evaluation.
Pass `--no_consolidated` to world-model and decoder commands when using these
episode files. The documented active recipe and selected 67k checkpoint are
for pipette/liquid transfer; use task-specific weights for insertion or wiping.

The dataset contains encoded world-model training trajectories. Model weights,
base-policy training code, and robot evaluation logs are separate.

## Schema

Each HDF5 file contains a `data` group. `T` is the action count; a raw
observation array can have `R=T` or `R=T+1` samples.

| Field | Shape | Dtype |
| --- | --- | --- |
| `actions` | `(T, 7)` | float64 |
| `states` | `(T, 8)` | float64 |
| `camera_0`, `camera_1` | `(R, 224, 224, 3)` | float32 |
| `camera_2` | `(R, H, W, 3)` | uint8 |
| `cam_0_patch_embd`, `cam_1_patch_embd` | `(T, 196, 768)` | float32 |
| `cam_tactile_patch_embd` | `(T, 392, 512)` | float32 |
| `cam_0_cls_embd`, `cam_1_cls_embd` | `(T, 768)` | float32 |
| `cam_tactile_cls_embd` | `(T, 512)` | float32 |
| `ee_states` | `(R, 16)` | float64 |
| `gripper_states` | `(R,)` | float64 |
| `joint_states` | `(R, 7)` | float64 |

Tactile images are 240×320 for pipette/wiping and 256×256 for insertion.
The world-model loader uses the first 196 tactile patch tokens; all 392
original tokens are preserved in the dataset.

There are extra terminal raw observations in 56 pipette, 43 insertion, and
29 wiping episodes. Actions, model states, and embeddings remain aligned.
The loaders use the first `T` raw samples. No terminal samples were discarded
or synthesized.

The `data` group records the original trajectory ID. It also retains
controller-configuration JSON where present: 100 pipette, 100 insertion, and
50 wiping episodes. These configurations contain control gains, interpolation,
state-estimator settings, and action scaling.

The trajectories do not include per-episode language instructions or success
labels. Task directories identify the task family.

## Splits and normalization

The original trajectory names are sorted lexicographically, shuffled with
seed 42, and split with the first 50 trajectories held out per task.
`splits.json` records the exact IDs. This matches the released loaders.

Actions are stored before normalization. Apply the accompanying `mean_acs`
and `std_acs` to all seven dimensions. The active recipe does not normalize
the eight-dimensional model state.

For pipette, `normalization.json` is the unchanged content of the original
`tactile_liquid_wm_liquid.json`. Some old robot launchers referred to an absent
`tactile_liquid_wm_info.json`; the release requires the matching training
statistics rather than silently falling back to generic values.

## Integrity and export

All arrays and dtypes are preserved under lossless gzip compression.
Each exported array was read back and compared to its source by SHA-256.
No NaN or infinite values were found. Source datasets were opened read-only.

Each task's `manifest.jsonl` records episode-file and per-array checksums.
`metadata.json` records counts, source basenames, and normalization hashes.
[DATASET_RELEASE.json](DATASET_RELEASE.json) contains the combined release
inventory and task-manifest hashes.

The exporter is included for reproducing the format:

```bash
python -m dino_wm.export_dataset \
  --source /path/to/consolidated.hdf5 \
  --normalization /path/to/normalization.json \
  --output /path/to/new_release/pipette \
  --task pipette --compression gzip --workers 4
```

Resume is allowed only when the source and export configuration match.
Unknown nonempty output directories and altered episode files are rejected.

## License and citation

The dataset uses [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Please cite [ViTaL](https://arxiv.org/abs/2606.14981); the BibTeX entry is in
the [main README](README.md#citation).
