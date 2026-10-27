# Latent Tactile: active liquid world model

[Paper: Inference-time Policy Steering via Vision and Touch (ViTaL)](https://arxiv.org/abs/2606.14981)
· [Project website](https://yilin-wu98.github.io/vital_website/)
· [Datasets](https://huggingface.co/datasets/yilin-wu/ViTaL-datasets)
· [Repository overview](../README.md)

This release contains the **k-step concatenation world model used by the liquid
experiment launchers**. It predicts future visual/tactile patch embeddings and
an eight-dimensional robot state from the current observation, state, and an
eight-action block.

**Contents:** [Installation](#installation) ·
[Data and checkpoints](#data-and-checkpoints) ·
[Training](#train-the-active-world-model) ·
[Evaluation](#evaluate-the-selected-checkpoint) ·
[Data format](#data-contract) ·
[Decoder training](#supporting-decoder-stage) ·
[Citation](#citation)

## Active version and checkpoint

| Component | Selected version |
| --- | --- |
| Training entry point | `dino_wm/train_dino_wm_tactile_k_step.py` |
| Model | `TactileKStepTransformer` in `dino_wm/dino_models_k_step.py` |
| Architecture | `concat` |
| Dataset loader | `dino_wm/tactile_k_step_dataset.py` |
| Evaluation entry point | `dino_wm/eval_dino_wm_tactile_k_step.py` |
| Training and evaluation recipe | [`dino_wm/configs/liquid.yaml`](dino_wm/configs/liquid.yaml) |
| Saved training run | `xrjdyidg`, started 2026-05-03 UTC |
| Checkpoint selected by the robot launchers | `wm_kstep_concat_h1_p1_fs8_iter67000.pth` |

All five world-model-enabled liquid launchers select this **67,000-iteration
checkpoint**. The `best_*.pth` and `iter80000.pth` files are separate checkpoints;
neither is selected automatically by this release.

Both world-model entry points default to `liquid.yaml`, including when invoked
directly with `python -m`. They support the released `concat` architecture and
require `num_pred=1`. The old video-model trainer, experimental cross-attention
model, and separate trajectory-ID evaluator are excluded from this repository
and preserved in the original research workspace. Original research files
remain in their original checkout.

`dino_models.py` remains a required shared dependency: the active model imports
its transformer layers, and training/evaluation import `NormStats`.
`tactile_dataset.py` supplies shared camera definitions, episode loading, and
the decoder dataset. Decoder and encoding scripts are supporting stages of
this workflow.

## Installation

Run the installation commands from `vital/latent-tactile`. Use Python 3.10 or
later; Linux with Python 3.12 is the reference setup. The liquid training environment used Python
3.12, PyTorch 2.4.0, torchvision 0.19.0, NumPy 1.26.4, and transformers 4.42.3.
The AnyTouch package is needed for encoding raw tactile images; training from
already encoded HDF5 files does not load AnyTouch.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 numpy==1.26.4
python -m pip install -e ".[decoder,tracking]"
```

For encoding new tactile images, also install the sibling reward package:

```bash
python -m pip install -e ../anytouch-reward
```

`decoder` adds LPIPS for decoder training; `tracking` adds optional W&B logging.
Data and weights are supplied separately. No datasets, checkpoints, experiment
logs, or credentials are included.

The world-model recipe selects `cuda:0`. Pass `--device cuda:N` to either
shell wrapper to select another GPU. For a small CPU execution check, use
`--device cpu --no_compile`; the full training recipe was configured for a GPU.

## Data and checkpoints

Prepare the following inputs before training or evaluation:

| Input | Training | Standalone evaluation |
| --- | --- | --- |
| Encoded HDF5 trajectories | Required | Required |
| Matching normalization JSON | Required | Required |
| World-model checkpoint | Starts from scratch | Required: selected liquid checkpoint is `wm_kstep_concat_h1_p1_fs8_iter67000.pth` |
| RGB and tactile decoder checkpoints | Optional, for W&B videos | Required |
| AnyTouch and DINOv3 encoder weights | Only for encoding raw demonstrations | Already encoded data do not load these encoders |

The release inventory contains three world-model datasets: liquid transfer
(pipette), insertion, and wiping. The recipe documented below reproduces the
**liquid configuration**; it is not a claim that all three tasks use the same
training schedule or checkpoint.

Download the pipette dataset from `vital/latent-tactile`:

```bash
python -m pip install "huggingface_hub>=0.36,<1"
hf download yilin-wu/ViTaL-datasets --repo-type dataset \
  --include "pipette/*" --local-dir ../datasets/vital
```

See the [dataset guide](../DATASETS.md) for sizes, the full schema, and the
other tasks. The released `pipette/normalization.json` contains the unchanged
statistics from `tactile_liquid_wm_liquid.json`.

Both consolidated HDF5 files and directories of episode HDF5 files are
supported. For episode files, set `TACTILE_DATASET` to the episode directory
and append `--no_consolidated` to the training or evaluation command. Each
episode file must contain a `data` group with the fields described below.
Always provide that task's normalization JSON.

## Train the active world model

Run from `vital/latent-tactile` using the encoded liquid dataset and its
matching normalization file:

```bash
export TACTILE_DATASET="$PWD/../datasets/vital/pipette/episodes"
export TACTILE_NORM_STATS="$PWD/../datasets/vital/pipette/normalization.json"
export WM_OUTPUT_DIR=outputs/liquid_world_model
bash scripts/train_liquid.sh --no_consolidated
```

The effective recipe is:

| Setting | Value |
| --- | --- |
| Conditioning and predicted cameras | `camera_0,camera_1,camera_2` |
| History / prediction frames | `num_hist=1`, `num_pred=1` |
| Temporal layout | `lpb`, `frameskip=8` |
| Training loss | `multi_step`, random rollout length 3–5 |
| Action noise | DDIM enabled, 100 diffusion timesteps, geometric probability 0.05 |
| Action normalization | Recorded mean/std for all seven action dimensions |
| State normalization | Disabled |
| Sampling / split | Uniform segments; 50 held-out trajectories; seed 42 |
| Batch sizes | Training 16, evaluation 8 |
| Training budget / evaluation interval | 100,000 / 1,000 iterations |

Each model call predicts one observation eight raw timesteps ahead. Multi-step
training feeds predicted embeddings and state into the next call, accumulating
state and camera MSE over 3–5 calls. Predictions are detached between calls,
matching the research implementation.

**Keep the 100,000-iteration training budget when reproducing the recipe.**
The action-noise ramp depends on this budget. The selected checkpoint is an
intermediate snapshot at iteration 67,000, not a run configured to stop at
67,000. A new training run does not guarantee identical weights or results.

Every 1,000 iterations, training evaluates one held-out minibatch and saves
`wm_kstep_concat_h1_p1_fs8_iter<iteration>.pth`. `best_wm_kstep_concat_h1_p1_fs8.pth`
tracks the lowest value from those minibatch evaluations. This is not a
full-test-set metric.

Logging is disabled by default. For W&B logging and reconstruction videos,
provide the decoder directory and opt in:

```bash
export TACTILE_DECODERS=/path/to/liquid_decoders
bash scripts/train_liquid.sh --no_consolidated --wandb
```

Use `--no_compile` to disable `torch.compile`. CLI overrides are supported;
changing the camera set, temporal settings, loss, or noise settings defines a
different experiment from the active recipe.

## Evaluate the selected checkpoint

Use the same dataset, normalization statistics, and camera configuration as
training. Both separate decoder files must be in `TACTILE_DECODERS`:

```text
liquid_decoders/
├── decoder_camera_0_camera_1_last.pth
└── decoder_camera_2_last.pth
```

```bash
export TACTILE_DATASET="$PWD/../datasets/vital/pipette/episodes"
export TACTILE_NORM_STATS="$PWD/../datasets/vital/pipette/normalization.json"
export TACTILE_DECODERS=/path/to/liquid_decoders
export WM_CHECKPOINT=/path/to/wm_kstep_concat_h1_p1_fs8_iter67000.pth
export WM_EVAL_DIR=outputs/liquid_evaluation_67000
bash scripts/evaluate_liquid.sh --no_consolidated
```

The script loads `liquid.yaml`, uses the same seed-42/50-trajectory held-out
split, and evaluates the first 20 held-out trajectories, capped at 600 raw
frames each. It uses recorded actions without training-time action noise.
Add `--num_episode_videos 50` to visualize all held-out trajectories or
`--decode_gt_embeddings` to add a decoder-reconstruction comparison row.

**Evaluation protocol:** at each raw index `t`, the standalone evaluator uses
recorded embeddings and state at `t`, plus actions `t:t+8`, to predict the
observation at `t+8`. It repeats this with recorded history at the next anchor.
Although older comments called this “open-loop,” it is a sequence of
ground-truth-conditioned transitions, not an autoregressive rollout.

The output directory contains, per trajectory:

- `episode_<index>_traj_<id>.mp4`: ground truth, decoded prediction, and absolute
  image difference, with camera views side by side.
- `episode_<index>_traj_<id>_gt_embeds.npz`: reference embeddings.
- `episode_<index>_traj_<id>_pred_embeds.npz`: predicted embeddings.

Each NPZ contains `camera_0`, `camera_1`, and `camera_2` arrays with shape
`(T, 196, D)`, where `D=768` for RGB and `D=512` for tactile. The first eight
entries in the prediction file are copied ground-truth context; only
**indices `8:`** are predictions under the active recipe. Exclude that context
when computing prediction errors. The standalone script exports videos and
embeddings; it does not report aggregate test-set scores or robot success rates.

The optional training-time video has a different protocol: it feeds predicted
embeddings back on the subsampled timeline while retaining recorded robot
states. The training loss, training video, and standalone evaluation should
therefore be reported with their respective conditioning protocols.

## Data contract

A consolidated HDF5 file has `trajectory_*` groups. Each group contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `actions` | `(T, 7)` | Raw control actions |
| `states` | `(T, 8)` | Position, quaternion, gripper state |
| `cam_0_patch_embd` | `(T, N, 768)` | RGB camera 0 embeddings |
| `cam_1_patch_embd` | `(T, N, 768)` | RGB camera 1 embeddings |
| `cam_tactile_patch_embd` | `(T, N, 512)` | AnyTouch tactile embeddings |
| `camera_0`, `camera_1`, `camera_2` | per-frame images | Required for decoder training and videos |

Some episodes retain one extra terminal sample in raw camera and auxiliary
robot-state arrays. Actions, model states, and embeddings have the same length
`T`; the loaders use the first `T` raw samples. The dataset exporter preserves
the extra terminal samples and records each array's full shape.

The k-step loader pads or takes the first 196 patches, preserving the original
experiment behavior. A four-frame AnyTouch window at stride 2 produces 392
patch tokens before this step. This differs from offline video reward scoring,
which can mean-pool all 392 tokens. Do not change this reduction when using
existing world-model weights.

The normalizer JSON contains `mean_acs`, `std_acs`, and optionally state
statistics. Use `tactile_liquid_wm_liquid.json` for the selected liquid
checkpoint. Some old robot scripts referred to `tactile_liquid_wm_info.json`,
which was absent on the research machine. A supplied missing path now raises
an error rather than silently substituting generic statistics.

## Encode new demonstrations

Provide the upstream DINOv3 checkout and ViT-B/16 weights, plus the four-frame
AnyTouch checkpoint. The code uses **DINOv3 ViT-B/16** for the 768-dimensional
RGB embeddings; old experiment directory names containing `vits16` are not a
reliable architecture description.

```bash
python -m dino_wm.hdf5_to_dataset_tactile \
  --hdf5_dir /path/to/raw_trajectories \
  --work_dir /path/to/new_encoded_workdir \
  --checkpoint /path/to/checkpoint-4frames.pth \
  --dinov3_hub_dir /path/to/dinov3 \
  --dinov3_weights /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The command copies trajectories into the new work directory before adding
embeddings. It leaves source trajectories intact and refuses existing output
files. Plan storage for these working copies and the consolidated output.

## Supporting decoder stage

Decoder training is separate from world-model training. It is needed to
produce reconstruction videos and to decode visual predictions for control.
The existing evaluator expects a shared RGB decoder and a separate tactile
decoder. In this code, **`--mode tactile` creates those two groups**:

```bash
python -m dino_wm.train_dino_decoder_tactile \
  --hdf5_path "$TACTILE_DATASET" --no_consolidated --mode tactile --loss rae \
  --iters 15000 --resize_to_224 --compile \
  --output_dir outputs/liquid_decoders
```

These commands use the downloaded episode directory. To use an original
consolidated HDF5 file instead, point `TACTILE_DATASET` to that file and omit
`--no_consolidated` from the training and evaluation commands.

`--mode all` requests one decoder across all cameras and is incompatible with
their different embedding dimensions. The saved decoder configuration
`d5gwxcla` records `mode=tactile`, 15,000 iterations, and the RAE loss. That
metadata alone does not establish the provenance of each existing decoder
checkpoint. Decoder defaults are in `dino_wm/configs/default.yaml`.

For a separate decoder reconstruction evaluation:

```bash
python -m dino_wm.eval_dino_decoder_tactile \
  --hdf5_path "$TACTILE_DATASET" --no_consolidated \
  --checkpoint "$TACTILE_DECODERS/decoder_camera_2_last.pth" \
  --cameras camera_2 --num_test 20 --seed 42 \
  --output_dir outputs/tactile_decoder_evaluation
```

## Validation and source

```bash
python -m unittest discover -s tests -v
```

These tests check the active recipe against the saved run, matching training
and evaluation defaults, temporal alignment, split isolation, action gradients,
normalization, and preservation of source trajectories. They do not reproduce
a full training run or its reported metrics.

Dataset export checks also verify lossless values and dtypes, resume behavior,
preservation of terminal observations, and compatibility with both the loader
and evaluation output paths.

[`docs/recovered_runs.json`](docs/recovered_runs.json) contains the selected
world-model run and the supporting decoder configuration. It was recovered
from saved local W&B metadata, not a live W&B API response. The earlier
metadata collection, including RGB-only and tactile-only variants, is
preserved in the original research workspace archive.

The source is selected from the working tree of `CMU-IntentLab/latent-tactile`
at base commit `c6cd627ccf5114c364e36f4cdb76398f380e3d92`, including local changes.
The original MIT license is retained. See `SOURCE_MANIFEST.json` for the
source-file mapping and hashes.

## Citation

```bibtex
@misc{wu2026vital,
  title={Inference-time Policy Steering via Vision and Touch},
  author={Yilin Wu and Zilin Si and Zeynep Temel and Oliver Kroemer and Andrea Bajcsy},
  year={2026},
  eprint={2606.14981},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2606.14981}
}
```

The world-model implementation builds on
[DINO-WM](https://arxiv.org/abs/2411.04983). Tactile features use
[AnyTouch 2](https://arxiv.org/abs/2602.09617).
