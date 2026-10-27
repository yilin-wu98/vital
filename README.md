# ViTaL

**Inference-time Policy Steering via Vision and Touch**

[Paper](https://arxiv.org/abs/2606.14981) ·
[Project website](https://yilin-wu98.github.io/vital_website/) ·
[Datasets](https://huggingface.co/datasets/yilin-wu/ViTaL-datasets)

Yilin Wu, Zilin Si, Zeynep Temel, Oliver Kroemer, and Andrea Bajcsy.

ViTaL steers a diffusion policy at inference time using visual and tactile
rewards. This repository contains the active tactile world model, AnyTouch 2
reward scoring, and the liquid-transfer steering experiments.

| Package | Contents | Instructions |
| --- | --- | --- |
| `latent-tactile` | World-model training and evaluation, data preparation, image decoders | [README](latent-tactile/README.md) |
| `anytouch-reward` | Pretrained tactile/text encoders, video scoring, differentiable tactile reward | [README](anytouch-reward/README.md) |
| `liquid-tactile-control` | Liquid-task steering, sampling baselines, diffusion-policy adapter | [README](liquid-tactile-control/README.md) |

The datasets cover **pipette, insertion, and wiping**. The documented active
world-model recipe and bundled robot drivers are for **pipette/liquid transfer**.
Insertion and wiping require their own trained checkpoints and decoders.

## Installation

The reference environment is Linux with Python 3.12, PyTorch 2.4.0,
torchvision 0.19.0, and NumPy 1.26.4. From a fresh checkout:

```bash
git clone https://github.com/yilin-wu98/vital.git
cd vital
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 numpy==1.26.4
python -m pip install -e "./latent-tactile[decoder,tracking]" \
  -e ./anytouch-reward -e ./liquid-tactile-control
```

Training uses `cuda:0` by default. The package READMEs describe device overrides
and optional dependencies. Robot evaluation additionally requires configured
Deoxys/deoxys_vision installations, cameras, encoder weights, and a compatible
Robometer server.

## Datasets

The [Hugging Face release](https://huggingface.co/datasets/yilin-wu/ViTaL-datasets)
contains 980 episodes, 302,631 action frames, and 716.51 GB of losslessly
compressed HDF5 episode files. See [DATASETS.md](DATASETS.md) for the schema,
splits, normalization, and integrity checks.

**The initial dataset upload is in progress.** The release inventory describes
the complete intended dataset.

Download only the pipette task for the commands below:

```bash
python -m pip install "huggingface_hub>=0.36,<1"
hf download yilin-wu/ViTaL-datasets --repo-type dataset \
  --include "pipette/*" --local-dir datasets/vital
```

Replace `pipette/*` with `insertion/*` or `wiping/*` for another task, or omit
`--include` to download all three.

## Train the active world model

From the repository root:

```bash
export TACTILE_DATASET="$PWD/datasets/vital/pipette/episodes"
export TACTILE_NORM_STATS="$PWD/datasets/vital/pipette/normalization.json"
export WM_OUTPUT_DIR="$PWD/outputs/liquid_world_model"
bash latent-tactile/scripts/train_liquid.sh --no_consolidated
```

The active model is the **k-step concatenation model**, with all three cameras,
one history frame, one predicted frame, frame skip 8, multi-step loss, and noisy
actions. Its recorded training run is `xrjdyidg`.

The robot launchers select `wm_kstep_concat_h1_p1_fs8_iter67000.pth`.
Reproduction uses the original **100,000-iteration training budget** because
the action-noise schedule depends on that budget; 67,000 is the selected
intermediate checkpoint. See the
[full training recipe](latent-tactile/README.md#train-the-active-world-model).

## Evaluate

World-model and decoder weights are supplied separately from the dataset.
With the dataset variables above and the selected weights:

```bash
export WM_CHECKPOINT=/path/to/wm_kstep_concat_h1_p1_fs8_iter67000.pth
export TACTILE_DECODERS=/path/to/liquid_decoders
export WM_EVAL_DIR="$PWD/outputs/liquid_evaluation_67000"
bash latent-tactile/scripts/evaluate_liquid.sh --no_consolidated
```

Evaluation exports comparison videos and ground-truth/predicted embedding
files. Each prediction is conditioned on recorded history. The first eight
entries in the predicted embedding files are copied context and must be
excluded from prediction-error or reward calculations. This command does not
report aggregate paper metrics or robot success rates.

For the other stages:

- [AnyTouch reward evaluation](anytouch-reward/README.md#score-world-model-predictions)
  scores the exported tactile embeddings or raw tactile videos. It uses
  pretrained weights and has no separate reward-head training step.
- [Robot evaluation](liquid-tactile-control/README.md#configure-and-inspect)
  begins with configuration and file checks, then runs the selected method on
  the configured robot. Base-policy weights are required; base-policy training
  is outside this repository.
- [Decoder training and evaluation](latent-tactile/README.md#supporting-decoder-stage)
  are separate from world-model training.

## Validation

```bash
python -m unittest discover -s latent-tactile/tests -v
python -m unittest discover -s anytouch-reward/tests -v
python -m unittest discover -s liquid-tactile-control/tests -v
```

The release passed 29 tests, wheel build/install checks, existing-checkpoint
compatibility checks, and a real-episode world-model-to-reward execution check.
The full dataset export passed lossless verification.
[VALIDATION.md](VALIDATION.md) records the scope and limits of these checks.

## License and provenance

Component licenses and upstream notices are retained. See
[LICENSE.md](LICENSE.md) and each package's `SOURCE_MANIFEST.json`.
The dataset is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The release was selected from the research working trees, including local
changes. Original code, archived experimental variants, data, and checkpoints
remain preserved in the research workspace.

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

Please also cite the upstream methods and pretrained encoders described in
the component READMEs.
