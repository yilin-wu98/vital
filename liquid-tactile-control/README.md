# Liquid Tactile Control

[ViTaL paper](https://arxiv.org/abs/2606.14981)
· [Project website](https://yilin-wu98.github.io/vital_website/)
· [Datasets](https://huggingface.co/datasets/yilin-wu/ViTaL-datasets)
· [Repository overview](../README.md)

Liquid-transfer experiments that use a tactile world model to steer a
diffusion policy. The main method samples candidate action sequences, ranks
two-step visual predictions with Robometer, then re-noises and refines the
selected first chunk using a tactile-language reward gradient.

**Contents:** [Installation](#install-and-dependencies) ·
[Training inputs](#training-inputs) ·
[Configuration](#configure-and-inspect) ·
[Robot evaluation](#run-robot-evaluation) ·
[Methods](#methods) ·
[Validation](#logs-and-validation) ·
[Citation](#citation)

## Install and dependencies

Activate the Python environment prepared for `latent-tactile`. From the
`liquid-tactile-control` root, with the other two packages in sibling folders:

```bash
python -m pip install -e ../latent-tactile -e ../anytouch-reward -e .
```

The robot machine also needs its configured **Deoxys** and **deoxys_vision**
installations, camera streams, DINOv3 checkout and weights, and a compatible
Robometer streaming server. Hardware configuration files and model weights
are supplied separately.

The liquid drivers also require the local Deoxys `continuous_smooth` gripper
extension. `patches/` preserves the exact compatibility patch and explains
how to apply it to a fresh checkout. The research-machine checkout already
contains these changes.

The modified diffusion-policy modules needed by these experiments are
included in `liquid_tactile_control/policy_backend`. The checkpoint's
`data4robotics.*` Hydra targets are mapped to these modules when loading.
The adapter loads the full trained state and does not download ImageNet
initialization weights first.

External dependency snapshots used during extraction:

| Dependency | Source base commit |
| --- | --- |
| Deoxys | `5ca447a5df7d1cefadf43fbfba7fd5f147383d7c` |
| deoxys_vision | `00ca0c5ca83979d105011269bb337d84414b8ff9` |
| Robometer streaming protocol | `e0759915e2a8edc4ca24ec56ed897ed87e48b6fd` |
| Bundled diffusion4robotics modules | `4f705a3b1d8963c2bd018785bd13b06cadde2312` plus local changes |

These commits identify the source trees. The required gripper extension is
captured in `patches/`; robot deployment still depends on the configured local
hardware stack, which this release does not install.

## Training inputs

This package evaluates pretrained policies on the robot. World-model training
is provided by `latent-tactile/scripts/train_liquid.sh`; the AnyTouch reward
uses pretrained encoder weights and needs no separate reward-head training.

Supply the base diffusion-policy checkpoint and configuration files listed
below. The bundled `policy_backend` contains the model implementation and
inference code needed to load the checkpoint, not a base-policy training
pipeline. The world-model dataset download alone does not provide trained
weights or a turnkey base-policy training command.

## Configure and inspect

Copy `configs/liquid.example.json` to a local config and fill in the paths.
Relative paths are resolved against the config's directory.

```bash
cp configs/liquid.example.json configs/liquid.local.json
```

```bash
python -m liquid_tactile_control.run \
  --config configs/liquid.local.json --method two-step --dry-run

python -m liquid_tactile_control.run \
  --config configs/liquid.local.json --method two-step --check
```

`--dry-run` prints the command without importing robot code. `--check` checks
local input files and exits without contacting the robot, cameras, or reward
server. Removing both flags runs the experiment on the configured robot.

The policy directory must contain `itr_23999.ckpt`, `agent_config.yaml`,
`exp_config.yaml`, `obs_config.yaml`, `ob_norm.json`, `ac_norm.json`, and
`.hydra/config.yaml`. Change `policy_iteration` for another checkpoint.

The liquid world-model recipe uses the 67k concatenation checkpoint, all three
cameras, one history frame, one predicted frame, and eight actions per step.
Use its matching normalization file. The old launchers referred to a missing
`tactile_liquid_wm_info.json`; the archived training used
`tactile_liquid_wm_liquid.json`. Correcting this path can change results
relative to a run that silently used fallback statistics.

## Run robot evaluation

After the configuration checks pass, run the selected method on the configured
robot:

```bash
python -m liquid_tactile_control.run \
  --config configs/liquid.local.json --method two-step
```

This command executes the physical experiment. Select another method from
the table below to evaluate a baseline. The dataset release covers pipette,
insertion, and wiping demonstrations; this control package currently contains
the **liquid-task drivers**.

## Methods

| Method | Driver | Behavior |
| --- | --- | --- |
| `two-step` | `drivers/two_step.py` | Visual ranking, then tactile-gradient refinement |
| `tactile-guidance` | `drivers/tactile_guidance.py` | Tactile guidance during diffusion |
| `tactile-sampling` | `drivers/tactile_sampling.py` | Rank sampled actions by tactile reward |
| `vision-sampling` | `drivers/vision_sampling.py` | Rank two-step visual predictions |
| `combined-sampling` | `drivers/combined_sampling.py` | Original combined-reward selection baseline |
| `baseline` | `drivers/baseline.py` | Unguided diffusion policy |

Presets retain the effective options from the old liquid shell scripts,
removing duplicate arguments. They intentionally retain differences between
experiments: the combined baseline names the blue cup, the main method names
the yellow cup, and the tactile-sampling launcher did not enable
`separate_phase`. Set entries in `method_options` to change these deliberately:

```json
{
  "method_options": {
    "tactile-sampling": {"separate_phase": true},
    "combined-sampling": {"reward_task": "transfer liquid to the yellow cup"}
  }
}
```

Robometer is used for visual scoring and for phase switching when enabled.
The client expects `/stream/create`, `/stream/{id}/append`,
`/stream/{id}/evaluate_candidates`, and `DELETE /stream/{id}`.
It implements this wire format without importing the Robometer repository.
The server itself is not included.

## Logs and validation

Drivers retain the original HDF5/video logging and robot control loops.
The example configuration places outputs under the repository's `outputs/`
directory; change `output_dir` in the local JSON to select another location.
`analysis/` contains the standalone plotting scripts for two-step, visual,
and combined sampling logs.

```bash
python -m unittest discover -s tests -v
```

Tests run without hardware and cover temporal context alignment, checkpoint
target mapping, all preset parsers, missing-file handling, and the HTTP
payload format with mocked requests. Offline validation does not establish
robot behavior or experimental success rates; these require a hardware run.

## Attribution

Drivers and the policy adapter are selected from the local
`CMU-IntentLab/deoxys_control` working tree, including untracked experiments.
The Apache 2.0 license is retained. The bundled policy backend retains its MIT
license and source notices. `THIRD_PARTY_NOTICES.md` and
`SOURCE_MANIFEST.json` record provenance and cleanup changes.

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

The base policy uses [Diffusion Policy](https://arxiv.org/abs/2303.04137).
