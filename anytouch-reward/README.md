# AnyTouch Reward

[ViTaL paper](https://arxiv.org/abs/2606.14981)
· [Project website](https://yilin-wu98.github.io/vital_website/)
· [AnyTouch 2 paper](https://arxiv.org/abs/2602.09617)
· [Datasets](https://huggingface.co/datasets/yilin-wu/ViTaL-datasets)
· [Repository overview](../README.md)

Tactile-language reward prediction using the AnyTouch 2 encoder and the CLIP
text branch stored in its checkpoint. The liquid experiments compare
“grasp lightly” with “grasp heavily”; they do not train a separate scalar
reward head.

**Contents:** [Installation](#installation) ·
[Training requirements](#training-requirements) ·
[Video evaluation](#score-a-tactile-video) ·
[World-model reward](#score-world-model-predictions) ·
[Citation](#citation)

## Installation

From the `anytouch-reward` root, create an environment using the reference
Linux/Python 3.12 setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 numpy==1.26.4
python -m pip install -e .
```

If the `latent-tactile` environment is already active, only the final
installation command is needed.

Python 3.10+ is required. `transformers==4.42.3` is pinned because the original
encoder uses CLIP implementation details. PyTorch 2.4.0 and torchvision 0.19.0
match the archived training environment.

Supply `checkpoint-4frames.pth` from AnyTouch 2 separately. Small CLIP config
and tokenizer assets are included, so checkpoint loading does not depend on
the current working directory. Set `ANYTOUCH_CLIP_DIR` to override them.
Model weights are not bundled.

The upstream model repository is
[`xxuan01/AnyTouch2-Model`](https://huggingface.co/xxuan01/AnyTouch2-Model).
Obtain access there if required and download `checkpoint-4frames.pth`; the
ViTaL dataset release does not redistribute this third-party checkpoint.

## Training requirements

There is no reward-head training step in this package. It loads the pretrained
AnyTouch 2 tactile and text branches and computes text-conditioned cosine
similarities. Train the world model in the `latent-tactile` package, then score
its predicted tactile embeddings using the example below.

Retraining the AnyTouch 2 encoder is a separate upstream workflow; see the
[AnyTouch 2 repository](https://github.com/GeWu-Lab/AnyTouch2).

## Score a tactile video

```bash
export ANYTOUCH_CHECKPOINT=/path/to/checkpoint-4frames.pth
bash scripts/score_liquid.sh /path/to/camera_2.mp4
```

The command uses GelSight Mini, four-frame windows, temporal stride 2, and
mean patch pooling. It writes a CSV, plot, and annotated video. For a video
arranged as `[front 256px | wrist 256px | tactile 340px]`, append
`--crop_tactile`. Marker-motion overlays are optional and require `gsmini`;
reward inference works without it. Use `--no-save_video` to skip video output.

Each window is background-subtracted, offset by `130/255`, clamped, resized to
224 × 224, and CLIP-normalized. The first frame is the default background;
`--bg_image` supplies a separate no-contact reference. Pooling skips the CLS
token and five sensor tokens, then normalizes the pooled feature.

## Score world-model predictions

Use the `*_pred_embeds.npz` file produced by the active world-model evaluator:

```python
import numpy as np
import torch
import torch.nn.functional as F
from anytouch_reward.loading import load_text_encoder
from anytouch_reward.reward import score_patch_embeddings

model, tokenizer = load_text_encoder("/path/to/checkpoint-4frames.pth", device="cpu")
inputs = tokenizer(["grasp lightly", "grasp heavily"],
                   padding=True, return_tensors="pt")
with torch.no_grad():
    text = F.normalize(model.get_text_features(**inputs), dim=-1)

# For the active H1/P1/frame-skip-8 evaluator, the first eight entries are
# copied ground-truth context. Score only the genuinely predicted entries.
with np.load("/path/to/episode_0_traj_trajectory_0_pred_embeds.npz") as saved:
    predicted_patches = torch.from_numpy(saved["camera_2"][8:].copy()).float()

# Shape: (predicted_frames, patch_count, 512), with no special tokens.
reward = score_patch_embeddings(predicted_patches, text[:1], text[1:])
np.savetxt("tactile_reward.csv", reward.detach().cpu().numpy(),
           delimiter=",", header="light_minus_heavy", comments="")
```

The contrastive reward is
`cos(mean(patches), positive_text) - cos(mean(patches), negative_text)`.
Swap the two text embeddings for the second phase. The function retains
gradients for action guidance when passed predictions directly from the
world model; arrays loaded from an NPZ file have no model computation graph.
Scores are similarities, not calibrated success probabilities.

Offline video pooling and the world model's fixed 196-patch representation
are distinct; preserve the world-model preprocessing when scoring its
predictions. Checkpoint loading rejects missing learned encoder or text
weights instead of silently initializing them at random.

## Validation and attribution

```bash
python -m unittest discover -s tests -v
```

Tests cover special-token exclusion, cosine scoring, phase reversal, and
gradients. The encoder and video-scoring code are selected from the local
`GeWu-Lab/AnyTouch2` working tree at base commit
`3ffff32bce508b376f1d875a3f6829bff2f9c9c8`, including local changes.
The original MIT license is retained; CLIP asset attribution is in
`THIRD_PARTY_NOTICES.md`. See `SOURCE_MANIFEST.json` for file provenance.

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

Please also cite the upstream [AnyTouch 2 paper](https://arxiv.org/abs/2602.09617)
when using its encoder or pretrained weights.
