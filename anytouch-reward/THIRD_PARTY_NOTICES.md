# Third-party source and assets

- `anytouch_reward/model.py` derives from `model/tactile_mae.py` in
  GeWu-Lab/AnyTouch2. `anytouch_reward/video.py` derives from the local
  `scripts/infer_touch_text_similarity.py`. The original MIT license and
  GeWu-Lab copyright notice are retained in `LICENSE`.
- `anytouch_reward/assets/clip` contains the existing CLIP-B/16 configuration
  and tokenizer files used by that checkout. CLIP is an OpenAI project; its
  MIT license is retained in `assets/clip/LICENSE.txt`. That license was
  retrieved from the official `openai/CLIP` repository on 2026-10-27.
- PyTorch, torchvision, transformers, and other installed dependencies retain
  their own licenses. Their model weights are not distributed here.

Publication cleanup changes include a namespaced package, shared checkpoint
loaders, validation of required weights, portable asset resolution, optional
marker-tracking imports, and a reusable differentiable reward function.
The encoder architecture and checkpoint tensor names are preserved.
