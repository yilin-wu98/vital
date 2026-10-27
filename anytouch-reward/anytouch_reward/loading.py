"""Local checkpoint loading shared by video scoring and world-model inference."""
from argparse import Namespace
from pathlib import Path
import os

import torch
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

from .model import TactileVideoMAE

SENSOR_NAME_TO_ID = {
    "gelsight": 0, "digit": 1, "gelslim": 2, "gelsight_mini": 3,
    "duragel": 4, "dm": 5, "universal": -1,
}


def resolve_clip_dir(clip_dir=None, checkpoint_path=None):
    """Resolve assets independently of the process's working directory."""
    explicit = clip_dir or os.environ.get("ANYTOUCH_CLIP_DIR")
    if explicit:
        path = Path(explicit).expanduser()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"CLIP config not found in {path}")
        return path
    if checkpoint_path:
        adjacent = Path(checkpoint_path).expanduser().resolve().parent.parent / "CLIP-B-16"
        if (adjacent / "config.json").is_file():
            return adjacent
    return Path(__file__).resolve().parent / "assets" / "clip"


def _state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    for key in ("state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    result = {}
    for key, value in checkpoint.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        result[key] = value
    return result


def tactile_model_from_state(checkpoint, clip_dir=None, *, num_frames=4,
                             stride=2, model_size="base"):
    """Load the encoder; reject missing learned weights instead of using random ones."""
    if num_frames <= 0 or stride <= 0 or num_frames % stride:
        raise ValueError("num_frames must be positive and divisible by stride")
    config = CLIPConfig.from_pretrained(str(resolve_clip_dir(clip_dir)), local_files_only=True)
    args = Namespace(
        model="anytouch", model_size=model_size, dataset="material",
        pooling="global", mask_ratio=0, stride=stride, num_frames=num_frames,
        load_from_clip=False,
    )
    model = TactileVideoMAE(args, config, num_frames, 1)
    selected = {}
    expected = model.state_dict()
    for key, value in _state_dict(checkpoint).items():
        for prefix in ("touch_mae_model.", "tactile_model.", ""):
            if key.startswith(prefix):
                clean = key[len(prefix):]
                if clean in expected:
                    selected[clean] = value
                    break
    # This fixed index tensor was not serialized by some original checkpoints.
    selected.setdefault("new_position_ids", expected["new_position_ids"])
    missing = sorted(set(expected) - set(selected))
    if missing:
        raise ValueError(f"Incomplete tactile encoder checkpoint; missing {missing[:8]}")
    model.load_state_dict(selected, strict=True)
    return model.eval()


def text_model_from_state(checkpoint, clip_dir=None, device="cpu"):
    """Load the CLIP text branch contained in the AnyTouch checkpoint."""
    directory = resolve_clip_dir(clip_dir)
    config = CLIPConfig.from_pretrained(str(directory), local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = CLIPModel(config)
    expected = {
        key for key in model.state_dict()
        if key.startswith(("text_model.", "text_projection."))
    }
    selected = {}
    for key, value in _state_dict(checkpoint).items():
        for prefix in ("", "clip_model.", "hf_clip_model.", "clip."):
            if key.startswith(prefix) and key[len(prefix):] in expected:
                selected[key[len(prefix):]] = value
                break
    missing = sorted(expected - set(selected))
    if missing:
        raise ValueError(f"Incomplete CLIP text checkpoint; missing {missing[:8]}")
    model.load_state_dict(selected, strict=False)  # The unused vision branch is not required.
    model.requires_grad_(False)
    return model.to(device).eval(), tokenizer


def load_tactile_encoder(checkpoint_path, sensor="gelsight_mini", *,
                         num_frames=4, stride=2, device="cpu", clip_dir=None):
    if sensor not in SENSOR_NAME_TO_ID:
        raise ValueError(f"Unknown tactile sensor: {sensor}")
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    directory = resolve_clip_dir(clip_dir, path)
    model = tactile_model_from_state(
        checkpoint, directory, num_frames=num_frames, stride=stride,
    )
    model.requires_grad_(False)
    return model.to(device).eval(), SENSOR_NAME_TO_ID[sensor]


def load_text_encoder(checkpoint_path, device="cpu", clip_dir=None):
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return text_model_from_state(
        checkpoint, resolve_clip_dir(clip_dir, path), device,
    )
