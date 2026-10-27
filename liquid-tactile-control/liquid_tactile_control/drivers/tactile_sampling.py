# Publication cleanup (2026-10-27): adapted from deoxys_control/deoxys/examples/run_dp_model_tactile_sampling_liquid_log.py.
"""Replay demonstration trajectories with tactile-based action selection."""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import h5py
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as _tv_transforms
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer
# Add latent-tactile for dino_wm imports (sibling of deoxys_control)


from liquid_tactile_control.robometer_client import (
    stream_create,
    stream_append_frame,
    stream_delete,
)

from liquid_tactile_control.robot_config import robot_config_parse_args
from liquid_tactile_control.policy import RealPolicy

from dino_wm.dino_models import TactileVideoTransformer, NormStats
from dino_wm.dino_models_k_step import TactileKStepTransformer

import logging
logger = logging.getLogger(__name__)

ACT_HORIZON = 8

# World model constants (from eval_dino_wm_tactile)
CAMERA_EMB_DIMS = {"camera_0": 768, "camera_1": 768, "camera_2": 512}
PATCH_SIDE_DINOV3 = 14

# CLIP-style tactile preprocessing (match training; tensors cached per device)
_TACTILE_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_TACTILE_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_TACTILE_BG_OFFSET = 130.0 / 255.0
_TACTILE_NORM_CACHE = {}


def _tactile_clip_norm_tensors(device):
    """(3,1,1) mean/std on ``device`` for broadcasting over (4,3,H,W)."""
    d = device if isinstance(device, torch.device) else torch.device(device)
    key = str(d)
    if key not in _TACTILE_NORM_CACHE:
        _TACTILE_NORM_CACHE[key] = (
            torch.tensor(_TACTILE_CLIP_MEAN, device=d, dtype=torch.float32).view(3, 1, 1),
            torch.tensor(_TACTILE_CLIP_STD, device=d, dtype=torch.float32).view(3, 1, 1),
        )
    return _TACTILE_NORM_CACHE[key]


# DINOv3 preprocessing (same as guidance script)
_DINO_TRANSFORM_CAM0 = _tv_transforms.Compose([
    _tv_transforms.Resize(224),
    _tv_transforms.ToTensor(),
    _tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_DINO_TRANSFORM_CAM1 = _tv_transforms.Compose([
    _tv_transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1)),
    _tv_transforms.Resize((224, 224)),
    _tv_transforms.ToTensor(),
    _tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_vision_encoder(hub_dir, weights_path, device):
    dino = torch.hub.load(hub_dir, "dinov3_vitb16", source="local", weights=weights_path)
    return dino.to(device).eval()


def get_vision_embedding(encoder, img_np, camera_id, device):
    """Encode a single BGR uint8 numpy image → (1, 196, 768) patch embeddings."""
    img_pil = PILImage.fromarray(np.uint8(img_np)).convert("RGB")
    transform = _DINO_TRANSFORM_CAM0 if camera_id == 0 else _DINO_TRANSFORM_CAM1
    img_t = transform(img_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        return encoder.forward_features(img_t)["x_norm_patchtokens"]  # (1, 196, 768)


def parse_args():
    parser = argparse.ArgumentParser()
    robot_config_parse_args(parser)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ckpt_iter', type=str, default=23999)
    parser.add_argument('--use_tactile', action='store_true', default=False)
    parser.add_argument('--n_samples', type=int, default=10)
    # Tactile-based action selection (when use_tactile)
    parser.add_argument('--wm_checkpoint', type=str, default=None,
                        help='Path to world model checkpoint (.pth) for dynamics prediction')
    parser.add_argument('--anytouch_checkpoint', type=str, default=None,
                        help='Path to AnyTouch checkpoint for tactile encoding')
    parser.add_argument('--reward_text', type=str, default='wiping the board',
                        help='Text for reward (tactile-text similarity)')
    parser.add_argument('--action_select', type=str, default='argmax',
                        choices=['argmax', 'softmax', 'argmin'],
                        help='argmax: select action with highest reward; softmax: sample proportional to reward')
    parser.add_argument('--softmax_temp', type=float, default=1.0,
                        help='Temperature for softmax action selection (higher=more random)')
    parser.add_argument('--norm_stats_json', type=str, default=None,
                        help='Path to norm_stats.json for action normalization. Default: norm_stats.json in wm_checkpoint dir')
    parser.add_argument('--decoder_dir', type=str, default=None,
                        help='Path to decoder directory (decoder_camera_2.pth) for visualizing pred_embeds')
    parser.add_argument('--visualize_pred', action='store_true',
                        help='Decode and visualize predicted tactile embeddings (requires --decoder_dir)')
    parser.add_argument('--plot_reward', action='store_true',
                        help='Save reward vs action-dim plots (3 subplots) each WM query')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile on world model (compile is on by default for speed)')
    parser.add_argument('--amp', action='store_true',
                        help='Use FP16 autocast for world model and tactile encoder inference')
    parser.add_argument(
        '--wm_rollout_steps',
        type=int,
        default=None,
        metavar='K',
        help=(
            'Video WM only: autoregressive rollout depth (speed benchmarking). '
            'Default: full action chunk length. Clamped to [1, chunk_T]. '
            'Ignored for k_step WM (single forward, num_pred predictions from current chunk).'
        ),
    )
    parser.add_argument(
        '--wm_arch',
        type=str,
        choices=['video', 'k_step'],
        default='video',
        help=(
            'video: TactileVideoTransformer (one 7-dim action per temporal frame, num_frames=9). '
            'k_step: TactileKStepTransformer (collapsed frameskip actions). '
            'Match train_dino_wm_tactile_k_step.py / latent-tactile k-step training.'
        ),
    )
    parser.add_argument(
        '--wm_cameras',
        type=str,
        default='camera_2',
        help=(
            'Comma-separated camera list used in the WM checkpoint (must match training). '
            'E.g. "camera_0,camera_1,camera_2" for a 3-camera / 3-head checkpoint. '
            'Default: camera_2 (tactile only). '
            'Non-tactile cameras receive zero embeddings during inference.'
        ),
    )
    parser.add_argument(
        '--wm_condition_cameras',
        type=str,
        default=None,
        help='Comma-separated conditioning cameras for WM. Default: same as --wm_cameras.',
    )
    parser.add_argument(
        '--wm_predict_cameras',
        type=str,
        default=None,
        help=(
            'Comma-separated prediction-head cameras for WM. Default: same as --wm_cameras. '
            'Reward is always computed from the camera_2 (tactile) head regardless of this list.'
        ),
    )
    parser.add_argument(
        '--temporal_mode',
        type=str,
        choices=['lpb', 'consecutive'],
        default='lpb',
        help=(
            'K-step only: history layout (same as tactile_k_step_dataset.py). '
            'lpb: subsampled indices t_last-(nh-1)*fs..t_last. '
            'consecutive: dense t_last-nh+1..t_last. Ignored for wm_arch=video.'
        ),
    )
    parser.add_argument(
        '--num_hist',
        type=int,
        default=None,
        help='K-step only: conditioning frames (must match checkpoint). Default: 1.',
    )
    parser.add_argument(
        '--num_pred',
        type=int,
        default=1,
        help='K-step only: prediction heads per forward (must match checkpoint). Default: 1.',
    )
    parser.add_argument(
        '--frameskip',
        type=int,
        default=8,
        help='K-step only: raw actions per collapsed slot (must match checkpoint). Default: 8.',
    )
    # ── Two-phase text switching via robometer ────────────────────────────────
    parser.add_argument(
        '--eval_server_url', type=str, default='http://localhost:8000',
        help='Robometer streaming eval server URL (default: port 8000).',
    )
    parser.add_argument(
        '--robometer_timeout_s', type=float, default=30.0,
        help='HTTP timeout for robometer calls (default: 30 s)',
    )
    parser.add_argument(
        '--separate_phase', action='store_true', default=False,
        help='Enable two-phase guidance: robometer evaluates real camera_0 frames; '
             'when the reward exceeds --phase_threshold the guidance text switches to '
             '--reward_text_phase2.',
    )
    parser.add_argument(
        '--mixed_reward', action='store_true', default=False,
        help='Use a contrastive reward: r(text_pos, tactile) - r(text_neg, tactile). '
             'Phase 1: text_pos=--reward_text, text_neg=--reward_text_phase2. '
             'Phase 2 (after phase switch): flipped. '
             'Requires --reward_text_phase2 to be set.',
    )
    parser.add_argument(
        '--reward_text_phase2', type=str, default='',
        help='CLIP reward text used for phase-2 guidance (used when --separate_phase is set)',
    )
    parser.add_argument(
        '--phase_threshold', type=float, default=0.64,
        help='Robometer reward threshold on real observations that triggers the '
             'phase-1 → phase-2 text switch (default: 0.64)',
    )
    parser.add_argument(
        '--binary_gripper', action='store_true', default=False,
        help='Binarize gripper action to +1.0 (open) or -1.0 (close) before sending to robot.',
    )
    # DINOv3 for RGB camera embeddings (camera_0, camera_1)
    parser.add_argument(
        '--dinov3_hub_dir', type=str,
        default=None,
    )
    parser.add_argument(
        '--dinov3_weights', type=str,
        default=None,
    )
    return parser.parse_args()

def ensure_patch_grid(embd, target_side, device='cuda'):
    """Reshape or interpolate patch embeddings to target_side x target_side grid."""
    import torch.nn as nn
    B, T, N, D = embd.shape
    side = int(N ** 0.5)
    if side * side != N:
        target_n = target_side * target_side
        if N >= target_n:
            embd = embd[:, :, :target_n, :]
        else:
            pad = torch.zeros(B, T, target_n - N, D, device=device, dtype=embd.dtype)
            embd = torch.cat([embd, pad], dim=2)
        N = target_n
        side = target_side
    if side != target_side:
        embd = rearrange(embd, "b t (h w) d -> (b t) d h w", h=side, w=side)
        embd = nn.functional.interpolate(
            embd, size=(target_side, target_side), mode="bilinear", align_corners=False
        )
        embd = rearrange(embd, "(b t) d h w -> b t (h w) d", h=target_side, w=target_side)
    return embd


def _strip_compile_prefix(state_dict):
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k.replace(prefix, "", 1): v for k, v in state_dict.items()}


def _find_decoder_path(decoder_dir, base_name):
    """Try base, _adv, _last suffixes."""
    for suffix in ("", "_adv", "_last"):
        path = os.path.join(decoder_dir, f"{base_name}{suffix}.pth")
        if os.path.isfile(path):
            return path
    return None


def load_tactile_decoder(decoder_dir, device='cuda'):
    """Load VQVAE decoder for camera_2 (tactile, 512-dim)."""
    tactile_path = _find_decoder_path(decoder_dir, "decoder_camera_2_last")
    if not tactile_path:
        return None
    import dino_wm.dino_decoder as _dino_dec_mod
    if not hasattr(_dino_dec_mod, "dist_fn"):
        _dino_dec_mod.dist_fn = type("DistFn", (), {"all_reduce": lambda x: None})()
    from dino_wm.dino_decoder import VQVAE
    dec = VQVAE(emb_dim=512).to(device)
    dec.load_state_dict(torch.load(tactile_path, map_location=device))
    dec.eval()
    return dec


def decode_pred_embeds(decoder, pred_embeds, device='cuda'):
    """
    Decode predicted tactile embeddings to images.
    pred_embeds: (N, N_patches, 512)
    Returns: list of (H, W, 3) numpy arrays in [0, 1]
    """
    N, n_patches, D = pred_embeds.shape
    side = int(n_patches ** 0.5)
    if side * side != n_patches:
        return []
    inp = pred_embeds.unsqueeze(1).to(device)
    with torch.no_grad():
        pred_img, _ = decoder(inp)
    pred_img = rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)
    images = []
    for i in range(N):
        img = pred_img[i, 0].cpu().numpy().transpose(1, 2, 0)
        images.append(np.clip(img, 0, 1))
    return images


def load_tactile_encoder(checkpoint_path, sensor='gelsight_mini', num_frames=4, device='cuda'):
    """Load AnyTouch tactile encoder for single-frame embedding."""
    from dino_wm.hdf5_to_dataset_tactile import _load_tactile_model
    model, sensor_id = _load_tactile_model(
        checkpoint_path, sensor, num_frames=num_frames, device=str(device)
    )
    return model, sensor_id


def load_world_model_video(
    checkpoint_path,
    device='cuda',
    use_compile=True,
    cameras=None,
    condition_cameras=None,
    predict_cameras=None,
):
    """Load TactileVideoTransformer (per-frame actions). Optionally torch.compile."""
    cameras = cameras or ["camera_2"]
    condition_cameras = condition_cameras or cameras
    predict_cameras = predict_cameras or cameras
    camera_dims = {c: CAMERA_EMB_DIMS.get(c, 512) for c in cameras}
    transition = TactileVideoTransformer(
        cameras=cameras,
        camera_dims=camera_dims,
        condition_cameras=condition_cameras,
        predict_cameras=predict_cameras,
        common_dim=384,
        ac_dim=10,
        state_dim=8,
        patches_per_frame=PATCH_SIDE_DINOV3 * PATCH_SIDE_DINOV3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=9,
        dropout=0.1,
    ).to(device)
    wm_state = torch.load(checkpoint_path, map_location=device)
    transition.load_state_dict(_strip_compile_prefix(wm_state))
    transition.eval()
    if use_compile and hasattr(torch, "compile"):
        transition = torch.compile(transition, mode="reduce-overhead")
        logger.info("Using torch.compile (mode=reduce-overhead) on world model (video)")
    return transition


def load_world_model_k_step(
    checkpoint_path,
    device='cuda',
    use_compile=True,
    num_hist=1,
    num_pred=1,
    frameskip=8,
    cameras=None,
    condition_cameras=None,
    predict_cameras=None,
):
    """
    Load TactileKStepTransformer (LPB-style collapsed actions).
    Hyperparameters must match train_dino_wm_tactile_k_step.py / checkpoint.
    cameras/condition_cameras/predict_cameras must match training configuration.
    When a multi-camera checkpoint is loaded, non-tactile cameras receive zero embeddings
    during inference; reward is always computed from the camera_2 (tactile) head.
    """
    cameras = cameras or ["camera_2"]
    condition_cameras = condition_cameras or cameras
    predict_cameras = predict_cameras or cameras
    camera_dims = {c: CAMERA_EMB_DIMS.get(c, 512) for c in cameras}
    transition = TactileKStepTransformer(
        cameras=cameras,
        camera_dims=camera_dims,
        condition_cameras=condition_cameras,
        predict_cameras=predict_cameras,
        common_dim=384,
        ac_dim=64,
        state_dim=8,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
        patches_per_frame=PATCH_SIDE_DINOV3 * PATCH_SIDE_DINOV3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
    ).to(device)
    wm_state = torch.load(checkpoint_path, map_location=device)
    transition.load_state_dict(_strip_compile_prefix(wm_state))
    transition.eval()
    if use_compile and hasattr(torch, "compile"):
        transition = torch.compile(transition, mode="reduce-overhead")
        logger.info("Using torch.compile (mode=reduce-overhead) on world model (k_step)")
    return transition


def load_world_model(
    checkpoint_path,
    device='cuda',
    use_compile=True,
    wm_arch='video',
    num_hist=1,
    num_pred=1,
    frameskip=8,
    cameras=None,
    condition_cameras=None,
    predict_cameras=None,
):
    """Dispatch: video = TactileVideoTransformer; k_step = TactileKStepTransformer."""
    if wm_arch == 'k_step':
        return load_world_model_k_step(
            checkpoint_path, device, use_compile, num_hist, num_pred, frameskip,
            cameras=cameras, condition_cameras=condition_cameras, predict_cameras=predict_cameras,
        )
    return load_world_model_video(
        checkpoint_path, device, use_compile,
        cameras=cameras, condition_cameras=condition_cameras, predict_cameras=predict_cameras,
    )


def load_clip_for_reward(checkpoint_path, device):
    from anytouch_reward.loading import load_text_encoder
    return load_text_encoder(checkpoint_path, device=device)


def _preprocess_tactile_frame(img):
    """Convert tactile image to RGB uint8."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.max() <= 1.0:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def _tactile_window_rgb_to_chw_stack(window_rgb_uint8):
    """window_rgb_uint8: list of H,W,3 uint8 -> (4,3,H,W) float32 [0,1] CPU tensor."""
    batch = np.stack(window_rgb_uint8, axis=0)
    x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).contiguous()
    return x.float().mul_(1.0 / 255.0)


def _rgb_uint8_chw01_match_hw(bg_u8: np.ndarray, ref_hw) -> np.ndarray:
    """Resize ``bg_u8`` (H,W,3) to ``ref_hw`` (height, width) if needed."""
    h, w = ref_hw
    if bg_u8.shape[0] == h and bg_u8.shape[1] == w:
        return bg_u8
    return cv2.resize(bg_u8, (w, h), interpolation=cv2.INTER_AREA)


def get_tactile_embedding_from_image(
    tactile_model,
    tactile_frames,
    sensor_id,
    device="cuda",
    use_amp=False,
    bg_image=None,
):
    """
    Encode tactile image to patch embeddings using last 4 frames.
    tactile_frames: list/deque of 1-4 (H, W) or (H, W, 3) numpy arrays. When len < 4,
        pads with the first frame (like build_tactile_windows_per_timestep).
    bg_image: optional RGB tactile **episode** reference (e.g. first frame of the trajectory).
        When set, each frame is processed as ``frame - bg_image + OFFSET`` (matches
        ``preprocess_tactile_window`` / dataset & plot GT reward). When None, falls back to
        subtracting the first frame of the **current window** (legacy behavior).
    Returns: (1, N_patches, 512) tensor

    Avoids PIL and per-frame torchvision Resize (faster: one H2D transfer + batched GPU resize).
    """
    frames = [_preprocess_tactile_frame(f) for f in tactile_frames]
    if not frames:
        raise ValueError("tactile_frames must contain at least 1 frame")
    pad_ref = frames[0]
    n = len(frames)
    if n < 4:
        window_frames = [pad_ref] * (4 - n) + frames
    else:
        window_frames = frames[-4:]

    stacked = _tactile_window_rgb_to_chw_stack(window_frames)
    dev = device if isinstance(device, torch.device) else torch.device(device)
    non_blocking = dev.type == "cuda" and torch.cuda.is_available()
    x = stacked.to(dev, non_blocking=non_blocking)
    ref_hw = window_frames[0].shape[:2]
    if bg_image is not None:
        bg_u8 = _preprocess_tactile_frame(bg_image)
        bg_u8 = _rgb_uint8_chw01_match_hw(bg_u8, ref_hw)
        bg_chw = _tactile_window_rgb_to_chw_stack([bg_u8])[0].to(dev)
        x = x - bg_chw.unsqueeze(0) + _TACTILE_BG_OFFSET
    else:
        x = x - x[0:1] + _TACTILE_BG_OFFSET
    x = x.clamp_(0.0, 1.0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean, std = _tactile_clip_norm_tensors(dev)
    x = (x - mean) / std
    imgs = x.unsqueeze(0)

    sensors = torch.full((1,), sensor_id, dtype=torch.long, device=dev)
    with torch.inference_mode():
        if use_amp and dev.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = tactile_model(
                    imgs, sensor_type=sensors, probe=False, get_cls=False
                )
        else:
            outputs = tactile_model(imgs, sensor_type=sensors, probe=False, get_cls=False)
    patch_emb = outputs[:, 6:, :]
    patch_emb = F.normalize(patch_emb, dim=-1)
    return patch_emb


def _is_k_step_wm(m):
    """
    True if world model is TactileKStepTransformer.
    torch.compile wraps the module, so isinstance(m, TactileKStepTransformer) is often False;
    we also check k-step-only attributes.
    """
    if isinstance(m, TactileKStepTransformer):
        return True
    orig = getattr(m, "_orig_mod", None)
    if isinstance(orig, TactileKStepTransformer):
        return True
    return getattr(m, "action_per_frame_dim", None) is not None


def _wm_condition_cameras(wm_model):
    """Return condition_cameras list, unwrapping torch.compile if needed."""
    orig = getattr(wm_model, "_orig_mod", wm_model)
    return list(getattr(orig, "condition_cameras", ["camera_2"]))


def _build_wm_inputs(
    tactile_inp,
    wm_model,
    N,
    device,
    vision_emb_histories=None,
    temporal_mode="lpb",
):
    """
    Build the camera_inputs dict for a WM forward pass.
    camera_2 gets real tactile embeddings; other cameras use real vision embeddings
    when vision_emb_histories is provided, otherwise fall back to zeros.

    tactile_inp: (N, T, P, D) already expanded to batch size N.
    vision_emb_histories: dict cam_name → deque of (1, n_patches, D) embeddings.
    """
    condition_cameras = _wm_condition_cameras(wm_model)
    T = tactile_inp.shape[1]
    P = tactile_inp.shape[2]
    inputs = {}
    for cam in condition_cameras:
        if cam == "camera_2":
            inputs[cam] = tactile_inp
        else:
            cam_dim = CAMERA_EMB_DIMS.get(cam, 768)
            hist = (vision_emb_histories or {}).get(cam, deque())
            if hist:
                fallback = list(hist)[-1].detach()
                if _is_k_step_wm(wm_model):
                    num_hist = int(getattr(wm_model, "num_hist"))
                    fs = int(getattr(wm_model, "frameskip"))
                    ctx = build_tactile_k_step_context(
                        hist, num_hist, fs, temporal_mode, device,
                        tactile_emb_fallback=fallback,
                    )
                else:
                    ctx = build_tactile_wm_context(
                        hist, T, device, tactile_emb_fallback=fallback
                    )
                inputs[cam] = ctx.expand(N, -1, -1, -1).contiguous()
            else:
                inputs[cam] = torch.zeros(N, T, P, cam_dim, device=device, dtype=torch.float32)
    return inputs


def _wm_context_length(wm_model):
    """Temporal context length H (video: num_frames; k_step: num_hist)."""
    if _is_k_step_wm(wm_model):
        return int(getattr(wm_model, "num_hist"))
    return int(wm_model.temp_embedding.shape[1])


def _tactile_emb_history_maxlen(wm_model, temporal_mode):
    """Deque capacity for tactile embeddings (consecutive vs LPB span)."""
    if wm_model is None:
        return None
    if _is_k_step_wm(wm_model):
        nh = int(getattr(wm_model, "num_hist"))
        fs = int(getattr(wm_model, "frameskip"))
        if temporal_mode == "lpb":
            return max(nh, (nh - 1) * fs + 1)
        return nh
    return _wm_context_length(wm_model)


def k_step_hist_frame_indices(
    t_last: int, num_hist: int, frameskip: int, temporal_mode: str
) -> list[int]:
    """
    Raw frame indices for k-step history (aligned with eval_dino_wm_tactile_k_step).
    t_last: index of the most recent frame in the episode log (inclusive).
    """
    num_hist = int(num_hist)
    fs = int(frameskip)
    if temporal_mode == "consecutive":
        idxs = [t_last - num_hist + 1 + j for j in range(num_hist)]
    else:
        idxs = [t_last - (num_hist - 1 - j) * fs for j in range(num_hist)]
    return [max(0, min(i, t_last)) for i in idxs]


def build_tactile_k_step_context(
    tactile_emb_history,
    num_hist,
    frameskip,
    temporal_mode,
    device,
    tactile_emb_fallback=None,
):
    """
    Stack tactile patch embeddings for k-step WM (eval_dino_wm_tactile_k_step.py).

    temporal_mode \"lpb\": hist_idx[j] = t_last - (num_hist-1-j)*frameskip (clamped).
    temporal_mode \"consecutive\": hist_idx[j] = t_last - num_hist + 1 + j (clamped).
    """
    hist = list(tactile_emb_history)
    if len(hist) == 0:
        if tactile_emb_fallback is None:
            raise ValueError("tactile_emb_history is empty and no tactile_emb_fallback")
        e = tactile_emb_fallback.detach().to(device)
        if e.dim() == 2:
            e = e.unsqueeze(0)
        inp = e.expand(1, num_hist, -1, -1).contiguous()
    else:
        t_last = len(hist) - 1
        idxs = k_step_hist_frame_indices(t_last, num_hist, frameskip, temporal_mode)
        emb_list = []
        for i in idxs:
            t = hist[i]
            emb_list.append(t.squeeze(0) if t.dim() == 3 else t)
        seq = torch.stack(emb_list, dim=0)
        inp = seq.unsqueeze(0).contiguous().to(device)

    n_patches = inp.shape[2]
    side = int(n_patches ** 0.5)
    if side * side != n_patches or side != PATCH_SIDE_DINOV3:
        inp = ensure_patch_grid(inp, PATCH_SIDE_DINOV3, device)
    return inp


def build_k_step_state_tensor(
    state_history,
    num_hist,
    frameskip,
    temporal_mode,
    state_fallback,
    device,
    batch_size,
):
    """
    Build (N, num_hist, 8) proprio states using the same index layout as
    ``build_tactile_k_step_context`` (eval_dino_wm_tactile_k_step: ``states[:, hist_idx, :]``).

    state_history: deque of (8,) float32 vectors, oldest first — one entry per timestep we
    appended a tactile embedding (aligned with tactile_emb_history).
    state_fallback: (8,) current state when history is empty (mirrors tactile_emb_fallback).
    """
    num_hist = int(num_hist)
    dev = device if isinstance(device, torch.device) else torch.device(device)
    N = int(batch_size)
    sf = np.asarray(state_fallback, dtype=np.float32).reshape(8)

    if state_history is None or len(state_history) == 0:
        t = torch.from_numpy(sf).to(dev).view(1, 1, 8).expand(N, num_hist, 8).contiguous()
        return t

    hist = list(state_history)
    t_last = len(hist) - 1
    idxs = k_step_hist_frame_indices(t_last, num_hist, frameskip, temporal_mode)
    rows = [np.asarray(hist[i], dtype=np.float32).reshape(8) for i in idxs]
    stacked = np.stack(rows, axis=0)
    t_one = torch.from_numpy(np.ascontiguousarray(stacked)).to(dev).unsqueeze(0).float()
    return t_one.expand(N, -1, -1).contiguous()


def build_k_step_inp_and_st_from_logged(
    tactile_encoder,
    sensor_id,
    device,
    cam2_list,
    proprio_list,
    num_hist,
    frameskip,
    temporal_mode,
    tactile_emb_fallback,
    state_fallback,
    use_amp,
):
    """
    Build (1, num_hist, P, D) tactile and (1, num_hist, 8) state from **full episode logs**
    ``camera_2`` and ``proprio`` (same length, one row per control tick).

    For each history slot, tactile uses a 4-frame window ending at the chosen index (same as
    ``get_tactile_embedding_from_image``), so LPB indices far in the past still get correct
    AnyTouch windows from stored images — not limited by deque maxlen.
    Background subtraction uses ``cam2_list[0]`` as the episode reference (same as live run).
    """
    num_hist = int(num_hist)
    fs = int(frameskip)
    dev = device if isinstance(device, torch.device) else torch.device(device)

    if not cam2_list or len(cam2_list) == 0:
        if tactile_emb_fallback is None:
            raise ValueError("logged camera_2 empty and no tactile_emb_fallback")
        e = tactile_emb_fallback.detach().to(device)
        if e.dim() == 2:
            e = e.unsqueeze(0)
        inp = e.expand(1, num_hist, -1, -1).contiguous()
        sf = np.asarray(state_fallback, dtype=np.float32).reshape(8)
        st = torch.from_numpy(sf).to(dev).view(1, 1, 8).expand(1, num_hist, 8).contiguous()
        return inp, st

    if len(proprio_list) != len(cam2_list):
        raise ValueError(
            f"proprio length {len(proprio_list)} != camera_2 length {len(cam2_list)}"
        )

    t_last = len(cam2_list) - 1
    idxs = k_step_hist_frame_indices(t_last, num_hist, fs, temporal_mode)

    emb_slots = []
    for idx in idxs:
        start = max(0, idx - 3)
        tactile_frames = [cam2_list[i] for i in range(start, idx + 1)]
        emb = get_tactile_embedding_from_image(
            tactile_encoder,
            tactile_frames,
            sensor_id,
            device,
            use_amp=use_amp,
            bg_image=cam2_list[0],
        )
        emb_slots.append(emb.squeeze(0) if emb.dim() == 3 else emb)

    inp = torch.stack(emb_slots, dim=0).unsqueeze(0).contiguous().to(dev)
    n_patches = inp.shape[2]
    side = int(n_patches**0.5)
    if side * side != n_patches or side != PATCH_SIDE_DINOV3:
        inp = ensure_patch_grid(inp, PATCH_SIDE_DINOV3, device)

    rows = [np.asarray(proprio_list[i], dtype=np.float32).reshape(8) for i in idxs]
    st = torch.from_numpy(np.stack(rows, axis=0)).to(dev).unsqueeze(0).float()
    return inp, st


def _normalize_collapsed_actions_tensor(acs, norm_stats):
    """acs: (..., frameskip*7) -> normalized same shape."""
    shape = acs.shape
    f7 = shape[-1]
    flat = acs.reshape(-1, 7)
    flat = norm_stats.normalize_acs(flat)
    return flat.view(shape)


def _k_step_collapsed_actions_batch(
    action_chunks, rollout_steps, num_hist, frameskip, temporal_mode, chunk_T, device
):
    """
    (N, chunk_T, 7) -> (rollout_steps, N, num_hist, frameskip*7).
    Aligns with tactile_k_step_dataset consecutive vs lpb indexing at anchor = rollout step.
    """
    N = action_chunks.shape[0]
    fs = int(frameskip)
    out = torch.zeros(
        (rollout_steps, N, num_hist, fs * 7),
        device=device,
        dtype=torch.float32,
    )
    for step in range(rollout_steps):
        for j in range(num_hist):
            if temporal_mode == "consecutive":
                idx_start = step - num_hist + 1 + j
            else:
                idx_start = step - (num_hist - 1 - j) * fs
            idx_start = max(0, min(idx_start, chunk_T - 1))
            for k in range(fs):
                idx = min(idx_start + k, chunk_T - 1)
                out[step, :, j, k * 7 : (k + 1) * 7] = action_chunks[:, idx, :]
    return out


def _action_window_from_chunk(action_chunks, rollout_step, h_ctx, device):
    """(N, T, 7) -> (N, h_ctx, 7) with indices rollout_step .. rollout_step+h_ctx-1 clamped to T-1."""
    N, t_hor, d_ac = action_chunks.shape
    t_hor = int(t_hor)
    idxs = [min(rollout_step + j, t_hor - 1) for j in range(h_ctx)]
    parts = [action_chunks[:, idx : idx + 1, :] for idx in idxs]
    return torch.cat(parts, dim=1).to(device=device, dtype=torch.float32)


def _action_windows_batch(action_chunks, rollout_steps, h_ctx, device):
    """Precompute all action windows for all rollout steps. (N, T, 7) -> (rollout_steps, N, h_ctx, 7)."""
    parts = [
        _action_window_from_chunk(action_chunks, step, h_ctx, device)
        for step in range(rollout_steps)
    ]
    return torch.stack(parts, dim=0)


def build_tactile_wm_context(tactile_emb_history, h_ctx, device, tactile_emb_fallback=None):
    """
    Stack GT tactile patch embeddings for world-model context (like eval GT history).

    tactile_emb_history: deque or sequence of (1, N_patches, D), oldest first (from encoder).
    If len < h_ctx: left-pad by repeating the **first** (oldest) embedding.
    If empty: use tactile_emb_fallback (1, P, D) repeated h_ctx times.

    Returns: (1, h_ctx, N_patches, D) on device
    """
    hist = list(tactile_emb_history)
    if len(hist) == 0:
        if tactile_emb_fallback is None:
            raise ValueError("tactile_emb_history is empty and no tactile_emb_fallback")
        e = tactile_emb_fallback.detach().to(device)
        if e.dim() == 2:
            e = e.unsqueeze(0)
        inp = e.expand(1, h_ctx, -1, -1).contiguous()
    else:
        seq = torch.cat(hist, dim=0)
        L, n_patches, D = seq.shape
        if L >= h_ctx:
            seq = seq[-h_ctx:]
        else:
            pad = seq[0:1].expand(h_ctx - L, n_patches, D)
            seq = torch.cat([pad, seq], dim=0)
        inp = seq.unsqueeze(0).contiguous().to(device)

    n_patches = inp.shape[2]
    side = int(n_patches ** 0.5)
    if side * side != n_patches or side != PATCH_SIDE_DINOV3:
        inp = ensure_patch_grid(inp, PATCH_SIDE_DINOV3, device)
    return inp


def _predict_tactile_rollout_k_step(
    wm_model,
    tactile_emb_history,
    state,
    action_chunks,
    norm_stats,
    text_embeds,
    device,
    temporal_mode,
    tactile_emb_fallback,
    use_amp,
    state_history=None,
    logged_cam2=None,
    logged_proprio=None,
    tactile_encoder=None,
    sensor_id=0,
    vision_emb_histories=None,
    text_embeds_neg=None,
):
    """
    K-step WM: one forward per query.

    Uses tactile history + collapsed actions from the **current** action chunk only
    (anchor at chunk step 0). Predicts ``num_pred`` future tactile embeddings (typically 1).
    No autoregressive unrolling — matches ``num_pred``-step training targets.

    Proprio ``state`` tensor is (N, num_hist, 8) from ``state_history`` aligned with tactile
    indices (same as eval); if ``state_history`` is empty, repeats ``state`` (fallback).

    When ``num_hist > 1`` and ``logged_cam2`` / ``logged_proprio`` are provided (full episode
    lists aligned with ``policy_query_frame_idx``), tactile and state are built from those
    logs so LPB indices can reach far enough back without deque maxlen loss.
    """
    num_hist = int(wm_model.num_hist)
    fs = int(wm_model.frameskip)
    N = action_chunks.shape[0]
    chunk_T = int(action_chunks.shape[1])

    use_logged = (
        num_hist > 1
        and logged_cam2 is not None
        and logged_proprio is not None
        and len(logged_cam2) >= 1
        and tactile_encoder is not None
    )
    if use_logged:
        if len(logged_proprio) != len(logged_cam2):
            raise ValueError(
                f"logged proprio length {len(logged_proprio)} != camera_2 length {len(logged_cam2)}"
            )
        inp, st_one = build_k_step_inp_and_st_from_logged(
            tactile_encoder,
            sensor_id,
            device,
            logged_cam2,
            logged_proprio,
            num_hist,
            fs,
            temporal_mode,
            tactile_emb_fallback,
            state,
            use_amp,
        )
        tactile_expanded = inp.expand(N, -1, -1, -1).contiguous()
        inputs = _build_wm_inputs(
            tactile_expanded, wm_model, N, device,
            vision_emb_histories=vision_emb_histories, temporal_mode=temporal_mode,
        )
        st = st_one.expand(N, -1, -1).contiguous()
    else:
        inp = build_tactile_k_step_context(
            tactile_emb_history,
            num_hist,
            fs,
            temporal_mode,
            device,
            tactile_emb_fallback=tactile_emb_fallback,
        )
        tactile_expanded = inp.expand(N, -1, -1, -1).contiguous()
        inputs = _build_wm_inputs(
            tactile_expanded, wm_model, N, device,
            vision_emb_histories=vision_emb_histories, temporal_mode=temporal_mode,
        )
        st = build_k_step_state_tensor(
            state_history,
            num_hist,
            fs,
            temporal_mode,
            state,
            device,
            N,
        )

    start_time = time.time()
    acs_raw_all = _k_step_collapsed_actions_batch(
        action_chunks, 1, num_hist, fs, temporal_mode, chunk_T, device
    )
    acs = _normalize_collapsed_actions_tensor(acs_raw_all[0], norm_stats)
    action_norm_time = time.time() - start_time
    # print(f"Action normalization time: {action_norm_time:.3f}s")

    start_time = time.time()
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds, _pred_state = wm_model(inputs, st, acs)
        pred_e = preds["camera_2"].clone()
    forward_time = time.time() - start_time
    # print(f"WM forward time: {forward_time:.3f}s (k_step: single pass)")
    # (N, num_pred, P, D) — reward over predicted horizon
    pred_embeds_seq = pred_e
    reward_time = time.time()
    rewards_seq = compute_rewards_from_embeds_batch(
        pred_embeds_seq, text_embeds, device, text_embeds_neg=text_embeds_neg
    )
    reward_time = time.time() - reward_time
    # print(f"Reward computation time: {reward_time:.3f}s")
    total_time = time.time() - start_time
    n_out = int(pred_embeds_seq.shape[1])
    # print(f"WM k_step total time: {total_time:.3f}s (num_pred={n_out} future embedding(s))")
    return pred_embeds_seq, rewards_seq


def predict_tactile_rollout(
    wm_model,
    tactile_emb_history,
    state,
    action_chunks,
    norm_stats,
    text_embeds,
    device="cuda",
    rollout_steps=None,
    tactile_emb_fallback=None,
    use_amp=False,
    temporal_mode="lpb",
    state_history=None,
    logged_cam2=None,
    logged_proprio=None,
    tactile_encoder=None,
    sensor_id=0,
    vision_emb_histories=None,
    text_embeds_neg=None,
):
    """
    Autoregressive tactile embedding rollout for each of N action sequences.
    Optimized: action normalization and reward computation batched outside loop.

    tactile_emb_history: deque of (1, N_patches, 512) GT embeddings from encoder, oldest first
        (maxlen=wm_h_ctx). build_tactile_wm_context pads left with oldest when len < h_ctx.
    tactile_emb_fallback: if history is empty, repeat this embedding (only first control step).
    state: (8,) current proprio state (fallback when k-step state_history is empty).
    state_history: k-step only — deque of (8,) states aligned with tactile_emb_history (same
        length/append cadence); used to build (N, num_hist, 8) like eval, not broadcast current.
    logged_cam2 / logged_proprio: k-step, num_hist>1 — full lists per timestep (same length);
        used instead of deque embeddings when set (see ``build_k_step_inp_and_st_from_logged``).
    action_chunks: (N, ACT_HORIZON, 7) torch or numpy per-sample action chunks
    temporal_mode: k_step only — \"lpb\" or \"consecutive\" (ignored for video WM).
    k_step: one forward per query; pred_embeds_seq is (N, num_pred, N_patches, 512); ``--wm_rollout_steps`` is ignored.
    video: autoregressive unrolling over ``rollout_steps``.
    Returns:
        pred_embeds_seq: (N, T, N_patches, 512) — T = num_pred (k_step) or rollout_steps (video)
        rewards_seq: (N, T) numpy, cosine reward per step
    """
    if not torch.is_tensor(action_chunks):
        action_chunks = torch.tensor(
            np.asarray(action_chunks), dtype=torch.float32, device=device
        )
    chunk_T = int(action_chunks.shape[1])

    if _is_k_step_wm(wm_model):
        if temporal_mode not in ("lpb", "consecutive"):
            raise ValueError('temporal_mode must be "lpb" or "consecutive"')
        if rollout_steps is not None:
            logger.info(
                "k_step WM: --wm_rollout_steps=%s ignored (single forward; num_pred=%s).",
                rollout_steps,
                int(getattr(wm_model, "num_pred")),
            )
        return _predict_tactile_rollout_k_step(
            wm_model,
            tactile_emb_history,
            state,
            action_chunks,
            norm_stats,
            text_embeds,
            device,
            temporal_mode,
            tactile_emb_fallback,
            use_amp,
            state_history=state_history,
            logged_cam2=logged_cam2,
            logged_proprio=logged_proprio,
            tactile_encoder=tactile_encoder,
            sensor_id=sensor_id,
            vision_emb_histories=vision_emb_histories,
            text_embeds_neg=text_embeds_neg,
        )

    if rollout_steps is None:
        rollout_steps = chunk_T
    else:
        rollout_steps = int(rollout_steps)
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
    if rollout_steps > chunk_T:
        logger.warning(
            "rollout_steps=%s > action chunk length %s; clamping to chunk_T",
            rollout_steps,
            chunk_T,
        )
        rollout_steps = chunk_T

    h_ctx = _wm_context_length(wm_model)
    N = action_chunks.shape[0]
    inp = build_tactile_wm_context(
        tactile_emb_history, h_ctx, device, tactile_emb_fallback=tactile_emb_fallback
    )
    tactile_expanded = inp.expand(N, -1, -1, -1).contiguous()
    inputs = _build_wm_inputs(
        tactile_expanded, wm_model, N, device,
        vision_emb_histories=vision_emb_histories, temporal_mode=temporal_mode,
    )
    st = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, 8).expand(N, h_ctx, 8).contiguous()

    # Pre-batch action normalization: (rollout_steps, N, h_ctx, 7)
    start_time = time.time()
    acs_raw_all = _action_windows_batch(action_chunks, rollout_steps, h_ctx, device)
    acs_normed_all = norm_stats.normalize_acs(acs_raw_all)
    action_norm_time = time.time() - start_time
    # print(f"Action normalization time: {action_norm_time:.3f}s")
    # pred_steps = []
    pred_embeds_seq = torch.zeros((N, rollout_steps, PATCH_SIDE_DINOV3 * PATCH_SIDE_DINOV3, 512), device=device)
    start_time = time.time()
    with torch.inference_mode():
        for step in range(rollout_steps):
            acs = acs_normed_all[step]
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds, pred_state = wm_model(inputs, st, acs)
            next_e = preds["camera_2"][:, -1:]
            # CUDAGraphs from torch.compile reuse output buffers; clone before retaining per step.
            # pred_steps.append(next_e.squeeze(1).clone())
            pred_embeds_seq[:, step, ] = next_e.squeeze(1)
            # Roll the tactile context window; vision cameras stay at their snapshot.
            new_tactile = torch.cat([inputs["camera_2"][:, 1:], next_e], dim=1)
            inputs = _build_wm_inputs(
                new_tactile, wm_model, N, device,
                vision_emb_histories=vision_emb_histories, temporal_mode=temporal_mode,
            )
            next_s = pred_state[:, -1:]
            st = torch.cat([st[:, 1:], next_s], dim=1)
    forward_time = time.time() - start_time
    # print(f"WM forward time: {forward_time:.3f}s")
    # pred_embeds_seq = torch.stack(pred_steps, dim=1)
    reward_time = time.time()
    # Batch reward computation: (N, rollout_steps, N_patches, 512) -> (N, rollout_steps)
    rewards_seq = compute_rewards_from_embeds_batch(
        pred_embeds_seq, text_embeds, device, text_embeds_neg=text_embeds_neg
    )
    reward_time = time.time() - reward_time
    # print(f"Reward computation time: {reward_time:.3f}s")
    total_time = time.time() - start_time
    # print(f"WM rollout total time: {total_time:.3f}s ({total_time / rollout_steps:.3f}s/step)")
    return pred_embeds_seq, rewards_seq



def plot_reward_vs_actions(action_chunks, rewards_seq, out_path, action_dim_labels=None):
    """
    Three subplots for action dims 0, 1, 2; each has N curves.
    Curve n: x = action_chunks[n, :, dim], y = rewards_seq[n, :] (length ACT_HORIZON).
    """
    ac = np.asarray(action_chunks)
    rewards_seq = np.asarray(rewards_seq)
    N, T, d = ac.shape
    if action_dim_labels is None:
        action_dim_labels = [f"action[{i}]" for i in range(3)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        dim = min(i, d - 1)
        lbl = action_dim_labels[dim] if dim < len(action_dim_labels) else f"dim {dim}"
        for n in range(N):
            ax.plot(
                ac[n, :, dim],
                rewards_seq[n, :],
                marker="o",
                markersize=3,
                alpha=0.85,
                label=f"n={n}" if N <= 12 else None,
            )
        ax.set_xlabel(lbl)
        ax.set_ylabel("reward")
        ax.set_title(f"dim {dim} vs reward")
        ax.grid(True, alpha=0.3)
    if N <= 12:
        axes[0].legend(fontsize=6, loc="best")
    fig.suptitle("Predicted tactile-text reward vs action (per rollout step)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_rewards_from_embeds(pred_embeds, text_embeds, device='cuda'):
    """
    pred_embeds: (N, N_patches, 512)
    text_embeds: (1, 512) or (num_texts, 512)
    Returns: (N,) reward per action sample (cosine similarity to first text)
    """
    emb = pred_embeds.mean(dim=1)
    emb = F.normalize(emb, dim=-1)
    if text_embeds.dim() == 1:
        text_embeds = text_embeds.unsqueeze(0)
    sim = emb @ text_embeds.T
    return sim[:, 0].cpu().numpy()


def compute_rewards_from_embeds_batch(pred_embeds_seq, text_embeds, device='cuda', text_embeds_neg=None):
    """
    pred_embeds_seq: (N, rollout_steps, N_patches, 512)
    text_embeds: (1, 512) or (num_texts, 512)
    text_embeds_neg: optional (1, 512) negative text; reward = sim_pos - sim_neg (contrastive)
    Returns: (N, rollout_steps) numpy, cosine reward per sample per step
    """
    N, T, n_patches, D = pred_embeds_seq.shape
    emb = pred_embeds_seq.mean(dim=2)
    emb = F.normalize(emb, dim=-1)
    if text_embeds.dim() == 1:
        text_embeds = text_embeds.unsqueeze(0)
    sim = emb @ text_embeds.T  # (N, T, 1)
    if text_embeds_neg is not None:
        if text_embeds_neg.dim() == 1:
            text_embeds_neg = text_embeds_neg.unsqueeze(0)
        sim = sim - emb @ text_embeds_neg.T
    return sim[:, :, 0].cpu().numpy()


def select_action_chunk_by_reward(action_chunks, rewards, mode="argmax", temperature=1.0):
    """
    action_chunks: (N, ACT_HORIZON, 7), rewards: (N,) scalar score per sample (e.g. mean over horizon)
    mode: 'argmax' or 'softmax'
    Returns: list of ACT_HORIZON actions (7,) each, for the queue
    """
    if mode == 'argmax':
        idx = np.argmax(rewards)
    elif mode == 'argmin':
        idx = np.argmin(rewards)
    else:
        rewards = np.array(rewards, dtype=np.float64)
        rewards = rewards - rewards.max()
        probs = np.exp(rewards / temperature)
        probs = probs / probs.sum()
        idx = np.random.choice(len(action_chunks), p=probs)
    return [action_chunks[idx][t].copy() for t in range(action_chunks.shape[1])]


def eef_pose_to_state(T, gripper):
    # Extract translation
    x, y, z = T[:3, 3]

    # Extract rotation matrix and convert to quaternion
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # Returns [qx, qy, qz, qw]

    # Concatenate into final state
    eef_state = np.concatenate(([x, y, z], quat, [gripper]))
    return eef_state

def main():
    args = parse_args()

    global logger
    from easydict import EasyDict
    from deoxys import config_root
    from deoxys.experimental.motion_utils import reset_joints_to
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils import YamlConfig
    from deoxys.utils.log_utils import get_deoxys_example_logger
    from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface
    logger = get_deoxys_example_logger()

    # Setting up robot, teleop, and camera interfaces
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg), gripper_mode='continuous_smooth')
    policy = RealPolicy(['state'], args.ckpt_dir, args.ckpt_iter)
    camera_ids = [0, 1, 2]
    camera_types = ['rs', 'zed', 'tactile']
    camera_info = EasyDict({
        "camera_id": camera_ids,
        "camera_type": camera_types,
        "camera_name": [f"camera_{camera_types[id]}_{camera_ids[id]}" for id,_ in enumerate(camera_ids)]
    })
    cr_interfaces = {}
    for camera_id in camera_ids:
        print(camera_id, 'id')
        if 'tactile' in camera_types[camera_id]:
            cr_interface = CameraRedisSubInterface(camera_info=camera_info, camera_id=camera_id, use_color=False, use_tactile=True)
        else:
            cr_interface = CameraRedisSubInterface(camera_info=camera_info, camera_id=camera_id)

        cr_interface.start()
        cr_interfaces[camera_id] = cr_interface

    # demo_file = h5py.File(demo_file_name, "w")
    controller_type = args.controller_type
    print(controller_type)
    controller_cfg = YamlConfig(
        os.path.join(config_root, args.controller_cfg)
    ).as_easydict()

    args.folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {args.folder}")

    wm_model = None
    tactile_encoder = None
    sensor_id = 0
    clip_model = None
    tokenizer = None
    text_embeds = None
    text_embeds_phase2 = None
    norm_stats = None
    tactile_decoder = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.visualize_pred:
        decoder_dir = args.decoder_dir
        if not decoder_dir and args.wm_checkpoint:
            decoder_dir = os.path.dirname(os.path.abspath(args.wm_checkpoint))
        if decoder_dir:
            tactile_decoder = load_tactile_decoder(decoder_dir, device)
            if tactile_decoder:
                logger.info(f"Loaded tactile decoder for visualization from {decoder_dir}")
            else:
                logger.warning(f"No decoder_camera_2.pth found in {decoder_dir}")
    if args.use_tactile and args.wm_checkpoint and args.anytouch_checkpoint:
        logger.info("Loading dynamics model and tactile encoder for reward-based action selection...")
        nh = args.num_hist if args.num_hist is not None else 1
        npred = args.num_pred if args.num_pred is not None else 1
        fs = args.frameskip if args.frameskip is not None else 8
        wm_cameras = [c.strip() for c in args.wm_cameras.split(",")]
        wm_cond_cams = (
            [c.strip() for c in args.wm_condition_cameras.split(",")]
            if args.wm_condition_cameras else wm_cameras
        )
        wm_pred_cams = (
            [c.strip() for c in args.wm_predict_cameras.split(",")]
            if args.wm_predict_cameras else wm_cameras
        )
        wm_model = load_world_model(
            args.wm_checkpoint,
            device,
            use_compile=not args.no_compile,
            wm_arch=args.wm_arch,
            num_hist=nh,
            num_pred=npred,
            frameskip=fs,
            cameras=wm_cameras,
            condition_cameras=wm_cond_cams,
            predict_cameras=wm_pred_cams,
        )
        logger.info(
            "WM: arch=%s temporal_mode=%s (k_step hyperparams: num_hist=%s num_pred=%s frameskip=%s) "
            "cameras=%s condition=%s predict=%s (reward from camera_2 head only)",
            args.wm_arch,
            args.temporal_mode,
            nh,
            npred,
            fs,
            wm_cameras,
            wm_cond_cams,
            wm_pred_cams,
        )
        tactile_encoder, sensor_id = load_tactile_encoder(args.anytouch_checkpoint, device=str(device))
        clip_model, tokenizer = load_clip_for_reward(args.anytouch_checkpoint, device)
        norm_stats_path = args.norm_stats_json
        if norm_stats_path is None and args.wm_checkpoint:
            wm_dir = os.path.dirname(os.path.abspath(args.wm_checkpoint))
            default_path = os.path.join(wm_dir, "norm_stats.json")
            if os.path.isfile(default_path):
                norm_stats_path = default_path
        print('norm_stats_path', norm_stats_path)
        norm_stats = NormStats(norm_stats_path, device=str(device))
        inputs = tokenizer([args.reward_text], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeds = clip_model.get_text_features(**inputs)
            text_embeds = F.normalize(text_embeds, dim=-1)
        logger.info(f"Reward text: '{args.reward_text}', action_select: {args.action_select}")

        # ── Phase-2 text embedding ────────────────────────────────────────────
        text_embeds_phase2 = None
        if args.separate_phase and args.reward_text_phase2:
            inputs2 = tokenizer(
                [args.reward_text_phase2], padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                text_embeds_phase2 = clip_model.get_text_features(**inputs2)
                text_embeds_phase2 = F.normalize(text_embeds_phase2, dim=-1)
            logger.info(
                "Phase-2 reward text: '%s'  robometer threshold=%.3f  server=%s",
                args.reward_text_phase2, args.phase_threshold, args.eval_server_url,
            )

        # Warmup for torch.compile (first run compiles, subsequent runs are fast)
        # if not args.no_compile and hasattr(torch, "compile"):
        #     with torch.inference_mode():
        #         for _ in range(10):
        #             n_patches = PATCH_SIDE_DINOV3 * PATCH_SIDE_DINOV3
        #             if _is_k_step_wm(wm_model):
        #                 nh = int(getattr(wm_model, "num_hist"))
        #                 fs = int(getattr(wm_model, "frameskip"))
        #                 dummy_inp = torch.randn(1, nh, n_patches, 512, device=device)
        #                 dummy_st = torch.randn(1, nh, 8, device=device)
        #                 dummy_ac = torch.randn(1, nh, fs * 7, device=device)
        #                 dummy_inputs = {"camera_2": dummy_inp}
        #                 with torch.amp.autocast("cuda", enabled=args.amp):
        #                     _ = wm_model(dummy_inputs, dummy_st, dummy_ac)
        #             else:
        #                 h_ctx = _wm_context_length(wm_model)
        #                 dummy_inp = torch.randn(1, h_ctx, n_patches, 512, device=device)
        #                 dummy_st = torch.randn(1, h_ctx, 8, device=device)
        #                 dummy_ac = torch.randn(1, h_ctx, 7, device=device)
        #                 dummy_inputs = {"camera_2": dummy_inp}
        #                 with torch.amp.autocast("cuda", enabled=args.amp):
        #                     _ = wm_model(dummy_inputs, dummy_st, dummy_ac)
        #     logger.info("World model compile warmup done")
    wm_h_ctx = _tactile_emb_history_maxlen(wm_model, args.temporal_mode) if wm_model is not None else None

    # ── DINOv3 for RGB cameras (camera_0, camera_1) ───────────────────────────
    wm_cameras_list = [c.strip() for c in args.wm_cameras.split(",")] if wm_model is not None else []
    rgb_cams = [c for c in wm_cameras_list if c in ("camera_0", "camera_1")]
    vision_encoder = None
    if rgb_cams:
        vision_encoder = load_vision_encoder(args.dinov3_hub_dir, args.dinov3_weights, device)
        logger.info("DINOv3 encoder loaded for %s", rgb_cams)

    init_joints = [-0.14091554, 0.02724072, 0.33824686, -2.48239114, -0.12527768, 2.53220921, -2.87378524]
    episode_id = 0
    while True:
        data = {
            "actions": [],
            "ee_states": [],
            "joint_states": [],
            "gripper_states": [],
            "proprio": [],
            "pred_embeds_log": [],
            "rewards_log": [],
            "action_chunks_log": [],
            "decision_ee_xyz": [],
            "policy_query_frame_idx": [],
            "phase_log": [],
        }
        for camera_id in camera_ids:
            data[f"camera_{camera_id}"] = []

        # ── Two-phase state ───────────────────────────────────────────────────
        current_phase = 1
        current_task = args.reward_text
        text_embeds_active = text_embeds
        text_embeds_neg_active = (
            text_embeds_phase2 if (args.mixed_reward and text_embeds_phase2 is not None) else None
        )
        robometer_session_id = None

        i = 0
        action_chunk_queue = []
        # reward_sequence = []
        tactile_buffer = deque(maxlen=4)
        tactile_emb_history = deque(maxlen=wm_h_ctx) if wm_h_ctx is not None else None
        state_history = (
            deque(maxlen=wm_h_ctx)
            if (wm_h_ctx is not None and wm_model is not None and _is_k_step_wm(wm_model))
            else None
        )
        vision_emb_histories = (
            {cam: deque(maxlen=wm_h_ctx) for cam in rgb_cams}
            if rgb_cams else None
        )
        episode_tactile_bg = None
        pred_vis_dir = None
        if args.visualize_pred and tactile_decoder is not None:
            pred_vis_dir = args.folder / "pred_vis"
            pred_vis_dir.mkdir(parents=True, exist_ok=True)
        reward_plot_dir = None
        if args.plot_reward:
            reward_plot_dir = args.folder / "reward_plots"
            reward_plot_dir.mkdir(parents=True, exist_ok=True)

        reset_joints_to(robot_interface, init_joints)
        time.sleep(1)
        print('Beginning Trajectory')
        interrupted = False
        warmup_step = 0
        try:
            while True:
                # start_time = time.time()
                last_state = robot_interface._state_buffer[-1]
                last_gripper_state = robot_interface._gripper_state_buffer[-1]

                obs_dict = {}
                obs_dict['state'] = eef_pose_to_state(np.array(last_state.O_T_EE).reshape(4,4).T, np.array(last_gripper_state.width))
                ## save the state, joint, gripper, action to the data dictionary
                data["proprio"].append(np.asarray(obs_dict["state"], dtype=np.float32).copy())

                data["ee_states"].append(np.array(last_state.O_T_EE))
                data["joint_states"].append(np.array(last_state.q))
                data["gripper_states"].append(np.array(last_gripper_state.width))
                for camera_id in camera_ids:
                    img = cr_interfaces[camera_id].get_img()
                    if camera_id == 0:
                        img  = cv2.resize(img['color'], (256, 256), interpolation=cv2.INTER_LINEAR)
                    if camera_id == 1:
                        img = img['color']
                        img  = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    if camera_id == 2:
                        img = img['tactile']
                    data[f"camera_{camera_id}"].append(img)

                    if not args.use_tactile and camera_id == 2:
                        continue
                    obs_dict[f"camera_{camera_id}"] = img
                    if args.use_tactile and camera_id == 2:
                        tactile_buffer.append(img.copy())
                if args.use_tactile and episode_tactile_bg is None and obs_dict.get("camera_2") is not None:
                    episode_tactile_bg = np.asarray(obs_dict["camera_2"]).copy()
                # time_end = time.time()
                # print(f"States and tactile buffer time: {time_end - start_time:.3f}s")

                current_tactile_emb = None
                time_start = time.time()
                if (
                    tactile_emb_history is not None
                    and tactile_encoder is not None
                    and tactile_buffer
                ):
                    current_tactile_emb = get_tactile_embedding_from_image(
                        tactile_encoder,
                        tactile_buffer,
                        sensor_id,
                        device,
                        use_amp=args.amp,
                        bg_image=episode_tactile_bg,
                    )
                    tactile_emb_history.append(current_tactile_emb.detach())
                    if state_history is not None:
                        state_history.append(
                            np.asarray(obs_dict["state"], dtype=np.float32).copy()
                        )

                # ── Encode RGB camera frames (camera_0, camera_1) with DINOv3 ──
                if vision_encoder is not None and vision_emb_histories is not None:
                    for cam_name in rgb_cams:
                        cid_int = int(cam_name.split("_")[1])
                        if cam_name in obs_dict:
                            v_emb = get_vision_embedding(
                                vision_encoder, obs_dict[cam_name], cid_int, device
                            )
                            vision_emb_histories[cam_name].append(v_emb.detach())

                time_end = time.time()

                # print(f"Tactile embedding time: {time_end - time_start:.3f}s")
                if not action_chunk_queue:
                    time_start = time.time()

                    # ── Robometer phase switching on real observations ─────────
                    if args.separate_phase and warmup_step > 3:
                        real_cam0 = obs_dict.get("camera_1", obs_dict.get("camera_0"))
                        if robometer_session_id is None:
                            try:
                                robometer_session_id = stream_create(
                                    args.eval_server_url,
                                    "transfer the liquid to the blue cup or the yellow cup",
                                    timeout_s=args.robometer_timeout_s,
                                )
                                logger.info(
                                    "Robometer session created (phase %d, task='%s'): %s",
                                    current_phase, current_task, robometer_session_id,
                                )
                            except Exception as _e:
                                logger.warning("Could not create robometer session: %s", _e)
                        if robometer_session_id is not None and real_cam0 is not None:
                            try:
                                hist_reward, _ = stream_append_frame(
                                    args.eval_server_url,
                                    robometer_session_id,
                                    real_cam0,
                                    timeout_s=args.robometer_timeout_s,
                                )
                                logger.info(
                                    "Robometer real-obs reward (phase %d): %.4f  threshold=%.4f",
                                    current_phase, hist_reward, args.phase_threshold,
                                )
                                data["phase_log"].append(current_phase)
                                if (current_phase == 1
                                        and text_embeds_phase2 is not None
                                        and hist_reward > args.phase_threshold):
                                    input("Press Enter to continue...")
                                    logger.info(
                                        "Phase 1→2 switch: robometer reward %.4f > %.4f — "
                                        "switching guidance text to: '%s'",
                                        hist_reward, args.phase_threshold, args.reward_text_phase2,
                                    )
                                    input("Press Enter to continue...")
                                    try:
                                        stream_delete(
                                            args.eval_server_url, robometer_session_id,
                                            timeout_s=args.robometer_timeout_s,
                                        )
                                        logger.info("Phase-1 session deleted: %s", robometer_session_id)
                                    except Exception as _e:
                                        logger.warning("Could not delete phase-1 session: %s", _e)
                                    robometer_session_id = None
                                    current_phase = 2
                                    current_task = args.reward_text_phase2
                                    text_embeds_active = text_embeds_phase2
                                    if args.mixed_reward:
                                        text_embeds_neg_active = text_embeds
                                    try:
                                        robometer_session_id = stream_create(
                                            args.eval_server_url,
                                            current_task,
                                            timeout_s=args.robometer_timeout_s,
                                        )
                                        stream_append_frame(
                                            args.eval_server_url, robometer_session_id,
                                            real_cam0, timeout_s=args.robometer_timeout_s,
                                        )
                                        logger.info("Phase-2 session created: %s", robometer_session_id)
                                    except Exception as _e:
                                        logger.warning("Could not create phase-2 session: %s", _e)
                                        robometer_session_id = None
                            except Exception as _e:
                                logger.warning("stream_append_frame failed: %s", _e)

                    action_chunks = policy.get_action_samples(obs_dict, args.n_samples, binary_gripper=False)

                    if wm_model is not None and tactile_encoder is not None and tactile_buffer:
                        state = obs_dict["state"].astype(np.float32)
                        time_start = time.time()
                        nh_wm = int(getattr(wm_model, "num_hist", 1))
                        pred_embeds_seq, rewards_seq = predict_tactile_rollout(
                            wm_model,
                            tactile_emb_history,
                            state,
                            action_chunks,
                            norm_stats,
                            text_embeds_active,
                            device,
                            rollout_steps=args.wm_rollout_steps,
                            tactile_emb_fallback=current_tactile_emb,
                            use_amp=args.amp,
                            temporal_mode=args.temporal_mode,
                            state_history=state_history,
                            logged_cam2=data["camera_2"] if nh_wm > 1 else None,
                            logged_proprio=data["proprio"] if nh_wm > 1 else None,
                            tactile_encoder=tactile_encoder if nh_wm > 1 else None,
                            sensor_id=sensor_id,
                            vision_emb_histories=vision_emb_histories,
                            text_embeds_neg=text_embeds_neg_active,
                        )
                        time_end = time.time()
                        # print(f"Predict rollout time: {time_end - time_start:.3f}s")
                        time_start = time.time()
                        wm_steps = int(np.asarray(rewards_seq).shape[1])
                        rewards_agg = rewards_seq.mean(axis=1)
                        # print(
                        #     f"Rewards (mean over {wm_steps} WM steps): "
                        #     f"{np.asarray(rewards_agg).round(4).tolist()}"
                        # )
                        data["pred_embeds_log"].append(
                            pred_embeds_seq.detach().cpu().numpy()
                        )
                        data["rewards_log"].append(
                            np.asarray(rewards_seq, dtype=np.float32)
                        )
                        data["action_chunks_log"].append(
                            np.asarray(action_chunks, dtype=np.float32)
                        )
                        data["decision_ee_xyz"].append(
                            np.asarray(obs_dict["state"][:3], dtype=np.float32)
                        )
                        data["policy_query_frame_idx"].append(
                            int(len(data["camera_0"]) - 1)
                        )

                        if reward_plot_dir is not None:
                            ac_plot = np.asarray(action_chunks, dtype=np.float32)[
                                :, :wm_steps, :
                            ]
                            plot_reward_vs_actions(
                                ac_plot,
                                rewards_seq,
                                str(reward_plot_dir / f"query_{i:05d}.png"),
                            )

                        pred_embeds_last = pred_embeds_seq[:, -1, :, :]
                        if tactile_decoder is not None and pred_vis_dir is not None:
                            pred_imgs = decode_pred_embeds(
                                tactile_decoder, pred_embeds_last, device
                            )
                            if pred_imgs:
                                best_idx = int(np.argmax(rewards_agg))
                                best_img = (pred_imgs[best_idx] * 255).astype(np.uint8)
                                cv2.imwrite(
                                    str(pred_vis_dir / f"step_{i:05d}_best.png"),
                                    cv2.cvtColor(best_img, cv2.COLOR_RGB2BGR),
                                )
                                n_cols = min(5, len(pred_imgs))
                                n_rows = (len(pred_imgs) + n_cols - 1) // n_cols
                                grid_h, grid_w = pred_imgs[0].shape[:2]
                                grid_img = np.zeros((n_rows * grid_h, n_cols * grid_w, 3), dtype=np.uint8)
                                for idx, img in enumerate(pred_imgs):
                                    r, c = idx // n_cols, idx % n_cols
                                    grid_img[r * grid_h : (r + 1) * grid_h, c * grid_w : (c + 1) * grid_w] = (
                                        img * 255
                                    ).astype(np.uint8)
                                cv2.imwrite(
                                    str(pred_vis_dir / f"step_{i:05d}_grid.png"),
                                    cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR),
                                )
                        time_end = time.time()
                        # print(f"Visualization time: {time_end - time_start:.3f}s")
                        time_start = time.time()
                        action_chunk_queue.extend(
                            select_action_chunk_by_reward(
                                action_chunks,
                                rewards_agg,
                                mode=args.action_select,
                                temperature=args.softmax_temp,
                            )
                        )
                        time_end = time.time()
                        # print(f"Action chunk selection time: {time_end - time_start:.3f}s")
                    else:
                        data["action_chunks_log"].append(np.asarray(action_chunks, dtype=np.float32))
                        data["decision_ee_xyz"].append(
                            np.asarray(obs_dict["state"][:3], dtype=np.float32)
                        )
                        data["policy_query_frame_idx"].append(
                            int(len(data["camera_0"]) - 1)
                        )
                        action_chunk_queue.extend([action_chunks[0][t] for t in range(ACT_HORIZON)])
                if warmup_step <= 3:
                    warmup_step += 1
                    action_chunk_queue = []
                    continue
                action = action_chunk_queue.pop(0)
                # if args.binary_gripper:
                # action[-1] = 1.0 if action[-1] > 0.0 else -1.0
                print(f"step {i} action: {action}")
                data["actions"].append(action)
                i += 1

                if controller_type == "OSC_YAW":
                    action[3:5] = 0.0
                elif controller_type == "OSC_POSITION":
                    action[3:6] = 0.0

                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )

                if i > 900:
                    break
        except KeyboardInterrupt:
            print('\nKeyboard interrupt detected, saving trajectory...')
            interrupted = True

        # ── Clean up robometer session for this episode ───────────────────────
        if robometer_session_id is not None:
            try:
                stream_delete(
                    args.eval_server_url, robometer_session_id,
                    timeout_s=args.robometer_timeout_s,
                )
                logger.info("Robometer session deleted: %s", robometer_session_id)
            except Exception as _e:
                logger.warning("Could not delete robometer session: %s", _e)
            robometer_session_id = None

        reset_joints_to(robot_interface, init_joints)
        time.sleep(1)
        label = input(
            "Enter label (0=left success, 1=left failure, 2=right success, 3=right failure): "
        )

        episode_id = len(list(Path(args.folder).glob("traj_*.hdf5")))
        print('episode_id', episode_id)

        # ── Save per-camera videos ────────────────────────────────────────────
        episode_video_folder = args.folder / "videos" / f"episode_{episode_id:04d}"
        episode_video_folder.mkdir(parents=True, exist_ok=True)
        for cid in camera_ids:
            frames = data[f"camera_{cid}"]
            if frames:
                vid_path = episode_video_folder / f"camera_{cid}.mp4"
                imageio.mimsave(str(vid_path), frames, fps=10)
        print(f"Videos saved → {episode_video_folder}")

        with h5py.File(args.folder / f"traj_{episode_id:04d}.hdf5", "w") as f:
            grp = f.create_group("data")
            grp.attrs["label"] = label
            grp.attrs["video_path"] = episode_video_folder.as_posix()
            # Camera frames are stored as videos; skip raw arrays in HDF5
            skip_keys = {
                "decision_ee_xyz", "policy_query_frame_idx", "rewards_log",
                "pred_embeds_log", "action_chunks_log", "proprio", "phase_log",
                *(f"camera_{cid}" for cid in camera_ids),
            }
            for key, value in data.items():
                if key in skip_keys or not value:
                    continue
                grp.create_dataset(key, data=value)
            if data.get("proprio"):
                grp.create_dataset(
                    "proprio",
                    data=np.stack(data["proprio"], axis=0).astype(np.float32),
                )
            if data["decision_ee_xyz"]:
                grp.create_dataset(
                    "decision_ee_xyz",
                    data=np.stack(data["decision_ee_xyz"], axis=0),
                )
            if data["policy_query_frame_idx"]:
                grp.create_dataset(
                    "policy_query_frame_idx",
                    data=np.asarray(data["policy_query_frame_idx"], dtype=np.int64),
                )
            if data["action_chunks_log"]:
                grp.create_dataset(
                    "action_chunks_log",
                    data=np.stack(data["action_chunks_log"], axis=0),
                )
            if data["pred_embeds_log"]:
                grp.create_dataset(
                    "pred_embeds_log",
                    data=np.stack(data["pred_embeds_log"], axis=0),
                )
            if data["rewards_log"]:
                grp.create_dataset(
                    "rewards_log",
                    data=np.stack(data["rewards_log"], axis=0),
                )
            if data.get("phase_log"):
                grp.create_dataset(
                    "phase_log",
                    data=np.asarray(data["phase_log"], dtype=np.int32),
                )
            if args.separate_phase:
                grp.attrs["reward_text_phase2"] = args.reward_text_phase2
                grp.attrs["phase_threshold"] = args.phase_threshold
        print(f"Trajectory saved → {args.folder / f'traj_{episode_id:04d}.hdf5'}")
        episode_id += 1

        if interrupted:
            robot_interface.close()
            print('Exiting...')
            break

        robot_interface.close()


if __name__ == "__main__":
    main()
