# Publication cleanup (2026-10-27): adapted from deoxys_control/deoxys/examples/run_dp_tactile_sampling_vision_robometer_log.py.
"""
run_dp_tactile_sampling_vision_robometer_log.py

Like run_dp_tactile_sampling_vision_robometer.py, but saves per-episode videos
(one mp4 per camera) alongside the HDF5 trajectory file, mirroring the logging
approach used in run_dp_model_tactile_liquid_log.py.

Video layout:
  <folder>/videos/episode_XXXX/camera_0.mp4
  <folder>/videos/episode_XXXX/camera_1.mp4
  <folder>/videos/episode_XXXX/camera_2.mp4

The HDF5 file stores a `video_path` dataset (episode folder) instead of raw
camera frames to keep file sizes manageable.

Start the robometer streaming eval server first (in another terminal):
  uv run python robometer/evals/eval_server_streaming.py \\
      model_path=robometer/Robometer-4B num_gpus=1 server_port=8001
"""

from __future__ import annotations

import argparse
from typing import Optional
import io
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
import requests
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as _tv_transforms
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

# ── dino_wm ──────────────────────────────────────────────────────────────────

from dino_wm.dino_models import TactileVideoTransformer, NormStats
from dino_wm.dino_models_k_step import TactileKStepTransformer

# ── robometer client helpers ─────────────────────────────────────────────────

from liquid_tactile_control.robometer_client import (
    stream_create,
    stream_append_frame,
    stream_evaluate_candidates_batch,
    stream_delete,
)

# ── deoxys ───────────────────────────────────────────────────────────────────
from liquid_tactile_control.robot_config import robot_config_parse_args
from liquid_tactile_control.policy import RealPolicy

import logging
logger = logging.getLogger(__name__)

ACT_HORIZON = 8

CAMERA_EMB_DIMS = {"camera_0": 768, "camera_1": 768, "camera_2": 512}
PATCH_SIDE_DINOV3 = 14

_TACTILE_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_TACTILE_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_TACTILE_BG_OFFSET = 130.0 / 255.0
_TACTILE_NORM_CACHE: dict = {}

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


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    robot_config_parse_args(parser)

    # ── Policy ───────────────────────────────────────────────────────────────
    parser.add_argument('--ckpt_dir', type=str,
                        # default=None)
                        default=None)
    parser.add_argument('--ckpt_iter', type=str, default=23999)#31999)
    parser.add_argument('--use_tactile', action='store_true', default=False)
    parser.add_argument('--n_samples', type=int, default=10)

    # ── World model ──────────────────────────────────────────────────────────
    parser.add_argument('--wm_checkpoint', type=str, default=None,
                        help='Path to world model checkpoint (.pth)')
    parser.add_argument('--anytouch_checkpoint', type=str, default=None,
                        help='AnyTouch checkpoint (needed when camera_2 is in wm_cameras)')
    parser.add_argument('--norm_stats_json', type=str, default=None,
                        help='norm_stats.json path (default: sibling of wm_checkpoint)')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile on world model')
    parser.add_argument('--amp', action='store_true',
                        help='FP16 autocast for world model inference')
    parser.add_argument('--wm_arch', type=str, choices=['video', 'k_step'], default='video')
    parser.add_argument('--temporal_mode', type=str, choices=['lpb', 'consecutive'], default='lpb')
    parser.add_argument('--num_hist', type=int, default=None)
    parser.add_argument('--num_pred', type=int, default=1)
    parser.add_argument('--frameskip', type=int, default=8)
    parser.add_argument('--wm_cameras', type=str, default='camera_0,camera_2',
                        help='Cameras the world model was trained on. Must include camera_0.')
    parser.add_argument('--dinov3_hub_dir', type=str,
                        default=None)
    parser.add_argument('--dinov3_weights', type=str,
                        default=None)

    # ── Camera-1 decoder (for decoding predicted embeddings → RGB) ───────────
    parser.add_argument('--decoder_camera1', type=str, default=None,
                        help='Path to VQVAE decoder checkpoint for camera_1 (768-dim)')

    # ── Robometer reward ─────────────────────────────────────────────────────
    parser.add_argument('--eval_server_url', type=str, default='http://localhost:8000',
                        help='Robometer streaming eval server URL (default: port 8000)')
    parser.add_argument('--reward_task', type=str, default='',
                        help='Task description sent to robometer (required)')
    parser.add_argument('--robometer_timeout_s', type=float, default=30.0,
                        help='HTTP timeout for robometer calls (default: 30)')

    # ── Two-phase task switching ──────────────────────────────────────────────
    parser.add_argument('--separate_phase', action='store_true', default=False,
                        help='Enable two-phase reward: switch task text when reward exceeds threshold')
    parser.add_argument('--reward_task_phase2', type=str, default='',
                        help='Task description for phase 2 (used when --separate_phase is set)')
    parser.add_argument('--phase_threshold', type=float, default=0.64,
                        help='Reward threshold to trigger switch from phase 1 to phase 2 (default: 0.64)')

    # ── Lookahead ─────────────────────────────────────────────────────────────
    parser.add_argument('--lookahead_steps', type=int, default=1,
                        help='Number of autoregressive WM prediction steps before evaluating '
                             'the visual reward. 1 = predict end of current chunk (default); '
                             '2 = predict two chunks ahead. Requires the diffusion policy '
                             'ac_chunk to be >= lookahead_steps * ACT_HORIZON.')

    # ── Action selection ─────────────────────────────────────────────────────
    parser.add_argument('--action_select', type=str, default='argmax',
                        choices=['argmax', 'softmax', 'argmin'])
    parser.add_argument('--softmax_temp', type=float, default=1.0)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# World model helpers  (shared with run_dp_model_tactile_sampling_vision.py)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_compile_prefix(state_dict):
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k.replace(prefix, "", 1): v for k, v in state_dict.items()}


def _is_k_step_wm(m):
    if isinstance(m, TactileKStepTransformer):
        return True
    orig = getattr(m, "_orig_mod", None)
    if isinstance(orig, TactileKStepTransformer):
        return True
    return getattr(m, "action_per_frame_dim", None) is not None


def _wm_context_length(wm_model):
    if _is_k_step_wm(wm_model):
        return int(getattr(wm_model, "num_hist"))
    return int(wm_model.temp_embedding.shape[1])


def _tactile_emb_history_maxlen(wm_model, temporal_mode):
    if wm_model is None:
        return None
    if _is_k_step_wm(wm_model):
        nh = int(getattr(wm_model, "num_hist"))
        fs = int(getattr(wm_model, "frameskip"))
        if temporal_mode == "lpb":
            return max(nh, (nh - 1) * fs + 1)
        return nh
    return _wm_context_length(wm_model)


def ensure_patch_grid(embd, target_side, device="cuda"):
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
        embd = nn.functional.interpolate(embd, size=(target_side, target_side),
                                          mode="bilinear", align_corners=False)
        embd = rearrange(embd, "(b t) d h w -> b t (h w) d", h=target_side, w=target_side)
    return embd


def k_step_hist_frame_indices(t_last, num_hist, frameskip, temporal_mode):
    num_hist, fs = int(num_hist), int(frameskip)
    if temporal_mode == "consecutive":
        idxs = [t_last - num_hist + 1 + j for j in range(num_hist)]
    else:
        idxs = [t_last - (num_hist - 1 - j) * fs for j in range(num_hist)]
    return [max(0, min(i, t_last)) for i in idxs]


def build_tactile_wm_context(emb_history, h_ctx, device, emb_fallback=None):
    hist = list(emb_history)
    if not hist:
        if emb_fallback is None:
            raise ValueError("emb_history is empty and no fallback provided")
        e = emb_fallback.detach().to(device)
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


def _action_window_from_chunk(action_chunks, rollout_step, h_ctx, device):
    N, t_hor, d_ac = action_chunks.shape
    idxs = [min(rollout_step + j, t_hor - 1) for j in range(h_ctx)]
    parts = [action_chunks[:, idx:idx + 1, :] for idx in idxs]
    return torch.cat(parts, dim=1).to(device=device, dtype=torch.float32)


def _action_windows_batch(action_chunks, rollout_steps, h_ctx, device):
    return torch.stack(
        [_action_window_from_chunk(action_chunks, s, h_ctx, device) for s in range(rollout_steps)],
        dim=0,
    )


def _k_step_collapsed_actions_batch(action_chunks, rollout_steps, num_hist, frameskip,
                                     temporal_mode, chunk_T, device):
    N, fs = action_chunks.shape[0], int(frameskip)
    out = torch.zeros((rollout_steps, N, num_hist, fs * 7), device=device, dtype=torch.float32)
    for step in range(rollout_steps):
        for j in range(num_hist):
            if temporal_mode == "consecutive":
                idx_start = step - num_hist + 1 + j
            else:
                idx_start = step - (num_hist - 1 - j) * fs
            idx_start = max(0, min(idx_start, chunk_T - 1))
            for k in range(fs):
                idx = min(idx_start + k, chunk_T - 1)
                out[step, :, j, k * 7:(k + 1) * 7] = action_chunks[:, idx, :]
    return out


def _normalize_collapsed_actions_tensor(acs, norm_stats):
    shape = acs.shape
    flat = acs.reshape(-1, 7)
    flat = norm_stats.normalize_acs(flat)
    return flat.view(shape)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_world_model(checkpoint_path, device="cuda", use_compile=True,
                     wm_arch="video", num_hist=1, num_pred=1, frameskip=8, cameras=None):
    if cameras is None:
        cameras = ["camera_0", "camera_2"]
    camera_dims = {c: CAMERA_EMB_DIMS.get(c, 768) for c in cameras}
    if wm_arch == "k_step":
        m = TactileKStepTransformer(
            cameras=cameras, camera_dims=camera_dims,
            condition_cameras=cameras, predict_cameras=cameras,
            common_dim=384, ac_dim=64, state_dim=8,
            num_hist=num_hist, num_pred=num_pred, frameskip=frameskip,
            patches_per_frame=PATCH_SIDE_DINOV3 * PATCH_SIDE_DINOV3,
            depth=6, heads=16, mlp_dim=2048, dropout=0.1,
        ).to(device)
    else:
        m = TactileVideoTransformer(
            cameras=cameras, camera_dims=camera_dims,
            condition_cameras=cameras, predict_cameras=cameras,
            common_dim=384, ac_dim=10, state_dim=8,
            patches_per_frame=PATCH_SIDE_DINOV3 * PATCH_SIDE_DINOV3,
            depth=6, heads=16, mlp_dim=2048, num_frames=9, dropout=0.1,
        ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    m.load_state_dict(_strip_compile_prefix(state))
    m.eval()
    if use_compile and hasattr(torch, "compile"):
        m = torch.compile(m, mode="reduce-overhead")
        logger.info("torch.compile applied to world model (%s)", wm_arch)
    return m


def load_tactile_encoder(checkpoint_path, sensor="gelsight_mini", num_frames=4, device="cuda"):
    from dino_wm.hdf5_to_dataset_tactile import _load_tactile_model
    return _load_tactile_model(checkpoint_path, sensor, num_frames=num_frames, device=str(device))


def load_vision_encoder(hub_dir, weights_path, device="cuda"):
    dino = torch.hub.load(hub_dir, "dinov3_vitb16", source="local", weights=weights_path)
    return dino.to(device).eval()


def load_camera1_decoder(checkpoint_path, device="cuda"):
    """Load VQVAE decoder for camera_1 (DINOv3, 768-dim patch embeddings)."""
    import dino_wm.dino_decoder as _dmod
    if not hasattr(_dmod, "dist_fn"):
        _dmod.dist_fn = type("DistFn", (), {"all_reduce": lambda x: None})()
    from dino_wm.dino_decoder import VQVAE
    dec = VQVAE(emb_dim=768).to(device)
    dec.load_state_dict(torch.load(checkpoint_path, map_location=device))
    dec.eval()
    logger.info("Loaded camera_1 decoder from %s", checkpoint_path)
    return dec


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tactile_clip_norm_tensors(device):
    d = device if isinstance(device, torch.device) else torch.device(device)
    key = str(d)
    if key not in _TACTILE_NORM_CACHE:
        _TACTILE_NORM_CACHE[key] = (
            torch.tensor(_TACTILE_CLIP_MEAN, device=d, dtype=torch.float32).view(3, 1, 1),
            torch.tensor(_TACTILE_CLIP_STD, device=d, dtype=torch.float32).view(3, 1, 1),
        )
    return _TACTILE_NORM_CACHE[key]


def _preprocess_tactile_frame(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.max() <= 1.0:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def _tactile_window_rgb_to_chw_stack(window_rgb_uint8):
    batch = np.stack(window_rgb_uint8, axis=0)
    x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).contiguous()
    return x.float().mul_(1.0 / 255.0)


def get_tactile_embedding_from_image(tactile_model, tactile_frames, sensor_id,
                                      device="cuda", use_amp=False, bg_image=None):
    frames = [_preprocess_tactile_frame(f) for f in tactile_frames]
    pad_ref = frames[0]
    n = len(frames)
    window_frames = [pad_ref] * (4 - n) + frames if n < 4 else frames[-4:]
    stacked = _tactile_window_rgb_to_chw_stack(window_frames)
    dev = device if isinstance(device, torch.device) else torch.device(device)
    x = stacked.to(dev, non_blocking=dev.type == "cuda")
    if bg_image is not None:
        bg_u8 = _preprocess_tactile_frame(bg_image)
        bg_chw = _tactile_window_rgb_to_chw_stack([bg_u8])[0].to(dev)
        x = x - bg_chw.unsqueeze(0) + _TACTILE_BG_OFFSET
    else:
        x = x - x[0:1] + _TACTILE_BG_OFFSET
    x = x.clamp_(0.0, 1.0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean, std = _tactile_clip_norm_tensors(dev)
    x = (x - mean) / std
    sensors = torch.full((1,), sensor_id, dtype=torch.long, device=dev)
    with torch.inference_mode():
        if use_amp and dev.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = tactile_model(x.unsqueeze(0), sensor_type=sensors, probe=False, get_cls=False)
        else:
            outputs = tactile_model(x.unsqueeze(0), sensor_type=sensors, probe=False, get_cls=False)
    patch_emb = outputs[:, 6:, :]
    return F.normalize(patch_emb, dim=-1)


def get_vision_embedding_from_image(vision_encoder, img_np, camera_id, device="cuda"):
    img_pil = PILImage.fromarray(np.uint8(img_np)).convert("RGB")
    transform = _DINO_TRANSFORM_CAM0 if camera_id == 0 else _DINO_TRANSFORM_CAM1
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        patch_emb = vision_encoder.forward_features(img_tensor)["x_norm_patchtokens"]
    return patch_emb  # (1, 196, 768)


# ─────────────────────────────────────────────────────────────────────────────
# Front-camera prediction: WM rollout → camera_0 embedding at last step
# ─────────────────────────────────────────────────────────────────────────────

def _build_wm_inputs(wm_cameras, emb_histories, h_ctx, N, device):
    """Build {cam_name: (N, h_ctx, P, D)} input tensors from embedding history deques."""
    inputs = {}
    for cam_name, hist in emb_histories.items():
        if cam_name not in wm_cameras:
            continue
        fallback = list(hist)[-1] if hist else None
        inp = build_tactile_wm_context(hist, h_ctx, device, emb_fallback=fallback)
        inputs[cam_name] = inp.expand(N, -1, -1, -1).contiguous()
    return inputs


def predict_front_camera_last_step(
    wm_model,
    emb_histories,      # dict: cam_name -> deque of (1, N_patches, D) embeddings
    wm_cameras,         # list of camera names the WM was trained on
    state,              # (8,) current proprio
    action_chunks,      # (N, ACT_HORIZON, 7) torch tensor on device
    norm_stats,
    device,
    use_amp=False,
    temporal_mode="lpb",
):
    if not torch.is_tensor(action_chunks):
        action_chunks = torch.tensor(np.asarray(action_chunks), dtype=torch.float32, device=device)
    N = action_chunks.shape[0]

    if _is_k_step_wm(wm_model):
        num_hist = int(wm_model.num_hist)
        fs = int(wm_model.frameskip)
        chunk_T = int(action_chunks.shape[1])
        h_ctx = num_hist

        inputs = _build_wm_inputs(wm_cameras, emb_histories, h_ctx, N, device)
        st_fallback = torch.tensor(state, dtype=torch.float32, device=device)
        st = st_fallback.view(1, 1, 8).expand(N, h_ctx, 8).contiguous()

        acs_raw = _k_step_collapsed_actions_batch(
            action_chunks, 1, num_hist, fs, temporal_mode, chunk_T, device
        )
        acs = _normalize_collapsed_actions_tensor(acs_raw[0], norm_stats)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                preds, _ = wm_model(inputs, st, acs)

    else:  # video WM — autoregressive over ACT_HORIZON steps
        h_ctx = _wm_context_length(wm_model)
        inputs = _build_wm_inputs(wm_cameras, emb_histories, h_ctx, N, device)
        st = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, 8).expand(N, h_ctx, 8).contiguous()

        acs_raw_all = _action_windows_batch(action_chunks, ACT_HORIZON, h_ctx, device)
        acs_normed_all = norm_stats.normalize_acs(acs_raw_all)

        with torch.inference_mode():
            for step in range(ACT_HORIZON):
                acs = acs_normed_all[step]
                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    preds, pred_state = wm_model(inputs, st, acs)
                for cam in list(inputs.keys()):
                    if cam in preds:
                        next_e = preds[cam][:, -1:]
                        inputs[cam] = torch.cat([inputs[cam][:, 1:], next_e], dim=1)
                next_s = pred_state[:, -1:]
                st = torch.cat([st[:, 1:], next_s], dim=1)

    pred_cam0 = preds["camera_0"][:, -1, :, :].clone()  # (N, N_patches, 768)
    return pred_cam0


def get_action_samples_for_lookahead(policy, obs, N, lookahead_steps, binary_gripper=True):
    images, state = policy._get_images_and_states(obs)
    batch_images = {k: v.repeat(N, *([1] * (v.dim() - 1))) for k, v in images.items()}
    batch_state = state.repeat(N, 1)

    with torch.no_grad():
        actions_norm = policy.agent.get_actions(batch_images, batch_state)  # (N, ac_chunk, 7)

    ac_chunk = actions_norm.shape[1]
    end = min(lookahead_steps * ACT_HORIZON, ac_chunk)

    ac = actions_norm[:, :end, :].cpu().numpy().astype(np.float32)
    ac = (ac + 1.0) / 2.0
    ac = ac * (policy.ac_max - policy.ac_min) + policy.ac_min
    if binary_gripper:
        ac[:, :, -1] = np.sign(ac[:, :, -1])

    chunks = []
    for step in range(lookahead_steps):
        start = step * ACT_HORIZON
        stop = start + ACT_HORIZON
        if start >= end:
            chunks.append(chunks[-1].copy())
        else:
            chunk = ac[:, start:min(stop, end), :]
            if chunk.shape[1] < ACT_HORIZON:
                pad = np.repeat(chunk[:, -1:, :], ACT_HORIZON - chunk.shape[1], axis=1)
                chunk = np.concatenate([chunk, pad], axis=1)
            chunks.append(chunk)

    return chunks  # list of (N, ACT_HORIZON, 7) numpy arrays


def predict_camera_lookahead(
    wm_model,
    emb_histories,
    wm_cameras,
    state,              # (8,) float32 numpy
    action_chunks_list, # list of (N, ACT_HORIZON, 7) tensors already on device
    norm_stats,
    device,
    use_amp=False,
    temporal_mode="lpb",
    reward_camera="camera_0",
):
    N = action_chunks_list[0].shape[0]
    autocast_ctx = torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda"))

    if _is_k_step_wm(wm_model):
        num_hist = int(wm_model.num_hist)
        fs = int(wm_model.frameskip)

        inputs_cur = _build_wm_inputs(wm_cameras, emb_histories, num_hist, N, device)
        st_cur = (torch.tensor(state, dtype=torch.float32, device=device)
                  .view(1, 1, 8).expand(N, num_hist, 8).contiguous())

        preds = None
        pred_state_cur = None

        for step_idx, action_chunks in enumerate(action_chunks_list):
            chunk_T = int(action_chunks.shape[1])
            acs_raw = _k_step_collapsed_actions_batch(
                action_chunks, 1, num_hist, fs, temporal_mode, chunk_T, device
            )
            acs = _normalize_collapsed_actions_tensor(acs_raw[0], norm_stats)

            with torch.inference_mode():
                with autocast_ctx:
                    preds, pred_state_cur = wm_model(inputs_cur, st_cur, acs)
                    preds = {cam: t.clone() for cam, t in preds.items()}
                    pred_state_cur = pred_state_cur.clone()

            if step_idx < len(action_chunks_list) - 1:
                new_inputs = {}
                for cam in wm_cameras:
                    new_frame = preds[cam][:, -1:, :, :]
                    new_inputs[cam] = torch.cat(
                        [inputs_cur[cam][:, 1:], new_frame], dim=1
                    ).contiguous()
                inputs_cur = new_inputs
                st_cur = torch.cat([st_cur[:, 1:], pred_state_cur[:, -1:]], dim=1).contiguous()

    else:  # video WM — autoregressive within each chunk
        h_ctx = _wm_context_length(wm_model)
        inputs_cur = _build_wm_inputs(wm_cameras, emb_histories, h_ctx, N, device)
        st_cur = (torch.tensor(state, dtype=torch.float32, device=device)
                  .view(1, 1, 8).expand(N, h_ctx, 8).contiguous())

        preds = None

        for action_chunks in action_chunks_list:
            acs_raw_all = _action_windows_batch(action_chunks, ACT_HORIZON, h_ctx, device)
            acs_normed_all = norm_stats.normalize_acs(acs_raw_all)

            with torch.inference_mode():
                for step in range(ACT_HORIZON):
                    acs = acs_normed_all[step]
                    with autocast_ctx:
                        preds, pred_state = wm_model(inputs_cur, st_cur, acs)
                    for cam in list(inputs_cur.keys()):
                        if cam in preds:
                            inputs_cur[cam] = torch.cat(
                                [inputs_cur[cam][:, 1:], preds[cam][:, -1:]], dim=1
                            )
                    st_cur = torch.cat([st_cur[:, 1:], pred_state[:, -1:]], dim=1)

    return preds[reward_camera][:, -1, :, :].clone()  # (N, N_patches, D)


def decode_camera1_embeds(decoder, pred_embeds, device="cuda"):
    N = pred_embeds.shape[0]
    inp = pred_embeds.unsqueeze(1).to(device)  # (N, 1, N_patches, 768)
    with torch.no_grad():
        pred_img, _ = decoder(inp)  # (N, 3, H, W) float in [0,1]
    pred_img = pred_img.cpu().numpy()  # (N, 3, H, W)
    images = []
    for i in range(N):
        img = np.clip(pred_img[i].transpose(1, 2, 0), 0.0, 1.0)
        images.append((img * 255).astype(np.uint8))
    # breakpoint()
    return np.stack(images, axis=0)  # (N, H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Robometer reward (streaming)
# ─────────────────────────────────────────────────────────────────────────────

def compute_robometer_rewards_streaming(
    eval_server_url,
    session_id,            # str — active streaming session ID
    predicted_frames,      # (N, H, W, C) uint8 — one decoded frame per sample
    ref_shape,             # (H, W) — spatial shape of real cam1 frames in session
    timeout_s=30.0,
):
    N = len(predicted_frames)
    # breakpoint()
    if predicted_frames.shape[1:3] != ref_shape:
        resized = []
        for n in range(N):
            resized.append(cv2.resize(
                predicted_frames[n], (ref_shape[1], ref_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ))
        # breakpoint()
        predicted_frames = np.stack(resized, axis=0)

    try:
        rewards, _ = stream_evaluate_candidates_batch(
            eval_server_url, session_id, predicted_frames, timeout_s=timeout_s
        )
        return rewards
    except Exception as e:
        logger.warning(
            "Robometer batch evaluate_candidates failed: %s — returning zero rewards", e
        )
        return np.zeros(N, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Action selection
# ─────────────────────────────────────────────────────────────────────────────

def select_action_chunk_by_reward(action_chunks, rewards, mode="argmax", temperature=1.0):
    if mode == "argmax":
        idx = int(np.argmax(rewards))
    elif mode == "argmin":
        idx = int(np.argmin(rewards))
    else:
        r = np.array(rewards, dtype=np.float64)
        r = r - r.max()
        probs = np.exp(r / temperature)
        probs /= probs.sum()
        idx = int(np.random.choice(len(action_chunks), p=probs))
    return [action_chunks[idx][t].copy() for t in range(action_chunks.shape[1])]


def eef_pose_to_state(T, gripper):
    x, y, z = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate(([x, y, z], quat, [gripper]))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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

    if not args.reward_task:
        raise ValueError("--reward_task is required (task description for robometer)")

    wm_cameras_list = [c.strip() for c in args.wm_cameras.split(",")]
    if "camera_0" not in wm_cameras_list:
        raise ValueError("--wm_cameras must include camera_0 for front-camera prediction")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Robot / camera setup ─────────────────────────────────────────────────
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg), gripper_mode='continuous_smooth')
    policy = RealPolicy(["state"], args.ckpt_dir, args.ckpt_iter)

    camera_ids = [0, 1, 2]
    camera_types = ["rs", "zed", "tactile"]
    camera_info = EasyDict({
        "camera_id": camera_ids,
        "camera_type": camera_types,
        "camera_name": [f"camera_{camera_types[i]}_{camera_ids[i]}" for i in range(len(camera_ids))],
    })
    cr_interfaces = {}
    for camera_id in camera_ids:
        if "tactile" in camera_types[camera_id]:
            cr = CameraRedisSubInterface(camera_info=camera_info, camera_id=camera_id,
                                         use_color=False, use_tactile=True)
        else:
            cr = CameraRedisSubInterface(camera_info=camera_info, camera_id=camera_id)
        cr.start()
        cr_interfaces[camera_id] = cr

    controller_type = args.controller_type
    controller_cfg = YamlConfig(os.path.join(config_root, args.controller_cfg)).as_easydict()
    args.folder.mkdir(parents=True, exist_ok=True)

    # ── World model ──────────────────────────────────────────────────────────
    wm_model = None
    norm_stats = None
    tactile_encoder = None
    sensor_id = 0
    vision_encoder = None

    if args.wm_checkpoint:
        nh = args.num_hist if args.num_hist is not None else 1
        npred = args.num_pred
        fs = args.frameskip
        wm_model = load_world_model(
            args.wm_checkpoint, device,
            use_compile=not args.no_compile,
            wm_arch=args.wm_arch,
            num_hist=nh, num_pred=npred, frameskip=fs,
            cameras=wm_cameras_list,
        )
        logger.info("World model loaded: arch=%s cameras=%s", args.wm_arch, wm_cameras_list)

        norm_stats_path = args.norm_stats_json
        if norm_stats_path is None:
            candidate = os.path.join(os.path.dirname(os.path.abspath(args.wm_checkpoint)),
                                     "norm_stats.json")
            if os.path.isfile(candidate):
                norm_stats_path = candidate
        norm_stats = NormStats(norm_stats_path, device=str(device))

        if any(c in ("camera_0", "camera_1") for c in wm_cameras_list):
            vision_encoder = load_vision_encoder(args.dinov3_hub_dir, args.dinov3_weights, device)
            logger.info("DINOv3 vision encoder loaded")

        if "camera_2" in wm_cameras_list and args.anytouch_checkpoint:
            tactile_encoder, sensor_id = load_tactile_encoder(
                args.anytouch_checkpoint, device=str(device)
            )
            logger.info("AnyTouch tactile encoder loaded")

    # ── Camera-1 VQVAE decoder ───────────────────────────────────────────────
    camera1_decoder = None
    if args.decoder_camera1:
        camera1_decoder = load_camera1_decoder(args.decoder_camera1, device)
    elif wm_model is not None:
        logger.warning(
            "--decoder_camera1 not provided; cannot decode predicted cam1 embeddings. "
            "Robometer reward will not be computed."
        )

    wm_h_ctx = _tactile_emb_history_maxlen(wm_model, args.temporal_mode) if wm_model else None

    logger.info("Robometer server: %s  task: '%s'", args.eval_server_url, args.reward_task)

    # ── Reset position ───────────────────────────────────────────────────────
    # init_joints = [-0.18941418455646322, 0.3986268416401311, 0.37023725867044477,
    #                -1.8986898980216536, -0.1220696096038897, 2.1729692311931763,
                #    -2.1428127053565933]
    init_joints = [-0.14091554, 0.02724072, 0.33824686, -2.48239114, -0.12527768, 2.53220921, -2.87378524]


    while True:
        data = {
            "actions": [], "ee_states": [], "joint_states": [],
            "gripper_states": [], "proprio": [],
            "pred_embeds_log": [],  # (N, N_patches, D) predicted camera_0 per policy query
            "rewards_log": [], "action_chunks_log": [],
            "decision_ee_xyz": [], "policy_query_frame_idx": [],
        }
        # Store full-resolution images for video logging
        for camera_id in camera_ids:
            data[f"camera_{camera_id}"] = []

        i = 0
        action_chunk_queue = []
        tactile_buffer = deque(maxlen=4)

        emb_histories: dict[str, deque] = {}
        if wm_h_ctx is not None:
            for cam in wm_cameras_list:
                emb_histories[cam] = deque(maxlen=wm_h_ctx)

        robometer_session_id: Optional[str] = None
        robometer_ref_shape: Optional[tuple] = None
        episode_tactile_bg = None
        current_phase = 1
        current_task = args.reward_task
        data["phase_log"] = []

        reset_joints_to(robot_interface, init_joints)
        time.sleep(1)
        print("Beginning trajectory")
        warmup_step = 0
        interrupted = False
        start = False
        t_count = 0

        try:
            while True:
                step_start = time.time()
                last_state = robot_interface._state_buffer[-1]
                last_gripper = robot_interface._gripper_state_buffer[-1]

                obs_dict = {}
                obs_dict["state"] = eef_pose_to_state(
                    np.array(last_state.O_T_EE).reshape(4, 4).T,
                    np.array(last_gripper.width),
                )
                data["proprio"].append(np.asarray(obs_dict["state"], dtype=np.float32).copy())
                data["ee_states"].append(np.array(last_state.O_T_EE))
                data["joint_states"].append(np.array(last_state.q))
                data["gripper_states"].append(np.array(last_gripper.width))

                # Capture images: store full-res for video, resize for policy obs
                for camera_id in camera_ids:
                    raw = cr_interfaces[camera_id].get_img()
                    if camera_id == 0:
                        color_img = raw["color"]
                        obs_img = cv2.resize(color_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    elif camera_id == 1:
                        color_img = raw["color"]
                        obs_img = cv2.resize(color_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    else:  # camera_id == 2 (tactile)
                        color_img = raw["tactile"]
                        obs_img = color_img
                    data[f"camera_{camera_id}"].append(color_img)
                    obs_dict[f"camera_{camera_id}"] = obs_img
                    if args.use_tactile and camera_id == 2:
                        tactile_buffer.append(obs_img.copy())

                if args.use_tactile and episode_tactile_bg is None and obs_dict.get("camera_2") is not None:
                    episode_tactile_bg = np.asarray(obs_dict["camera_2"]).copy()

                # ── Encode current observations into WM history ──────────────
                if wm_model is not None:
                    if "camera_0" in emb_histories and vision_encoder is not None:
                        v_emb = get_vision_embedding_from_image(
                            vision_encoder, obs_dict["camera_0"], 0, device
                        )
                        emb_histories["camera_0"].append(v_emb.detach())

                    if "camera_1" in emb_histories and vision_encoder is not None:
                        v_emb = get_vision_embedding_from_image(
                            vision_encoder, obs_dict["camera_1"], 1, device
                        )
                        emb_histories["camera_1"].append(v_emb.detach())

                    if "camera_2" in emb_histories and tactile_encoder is not None and tactile_buffer:
                        t_emb = get_tactile_embedding_from_image(
                            tactile_encoder, tactile_buffer, sensor_id, device,
                            use_amp=args.amp, bg_image=episode_tactile_bg,
                        )
                        emb_histories["camera_2"].append(t_emb.detach())

                # ── Policy query: sample N action chunks, pick best via robometer ──
                if not action_chunk_queue:
                    if args.lookahead_steps > 1 and hasattr(policy, "agent"):
                        lookahead_chunks = get_action_samples_for_lookahead(
                            policy, obs_dict, args.n_samples, args.lookahead_steps, binary_gripper=False
                        )
                        action_chunks = lookahead_chunks[0]
                        lookahead_tensors = [
                            torch.tensor(c, dtype=torch.float32, device=device)
                            for c in lookahead_chunks
                        ]
                    else:
                        action_chunks = policy.get_action_samples(obs_dict, args.n_samples, binary_gripper=False)
                        lookahead_tensors = None

                    data["policy_query_frame_idx"].append(int(len(data["camera_0"]) - 1))
                    data["decision_ee_xyz"].append(
                        np.asarray(obs_dict["state"][:3], dtype=np.float32)
                    )
                    data["action_chunks_log"].append(np.asarray(action_chunks, dtype=np.float32))

                    real_cam0 = obs_dict["camera_0"].copy()
                    robometer_ref_shape = real_cam0.shape[:2]  # (H, W)

                    if warmup_step > 3 and robometer_session_id is None:
                        try:
                            robometer_session_id = stream_create(
                                args.eval_server_url,
                                current_task,
                                timeout_s=args.robometer_timeout_s,
                            )
                            logger.info(
                                "Robometer streaming session created (phase %d, task='%s'): %s",
                                current_phase, current_task, robometer_session_id,
                            )
                        except Exception as e:
                            logger.warning(
                                "Could not create robometer session: %s", e
                            )

                    if warmup_step > 3 and robometer_session_id is not None:
                        try:
                            _hist_reward, _ = stream_append_frame(
                                args.eval_server_url,
                                robometer_session_id,
                                real_cam0,
                                timeout_s=args.robometer_timeout_s,
                            )
                            # breakpoint()
                            logger.info(
                                "Robometer history frame appended, progress=%.4f",
                                _hist_reward,
                            )
                        except Exception as e:
                            logger.warning(
                                "stream_append_frame failed: %s — session may be stale", e
                            )

                    if (warmup_step > 3
                            and wm_model is not None and camera1_decoder is not None
                            and emb_histories.get("camera_0")):
                        t0 = time.time()
                        state_vec = obs_dict["state"].astype(np.float32)

                        if lookahead_tensors is not None:
                            pred_cam1 = predict_camera_lookahead(
                                wm_model=wm_model,
                                emb_histories=emb_histories,
                                wm_cameras=wm_cameras_list,
                                state=state_vec,
                                action_chunks_list=lookahead_tensors,
                                norm_stats=norm_stats,
                                device=device,
                                use_amp=args.amp,
                                temporal_mode=args.temporal_mode,
                                reward_camera="camera_0",
                            )  # (N, N_patches, 768)
                        else:
                            ac_tensor = torch.tensor(
                                np.asarray(action_chunks), dtype=torch.float32, device=device
                            )
                            pred_cam1 = predict_front_camera_last_step(
                                wm_model=wm_model,
                                emb_histories=emb_histories,
                                wm_cameras=wm_cameras_list,
                                state=state_vec,
                                action_chunks=ac_tensor,
                                norm_stats=norm_stats,
                                device=device,
                                use_amp=args.amp,
                                temporal_mode=args.temporal_mode,
                            )  # (N, N_patches, 768)

                        t1 = time.time()
                        logger.info("WM forward (%d-step lookahead): %.3fs",
                                    args.lookahead_steps, t1 - t0)

                        pe_np = pred_cam1.detach().float().cpu().numpy()
                        data["pred_embeds_log"].append(np.asarray(pe_np, dtype=np.float32))


                        predicted_frames = decode_camera1_embeds(
                            camera1_decoder, pred_cam1, device
                        )  # (N, 224, 224, 3) uint8

                        t2 = time.time()
                        logger.info("Decode: %.3fs", t2 - t1)

                        if robometer_session_id is not None:
                            rewards = compute_robometer_rewards_streaming(
                                eval_server_url=args.eval_server_url,
                                session_id=robometer_session_id,
                                predicted_frames=predicted_frames,
                                ref_shape=robometer_ref_shape,
                                timeout_s=args.robometer_timeout_s,
                            )
                        else:
                            rewards = np.zeros(len(predicted_frames), dtype=np.float32)
                        t3 = time.time()
                        rewards_variance = np.var(rewards)
                        print(f"Rewards variance: {rewards_variance}")
                        logger.info(
                            "Robometer: %.3fs  rewards=%s  best=%d (%.4f)",
                            t3 - t2, np.round(rewards, 4).tolist(),
                            int(np.argmax(rewards)), float(np.max(rewards)),
                        )

                        data["rewards_log"].append(rewards.copy())
                        data["phase_log"].append(current_phase)
                        if np.max(rewards) > args.phase_threshold and not start:
                            start = True
                        if start:
                            t_count+= 1


                        # ── Phase switching ──────────────────────────────────
                        if (args.separate_phase
                                and current_phase == 1
                                and args.reward_task_phase2
                                and t_count > 4):
                                # and float(np.max(rewards)) > args.phase_threshold):
                            ## change the color of the text to red in the logger
                            # from termcolor import colored
                            logger.info(
                                "Phase 1 max reward %.4f > threshold %.4f — "
                                "switching to phase 2 task: '%s'",
                                float(np.max(rewards)), args.phase_threshold,
                                args.reward_task_phase2,
                            )
                            input("Press Enter to continue...")
                            import termcolor
                            print(termcolor.colored(
                                f"[Phase switch] reward {float(np.max(rewards)):.4f} "
                                f"> {args.phase_threshold} — switching to phase 2"
                            ), "red")
                            if robometer_session_id is not None:
                                try:
                                    stream_delete(
                                        args.eval_server_url, robometer_session_id,
                                        timeout_s=args.robometer_timeout_s,
                                    )
                                    logger.info("Phase-1 session deleted: %s", robometer_session_id)
                                except Exception as e:
                                    logger.warning("Could not delete phase-1 session: %s", e)
                                robometer_session_id = None
                            current_phase = 2
                            current_task = args.reward_task_phase2
                            try:
                                robometer_session_id = stream_create(
                                    args.eval_server_url,
                                    current_task,
                                    timeout_s=args.robometer_timeout_s,
                                )
                                logger.info(
                                    "Phase-2 robometer session created: %s", robometer_session_id
                                )
                                # Seed phase-2 session with the current real frame
                                stream_append_frame(
                                    args.eval_server_url,
                                    robometer_session_id,
                                    real_cam0,
                                    timeout_s=args.robometer_timeout_s,
                                )
                            except Exception as e:
                                logger.warning("Could not create phase-2 session: %s", e)
                                robometer_session_id = None

                        action_chunk_queue.extend(
                            select_action_chunk_by_reward(
                                action_chunks, rewards,
                                mode=args.action_select,
                                temperature=args.softmax_temp,
                            )
                        )
                    else:
                        action_chunk_queue.extend(
                            [action_chunks[0][t] for t in range(ACT_HORIZON)]
                        )

                if warmup_step <= 3:
                    warmup_step += 1
                    action_chunk_queue = []
                    continue

                action = action_chunk_queue.pop(0)
                data["actions"].append(action)
                i += 1

                if controller_type == "OSC_YAW":
                    action[3:5] = 0.0
                elif controller_type == "OSC_POSITION":
                    action[3:6] = 0.0    # init_joints = [-0.18941418455646322, 0.3986268416401311, 0.37023725867044477,
    #                -1.8986898980216536, -0.1

                print(f"Step {i}  action={np.round(action, 4).tolist()}")
                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )

                if i > 900:
                    break

        except KeyboardInterrupt:
            print("\nKeyboard interrupt — saving trajectory …")
            interrupted = True

        # Clean up the streaming session for this episode
        if robometer_session_id is not None:
            try:
                stream_delete(args.eval_server_url, robometer_session_id,
                              timeout_s=args.robometer_timeout_s)
                logger.info("Robometer streaming session deleted: %s", robometer_session_id)
            except Exception as e:
                logger.warning("Could not delete robometer session: %s", e)
            robometer_session_id = None

        reset_joints_to(robot_interface, init_joints)
        time.sleep(1)

        label = input("Label (0=left success, 1=left failure, 2=right success, 3=right failure): ")
        episode_id = len(list(Path(args.folder).glob("traj_*.hdf5")))

        # ── Save per-camera videos ────────────────────────────────────────────
        episode_video_folder = args.folder / "videos" / f"episode_{episode_id:04d}"
        episode_video_folder.mkdir(parents=True, exist_ok=True)
        for camera_id in camera_ids:
            video_path = episode_video_folder / f"camera_{camera_id}.mp4"
            imageio.mimsave(str(video_path), data[f"camera_{camera_id}"], fps=10)
        print(f"Videos saved to {episode_video_folder}")

        # ── Save HDF5 (camera frames replaced by video_path) ─────────────────
        with h5py.File(args.folder / f"traj_{episode_id:04d}.hdf5", "w") as f:
            grp = f.create_group("data")
            grp.attrs["label"] = label
            grp.attrs["reward_task"] = args.reward_task

            skip_keys = {
                "pred_embeds_log", "rewards_log", "action_chunks_log",
                "decision_ee_xyz", "policy_query_frame_idx", "proprio",
                "camera_0", "camera_1", "camera_2", "phase_log",
            }
            for key, value in data.items():
                if key in skip_keys or not value:
                    continue
                grp.create_dataset(key, data=value)

            grp.create_dataset("video_path",
                               data=np.bytes_(episode_video_folder.as_posix()))

            if data["proprio"]:
                grp.create_dataset("proprio",
                                   data=np.stack(data["proprio"], axis=0).astype(np.float32))
            if data["decision_ee_xyz"]:
                grp.create_dataset("decision_ee_xyz",
                                   data=np.stack(data["decision_ee_xyz"], axis=0))
            if data["policy_query_frame_idx"]:
                grp.create_dataset("policy_query_frame_idx",
                                   data=np.asarray(data["policy_query_frame_idx"], dtype=np.int64))
            if data["action_chunks_log"]:
                grp.create_dataset("action_chunks_log",
                                   data=np.stack(data["action_chunks_log"], axis=0))
            if data["rewards_log"]:
                grp.create_dataset("rewards_log",
                                   data=np.stack(data["rewards_log"], axis=0))
            if data.get("phase_log"):
                grp.create_dataset("phase_log",
                                   data=np.asarray(data["phase_log"], dtype=np.int32))
            if args.separate_phase:
                grp.attrs["reward_task_phase2"] = args.reward_task_phase2
                grp.attrs["phase_threshold"] = args.phase_threshold
            if data["pred_embeds_log"]:
                grp.create_dataset(
                    "pred_embeds_log",
                    data=np.stack(data["pred_embeds_log"], axis=0),
                )

        print(f"Saved traj_{episode_id:04d}.hdf5  (label={label})")

        if interrupted:
            robot_interface.close()
            print("Exiting…")
            break


if __name__ == "__main__":
    main()
