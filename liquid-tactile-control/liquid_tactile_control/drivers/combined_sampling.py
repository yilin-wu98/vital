# Publication cleanup (2026-10-27): adapted from deoxys_control/deoxys/examples/run_dp_tactile_vision_combined_liquid_log.py.
"""
run_dp_tactile_vision_combined_liquid_log.py

Uses a single multi-head world model (e.g. trained on camera_0 + camera_2) to
compute both reward signals in one forward pass:

  - Vision reward : preds["camera_0"] decoded to RGB → robometer streaming server
                    with two-phase task switching (from run_dp_tactile_sampling_vision_robometer_log.py)
  - Tactile reward: preds["camera_2"] mean-pooled → CLIP cosine similarity with text
                    with optional contrastive reward and phase text switching
                    (from run_model_tactile_guidance_liquid_log.py)

Selection logic (std-based fallback):
  1. vision_std  >= --vision_std_threshold  → select by vision reward
  2. tactile_std >= --tactile_std_threshold → select by tactile reward (fallback)
  3. Both too low                           → random sample

Videos are saved separately per episode (camera_0/1/2.mp4) like
run_dp_model_tactile_liquid_log.py; HDF5 stores video_path instead of raw frames.

Start the robometer server first (separate terminal):
  uv run python robometer/evals/eval_server_streaming.py \\
      model_path=robometer/Robometer-4B num_gpus=1 server_port=8001
"""

from __future__ import annotations

import argparse
from typing import Optional
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
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as _tv_transforms
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer


from dino_wm.dino_models import TactileVideoTransformer, NormStats
from dino_wm.dino_models_k_step import TactileKStepTransformer


from liquid_tactile_control.robometer_client import (
    stream_create,
    stream_append_frame,
    stream_evaluate_candidates_batch,
    stream_delete,
)

from liquid_tactile_control.robot_config import robot_config_parse_args
from liquid_tactile_control.policy import RealPolicy

import logging
logger = logging.getLogger(__name__)

ACT_HORIZON = 8
CAMERA_EMB_DIMS = {"camera_0": 768, "camera_1": 768, "camera_2": 512}
PATCH_SIDE_DINOV3 = 14

_TACTILE_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_TACTILE_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
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
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    robot_config_parse_args(parser)

    # Policy
    parser.add_argument("--ckpt_dir", type=str,
                        default=None)
    parser.add_argument("--ckpt_iter", type=str, default=23999)
    parser.add_argument("--n_samples", type=int, default=10)

    # Single shared world model
    parser.add_argument("--wm_checkpoint", type=str, required=True,
                        help="Multi-head WM checkpoint (must predict camera_0 and camera_2)")
    parser.add_argument("--wm_cameras", type=str, default="camera_0,camera_2",
                        help="Cameras the WM was trained on (comma-separated)")
    parser.add_argument("--norm_stats_json", type=str, default=None,
                        help="norm_stats.json path (default: sibling of wm_checkpoint)")
    parser.add_argument("--wm_arch", type=str, choices=["video", "k_step"], default="k_step")
    parser.add_argument("--temporal_mode", type=str, choices=["lpb", "consecutive"], default="lpb")
    parser.add_argument("--num_hist", type=int, default=None)
    parser.add_argument("--num_pred", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=8)

    # Which WM output heads to use for each reward signal
    parser.add_argument("--vision_reward_camera", type=str, default="camera_0",
                        help="WM output head used for vision reward (decoded → robometer)")
    parser.add_argument("--tactile_reward_camera", type=str, default="camera_2",
                        help="WM output head used for tactile reward (CLIP cosine)")

    # Tactile encoder + CLIP
    parser.add_argument("--anytouch_checkpoint", type=str, required=True,
                        help="AnyTouch checkpoint (tactile encoder + CLIP text head)")
    parser.add_argument("--reward_text", type=str, default="wiping the board",
                        help="Text description for CLIP tactile cosine reward (phase 1)")
    parser.add_argument("--reward_text_phase2", type=str, default="",
                        help="Text description for CLIP tactile cosine reward (phase 2)")

    # DINOv3 (for RGB camera embeddings)
    parser.add_argument("--dinov3_hub_dir", type=str,
                        default=None)
    parser.add_argument("--dinov3_weights", type=str,
                        default=None)

    # Vision reward: VQVAE decoder + robometer
    parser.add_argument("--decoder_camera1", type=str, required=True,
                        help="VQVAE decoder checkpoint for the vision_reward_camera (768-dim)")
    parser.add_argument("--eval_server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--reward_task", type=str, default="",
                        help="Task description for robometer phase 1 (required)")
    parser.add_argument("--robometer_timeout_s", type=float, default=30.0)

    # Two-phase visual reward (from run_dp_tactile_sampling_vision_robometer_log.py)
    parser.add_argument("--separate_phase", action="store_true", default=False,
                        help="Enable two-phase reward: switch task text when reward exceeds threshold")
    parser.add_argument("--reward_task_phase2", type=str, default="",
                        help="Task description for robometer phase 2 (visual reward)")
    parser.add_argument("--phase_threshold", type=float, default=0.64,
                        help="Reward threshold to trigger switch from phase 1 to phase 2 (default: 0.64)")

    # Contrastive tactile reward (from run_model_tactile_guidance_liquid_log.py)
    parser.add_argument("--mixed_reward", action="store_true", default=False,
                        help="Use contrastive tactile reward: r(text_pos) - r(text_neg). "
                             "Phase 1: pos=--reward_text, neg=--reward_text_phase2. "
                             "Phase 2: flipped. Requires --reward_text_phase2.")

    # Lookahead (from run_dp_tactile_sampling_vision_robometer_log.py)
    parser.add_argument("--lookahead_steps", type=int, default=1,
                        help="WM prediction lookahead steps before evaluating visual reward. "
                             "1 = predict end of current chunk (default).")

    # Action selection
    parser.add_argument("--action_select", type=str, default="argmax",
                        choices=["argmax", "softmax", "argmin"])
    parser.add_argument("--softmax_temp", type=float, default=1.0)

    # Misc
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _strip_compile_prefix(sd):
    p = "_orig_mod."
    if not any(k.startswith(p) for k in sd):
        return sd
    return {k.replace(p, "", 1): v for k, v in sd.items()}


def _is_k_step(m):
    if isinstance(m, TactileKStepTransformer):
        return True
    if isinstance(getattr(m, "_orig_mod", None), TactileKStepTransformer):
        return True
    return getattr(m, "action_per_frame_dim", None) is not None


def _ctx_len(m):
    if _is_k_step(m):
        return int(getattr(m, "num_hist"))
    return int(m.temp_embedding.shape[1])


def _history_maxlen(m, temporal_mode):
    if m is None:
        return None
    if _is_k_step(m):
        nh, fs = int(getattr(m, "num_hist")), int(getattr(m, "frameskip"))
        return max(nh, (nh - 1) * fs + 1) if temporal_mode == "lpb" else nh
    return _ctx_len(m)


def _tactile_clip_norm_tensors(device):
    d = device if isinstance(device, torch.device) else torch.device(device)
    key = str(d)
    if key not in _TACTILE_NORM_CACHE:
        _TACTILE_NORM_CACHE[key] = (
            torch.tensor(_TACTILE_CLIP_MEAN, device=d, dtype=torch.float32).view(3, 1, 1),
            torch.tensor(_TACTILE_CLIP_STD,  device=d, dtype=torch.float32).view(3, 1, 1),
        )
    return _TACTILE_NORM_CACHE[key]


def _preprocess_tactile_frame(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.max() <= 1.0:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def _tactile_to_chw(frames_uint8):
    batch = np.stack(frames_uint8, axis=0)
    x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).contiguous()
    return x.float().mul_(1.0 / 255.0)


def eef_pose_to_state(T, gripper):
    x, y, z = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate(([x, y, z], quat, [gripper]))


def ensure_patch_grid(embd, target_side, device):
    B, T, N, D = embd.shape
    side = int(N ** 0.5)
    if side * side != N:
        target_n = target_side ** 2
        embd = embd[:, :, :target_n] if N >= target_n else torch.cat(
            [embd, torch.zeros(B, T, target_n - N, D, device=device, dtype=embd.dtype)], dim=2)
        side = target_side
    if side != target_side:
        embd = rearrange(embd, "b t (h w) d -> (b t) d h w", h=side, w=side)
        embd = F.interpolate(embd, size=(target_side, target_side), mode="bilinear", align_corners=False)
        embd = rearrange(embd, "(b t) d h w -> b t (h w) d", h=target_side, w=target_side)
    return embd


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_world_model(checkpoint_path, cameras, wm_arch, num_hist, num_pred,
                     frameskip, device, use_compile):
    camera_dims = {c: CAMERA_EMB_DIMS.get(c, 768) for c in cameras}
    if wm_arch == "k_step":
        m = TactileKStepTransformer(
            cameras=cameras, camera_dims=camera_dims,
            condition_cameras=cameras, predict_cameras=cameras,
            common_dim=384, ac_dim=64, state_dim=8,
            num_hist=num_hist, num_pred=num_pred, frameskip=frameskip,
            patches_per_frame=PATCH_SIDE_DINOV3 ** 2,
            depth=6, heads=16, mlp_dim=2048, dropout=0.1,
        ).to(device)
    else:
        m = TactileVideoTransformer(
            cameras=cameras, camera_dims=camera_dims,
            condition_cameras=cameras, predict_cameras=cameras,
            common_dim=384, ac_dim=10, state_dim=8,
            patches_per_frame=PATCH_SIDE_DINOV3 ** 2,
            depth=6, heads=16, mlp_dim=2048, num_frames=9, dropout=0.1,
        ).to(device)
    m.load_state_dict(_strip_compile_prefix(torch.load(checkpoint_path, map_location=device)))
    m.eval()
    if use_compile and hasattr(torch, "compile"):
        m = torch.compile(m, mode="reduce-overhead")
        logger.info("torch.compile applied (%s arch=%s)", checkpoint_path, wm_arch)
    return m


def load_tactile_encoder(checkpoint_path, device):
    from dino_wm.hdf5_to_dataset_tactile import _load_tactile_model
    return _load_tactile_model(checkpoint_path, "gelsight_mini", num_frames=4, device=str(device))


def load_clip_for_reward(checkpoint_path, device):
    from anytouch_reward.loading import load_text_encoder
    return load_text_encoder(checkpoint_path, device=device)


def load_vision_encoder(hub_dir, weights_path, device):
    dino = torch.hub.load(hub_dir, "dinov3_vitb16", source="local", weights=weights_path)
    return dino.to(device).eval()


def load_vqvae_decoder(checkpoint_path, emb_dim, device):
    import dino_wm.dino_decoder as _dmod
    if not hasattr(_dmod, "dist_fn"):
        _dmod.dist_fn = type("DistFn", (), {"all_reduce": lambda x: None})()
    from dino_wm.dino_decoder import VQVAE
    dec = VQVAE(emb_dim=emb_dim).to(device)
    dec.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return dec.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_tactile_embedding(model, frames, sensor_id, device, use_amp=False, bg=None):
    frames = [_preprocess_tactile_frame(f) for f in frames]
    n = len(frames)
    window = [frames[0]] * (4 - n) + frames if n < 4 else frames[-4:]
    dev = device if isinstance(device, torch.device) else torch.device(device)
    x = _tactile_to_chw(window).to(dev, non_blocking=dev.type == "cuda")
    ref = _tactile_to_chw([_preprocess_tactile_frame(bg)])[0].to(dev) if bg is not None else x[0:1]
    x = (x - ref.unsqueeze(0) + _TACTILE_BG_OFFSET).clamp_(0.0, 1.0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean, std = _tactile_clip_norm_tensors(dev)
    x = (x - mean) / std
    sensors = torch.full((1,), sensor_id, dtype=torch.long, device=dev)
    with torch.inference_mode():
        ctx = torch.amp.autocast("cuda") if (use_amp and dev.type == "cuda") else torch.no_grad()
        with ctx:
            out = model(x.unsqueeze(0), sensor_type=sensors, probe=False, get_cls=False)
    return F.normalize(out[:, 6:, :], dim=-1)  # (1, N_patches, 512)


def get_vision_embedding(encoder, img_np, camera_id, device):
    img_pil = PILImage.fromarray(np.uint8(img_np)).convert("RGB")
    transform = _DINO_TRANSFORM_CAM0 if camera_id == 0 else _DINO_TRANSFORM_CAM1
    img_t = transform(img_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        return encoder.forward_features(img_t)["x_norm_patchtokens"]  # (1, 196, 768)


# ─────────────────────────────────────────────────────────────────────────────
# WM context builders / action helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_wm_context(history, h_ctx, device, fallback=None):
    """Stack embedding history into (1, h_ctx, N_patches, D), left-padding if short."""
    hist = list(history)
    if not hist:
        e = fallback.detach().to(device)
        if e.dim() == 2:
            e = e.unsqueeze(0)
        inp = e.expand(1, h_ctx, -1, -1).contiguous()
    else:
        seq = torch.cat(hist, dim=0)
        L, P, D = seq.shape
        if L >= h_ctx:
            seq = seq[-h_ctx:]
        else:
            seq = torch.cat([seq[0:1].expand(h_ctx - L, P, D), seq], dim=0)
        inp = seq.unsqueeze(0).contiguous().to(device)
    if inp.shape[2] != PATCH_SIDE_DINOV3 ** 2 or int(inp.shape[2] ** 0.5) != PATCH_SIDE_DINOV3:
        inp = ensure_patch_grid(inp, PATCH_SIDE_DINOV3, device)
    return inp


def _action_windows_batch(ac, steps, h_ctx, device):
    N, T, _ = ac.shape
    def window(s):
        idxs = [min(s + j, T - 1) for j in range(h_ctx)]
        return torch.cat([ac[:, i:i+1] for i in idxs], dim=1).to(device=device, dtype=torch.float32)
    return torch.stack([window(s) for s in range(steps)], dim=0)


def _k_step_collapsed_actions(ac, steps, num_hist, fs, temporal_mode, chunk_T, device):
    N = ac.shape[0]
    out = torch.zeros((steps, N, num_hist, fs * 7), device=device, dtype=torch.float32)
    for step in range(steps):
        for j in range(num_hist):
            i0 = (step - num_hist + 1 + j) if temporal_mode == "consecutive" else (step - (num_hist - 1 - j) * fs)
            i0 = max(0, min(i0, chunk_T - 1))
            for k in range(fs):
                idx = min(i0 + k, chunk_T - 1)
                out[step, :, j, k*7:(k+1)*7] = ac[:, idx, :]
    return out


def _normalize_collapsed(acs, norm_stats):
    shape = acs.shape
    return norm_stats.normalize_acs(acs.reshape(-1, 7)).view(shape)


# ─────────────────────────────────────────────────────────────────────────────
# Single WM forward → predictions for all cameras
# ─────────────────────────────────────────────────────────────────────────────

def predict_all_cameras(
    wm_model, emb_histories, wm_cameras, state, action_chunks,
    norm_stats, device, use_amp, temporal_mode,
):
    """
    One forward pass through the multi-head WM.

    Returns:
        preds_last: dict {cam_name: (N, N_patches, D)} — predicted embedding
                    at the last step of the action horizon for each camera head.
    """
    if not torch.is_tensor(action_chunks):
        action_chunks = torch.tensor(np.asarray(action_chunks), dtype=torch.float32, device=device)
    N, chunk_T = action_chunks.shape[0], int(action_chunks.shape[1])

    def _build_inputs(h_ctx):
        out = {}
        for cam in wm_cameras:
            hist = emb_histories.get(cam, deque())
            fallback = list(hist)[-1] if hist else None
            out[cam] = build_wm_context(hist, h_ctx, device, fallback).expand(N, -1, -1, -1).contiguous()
        return out

    st_1d = torch.tensor(state, dtype=torch.float32, device=device)

    if _is_k_step(wm_model):
        num_hist = int(wm_model.num_hist)
        fs = int(wm_model.frameskip)
        inputs = _build_inputs(num_hist)
        st = st_1d.view(1, 1, 8).expand(N, num_hist, 8).contiguous()
        acs = _normalize_collapsed(
            _k_step_collapsed_actions(action_chunks, 1, num_hist, fs, temporal_mode, chunk_T, device)[0],
            norm_stats,
        )
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                preds, _ = wm_model(inputs, st, acs)

    else:  # video WM — autoregressive over ACT_HORIZON
        h_ctx = _ctx_len(wm_model)
        inputs = _build_inputs(h_ctx)
        st = st_1d.view(1, 1, 8).expand(N, h_ctx, 8).contiguous()
        acs_all = _action_windows_batch(action_chunks, ACT_HORIZON, h_ctx, device)
        acs_all = norm_stats.normalize_acs(acs_all)
        with torch.inference_mode():
            for step in range(ACT_HORIZON):
                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    preds, pred_st = wm_model(inputs, st, acs_all[step])
                for cam in wm_cameras:
                    if cam in preds:
                        inputs[cam] = torch.cat([inputs[cam][:, 1:], preds[cam][:, -1:]], dim=1)
                st = torch.cat([st[:, 1:], pred_st[:, -1:]], dim=1)

    return {cam: preds[cam][:, -1, :, :].clone() for cam in preds}


def predict_all_cameras_lookahead(
    wm_model, emb_histories, wm_cameras, state, action_chunks_list,
    norm_stats, device, use_amp, temporal_mode, reward_camera="camera_0",
):
    """Multi-step lookahead WM prediction; returns only the reward camera embedding."""
    N = action_chunks_list[0].shape[0]
    autocast_ctx = torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda"))

    def _build_inputs(h_ctx):
        out = {}
        for cam in wm_cameras:
            hist = emb_histories.get(cam, deque())
            fallback = list(hist)[-1] if hist else None
            out[cam] = build_wm_context(hist, h_ctx, device, fallback).expand(N, -1, -1, -1).contiguous()
        return out

    if _is_k_step(wm_model):
        num_hist = int(wm_model.num_hist)
        fs = int(wm_model.frameskip)
        inputs_cur = _build_inputs(num_hist)
        st_cur = (torch.tensor(state, dtype=torch.float32, device=device)
                  .view(1, 1, 8).expand(N, num_hist, 8).contiguous())
        preds = None
        pred_state_cur = None
        for step_idx, action_chunks in enumerate(action_chunks_list):
            chunk_T = int(action_chunks.shape[1])
            acs = _normalize_collapsed(
                _k_step_collapsed_actions(action_chunks, 1, num_hist, fs, temporal_mode, chunk_T, device)[0],
                norm_stats,
            )
            with torch.inference_mode():
                with autocast_ctx:
                    preds, pred_state_cur = wm_model(inputs_cur, st_cur, acs)
                    preds = {cam: t.clone() for cam, t in preds.items()}
                    pred_state_cur = pred_state_cur.clone()
            if step_idx < len(action_chunks_list) - 1:
                new_inputs = {}
                for cam in wm_cameras:
                    new_inputs[cam] = torch.cat(
                        [inputs_cur[cam][:, 1:], preds[cam][:, -1:]], dim=1
                    ).contiguous()
                inputs_cur = new_inputs
                st_cur = torch.cat([st_cur[:, 1:], pred_state_cur[:, -1:]], dim=1).contiguous()
    else:
        h_ctx = _ctx_len(wm_model)
        inputs_cur = _build_inputs(h_ctx)
        st_cur = (torch.tensor(state, dtype=torch.float32, device=device)
                  .view(1, 1, 8).expand(N, h_ctx, 8).contiguous())
        preds = None
        for action_chunks in action_chunks_list:
            acs_all = _action_windows_batch(action_chunks, ACT_HORIZON, h_ctx, device)
            acs_all = norm_stats.normalize_acs(acs_all)
            with torch.inference_mode():
                for step in range(ACT_HORIZON):
                    with autocast_ctx:
                        preds, pred_st = wm_model(inputs_cur, st_cur, acs_all[step])
                    for cam in wm_cameras:
                        if cam in preds:
                            inputs_cur[cam] = torch.cat([inputs_cur[cam][:, 1:], preds[cam][:, -1:]], dim=1)
                    st_cur = torch.cat([st_cur[:, 1:], pred_st[:, -1:]], dim=1)

    return {cam: preds[cam][:, -1, :, :].clone() for cam in preds}


def get_action_samples_for_lookahead(policy, obs, N, lookahead_steps, binary_gripper=False):
    images, state = policy._get_images_and_states(obs)
    batch_images = {k: v.repeat(N, *([1] * (v.dim() - 1))) for k, v in images.items()}
    batch_state = state.repeat(N, 1)
    with torch.no_grad():
        actions_norm = policy.agent.get_actions(batch_images, batch_state)
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-signal reward computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_tactile_rewards(pred_emb, text_embeds, text_embeds_neg=None):
    """
    pred_emb      : (N, N_patches, D) — predicted tactile embedding
    text_embeds   : (1, D) — positive CLIP text embedding
    text_embeds_neg: (1, D) or None — negative CLIP text embedding for contrastive reward
    Returns       : (N,) float32 numpy
    """
    emb = F.normalize(pred_emb.mean(dim=1), dim=-1)   # (N, D)
    if text_embeds.dim() == 1:
        text_embeds = text_embeds.unsqueeze(0)
    rewards = (emb @ text_embeds.T).squeeze(-1)  # (N,)
    if text_embeds_neg is not None:
        if text_embeds_neg.dim() == 1:
            text_embeds_neg = text_embeds_neg.unsqueeze(0)
        rewards = rewards - (emb @ text_embeds_neg.T).squeeze(-1)
    return rewards.cpu().numpy()  # (N,)


def decode_vision_frames(decoder, pred_emb, device):
    """(N, N_patches, 768) → (N, H, W, 3) uint8"""
    with torch.no_grad():
        pred_img, _ = decoder(pred_emb.unsqueeze(1).to(device))   # (N, 3, H, W)
    images = []
    for img in pred_img.cpu().numpy():
        images.append((np.clip(img.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8))
    return np.stack(images, axis=0)


def compute_vision_rewards(eval_server_url, session_id, predicted_frames,
                           ref_shape, timeout_s=30.0):
    """Evaluate N candidate frames; returns (N,) float32 rewards."""
    N = len(predicted_frames)
    if predicted_frames.shape[1:3] != ref_shape:
        predicted_frames = np.stack([
            cv2.resize(predicted_frames[n], (ref_shape[1], ref_shape[0]),
                       interpolation=cv2.INTER_LINEAR) for n in range(N)
        ], axis=0)
    try:
        rewards, _ = stream_evaluate_candidates_batch(
            eval_server_url, session_id, predicted_frames, timeout_s=timeout_s,
        )
        return rewards
    except Exception as e:
        logger.warning("Robometer batch evaluate failed: %s — returning zero rewards", e)
        return np.zeros(N, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Combined selection logic
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_rewards(r):
    """Z-score normalize; returns zeros if std is negligible."""
    r = np.array(r, dtype=np.float64)
    std = r.std()
    if std < 1e-8:
        return np.zeros_like(r)
    return (r - r.mean()) / std


def select_action_combined(action_chunks, vision_rewards, tactile_rewards,
                           mode="argmax", temperature=1.0):
    """
    Normalize vision and tactile rewards independently across the sample batch,
    sum them, then select the action with the highest combined reward.

    Returns (selected_actions, combined_rewards).
    """
    combined = _normalize_rewards(vision_rewards) + _normalize_rewards(tactile_rewards)

    if mode == "argmax":
        idx = int(np.argmax(combined))
    elif mode == "argmin":
        idx = int(np.argmin(combined))
    else:
        r = combined - combined.max()
        probs = np.exp(r / temperature); probs /= probs.sum()
        idx = int(np.random.choice(len(action_chunks), p=probs))

    return ([action_chunks[idx][t].copy() for t in range(action_chunks.shape[1])],
            combined)


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
    for cam in (args.vision_reward_camera, args.tactile_reward_camera):
        if cam not in wm_cameras_list:
            raise ValueError(f"--wm_cameras must include {cam}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Robot / camera setup ─────────────────────────────────────────────────
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg), gripper_mode='continuous_smooth')
    policy = RealPolicy(["state"], args.ckpt_dir, args.ckpt_iter)

    camera_ids   = [0, 1, 2]
    camera_types = ["rs", "zed", "tactile"]
    camera_info  = EasyDict({
        "camera_id":   camera_ids,
        "camera_type": camera_types,
        "camera_name": [f"camera_{camera_types[i]}_{camera_ids[i]}" for i in range(len(camera_ids))],
    })
    cr_interfaces = {}
    for cid in camera_ids:
        if "tactile" in camera_types[cid]:
            cr = CameraRedisSubInterface(camera_info=camera_info, camera_id=cid,
                                         use_color=False, use_tactile=True)
        else:
            cr = CameraRedisSubInterface(camera_info=camera_info, camera_id=cid)
        cr.start()
        cr_interfaces[cid] = cr

    controller_type = args.controller_type
    controller_cfg  = YamlConfig(os.path.join(config_root, args.controller_cfg)).as_easydict()
    args.folder.mkdir(parents=True, exist_ok=True)

    # ── World model (shared) ─────────────────────────────────────────────────
    nh = args.num_hist if args.num_hist is not None else 1
    wm_model = load_world_model(
        args.wm_checkpoint, wm_cameras_list,
        wm_arch=args.wm_arch, num_hist=nh, num_pred=args.num_pred,
        frameskip=args.frameskip, device=device,
        use_compile=not args.no_compile,
    )
    logger.info("WM loaded: arch=%s  cameras=%s", args.wm_arch, wm_cameras_list)

    ns_path = args.norm_stats_json
    if ns_path is None:
        cand = os.path.join(os.path.dirname(os.path.abspath(args.wm_checkpoint)), "norm_stats.json")
        if os.path.isfile(cand):
            ns_path = cand
    norm_stats = NormStats(ns_path, device=str(device))

    # ── Tactile encoder + CLIP ───────────────────────────────────────────────
    tactile_encoder, sensor_id = load_tactile_encoder(args.anytouch_checkpoint, device)
    clip_model, tokenizer = load_clip_for_reward(args.anytouch_checkpoint, device)

    tok_out = tokenizer([args.reward_text], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = F.normalize(clip_model.get_text_features(**tok_out), dim=-1)
    logger.info("Tactile CLIP reward text (phase 1): '%s'", args.reward_text)

    # Phase-2 text embedding for tactile reward
    text_embeds_phase2 = None
    if args.reward_text_phase2:
        tok_out2 = tokenizer([args.reward_text_phase2], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeds_phase2 = F.normalize(clip_model.get_text_features(**tok_out2), dim=-1)
        logger.info("Tactile CLIP reward text (phase 2): '%s'", args.reward_text_phase2)

    # ── DINOv3 for RGB camera embeddings ────────────────────────────────────
    vision_encoder = None
    rgb_cams = [c for c in wm_cameras_list if c in ("camera_0", "camera_1")]
    if rgb_cams:
        vision_encoder = load_vision_encoder(args.dinov3_hub_dir, args.dinov3_weights, device)
        logger.info("DINOv3 encoder loaded for %s", rgb_cams)

    # ── VQVAE decoder for vision reward camera (camera_0) ───────────────────
    vis_cam_dim = CAMERA_EMB_DIMS.get(args.vision_reward_camera, 768)
    vision_decoder = load_vqvae_decoder(args.decoder_camera1, vis_cam_dim, device)
    logger.info("VQVAE decoder loaded (emb_dim=%d) for %s", vis_cam_dim, args.vision_reward_camera)
    logger.info("Robometer server: %s  task: '%s'", args.eval_server_url, args.reward_task)

    wm_h_ctx = _history_maxlen(wm_model, args.temporal_mode)

    # init_joints = [-0.14091554, 0.02724072, 0.33824686, -2.48239114, -0.12527768, 2.53220921, -2.87378524]
    init_joints = [-0.14091554, 0.02724072, 0.33824686, -2.48239114, -0.12527768, 2.53220921, -2.87378524]

    while True:
        data: dict = {
            "actions": [], "ee_states": [], "joint_states": [],
            "gripper_states": [], "proprio": [],
            "tactile_rewards_log": [], "vision_rewards_log": [],
            "action_chunks_log": [], "decision_ee_xyz": [], "policy_query_frame_idx": [],
            "phase_log": [],
        }
        # Store full-resolution images for video logging
        for cid in camera_ids:
            data[f"camera_{cid}"] = []

        i = 0
        action_chunk_queue  = []
        tactile_buffer      = deque(maxlen=4)
        emb_histories: dict[str, deque] = {cam: deque(maxlen=wm_h_ctx) for cam in wm_cameras_list}

        robometer_session_id:  Optional[str]   = None
        robometer_ref_shape:   Optional[tuple] = None
        episode_tactile_bg = None

        # Phase state (mirrored across vision and tactile rewards)
        current_phase = 1
        current_task  = args.reward_task
        text_embeds_active     = text_embeds
        text_embeds_neg_active = text_embeds_phase2 if (args.mixed_reward and text_embeds_phase2 is not None) else None
        start   = False
        t_count = 0

        reset_joints_to(robot_interface, init_joints)
        time.sleep(1)
        print("Beginning trajectory")
        warmup_step = 0
        interrupted = False

        try:
            while True:
                last_state   = robot_interface._state_buffer[-1]
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

                # Store full-res for video; use resized for policy obs
                for cid in camera_ids:
                    raw = cr_interfaces[cid].get_img()
                    if cid in (0, 1):
                        color_img = raw["color"]
                        obs_img   = cv2.resize(color_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    else:  # tactile
                        color_img = raw["tactile"]
                        obs_img   = color_img
                    data[f"camera_{cid}"].append(color_img)
                    obs_dict[f"camera_{cid}"] = obs_img
                    if cid == 2:
                        tactile_buffer.append(obs_img.copy())

                if episode_tactile_bg is None and obs_dict.get("camera_2") is not None:
                    episode_tactile_bg = np.asarray(obs_dict["camera_2"]).copy()

                # ── Build embedding histories ────────────────────────────────
                current_tactile_emb = None
                if "camera_2" in emb_histories and tactile_buffer:
                    current_tactile_emb = get_tactile_embedding(
                        tactile_encoder, tactile_buffer, sensor_id, device,
                        use_amp=args.amp, bg=episode_tactile_bg,
                    )
                    emb_histories["camera_2"].append(current_tactile_emb.detach())

                if vision_encoder is not None:
                    for cam_name in rgb_cams:
                        cid_int = int(cam_name.split("_")[1])
                        v_emb = get_vision_embedding(vision_encoder, obs_dict[cam_name], cid_int, device)
                        emb_histories[cam_name].append(v_emb.detach())

                # ── Policy query ─────────────────────────────────────────────
                if not action_chunk_queue:
                    # Lookahead support: get multi-chunk samples when requested
                    if args.lookahead_steps > 1 and hasattr(policy, "agent"):
                        lookahead_chunks = get_action_samples_for_lookahead(
                            policy, obs_dict, args.n_samples, args.lookahead_steps,
                            binary_gripper=False,
                        )
                        action_chunks    = lookahead_chunks[0]
                        lookahead_tensors = [
                            torch.tensor(c, dtype=torch.float32, device=device)
                            for c in lookahead_chunks
                        ]
                    else:
                        action_chunks     = policy.get_action_samples(obs_dict, args.n_samples, binary_gripper=False)
                        lookahead_tensors = None

                    state_vec = obs_dict["state"].astype(np.float32)

                    data["policy_query_frame_idx"].append(int(len(data["camera_0"]) - 1))
                    data["decision_ee_xyz"].append(np.asarray(obs_dict["state"][:3], dtype=np.float32))
                    data["action_chunks_log"].append(np.asarray(action_chunks, dtype=np.float32))

                    # Single WM forward pass → predicted embeddings for all heads
                    t0 = time.time()
                    if lookahead_tensors is not None:
                        preds_last = predict_all_cameras_lookahead(
                            wm_model, emb_histories, wm_cameras_list,
                            state_vec, lookahead_tensors, norm_stats, device,
                            use_amp=args.amp, temporal_mode=args.temporal_mode,
                            reward_camera=args.vision_reward_camera,
                        )
                        # For lookahead we only get the reward camera; re-run for tactile if needed
                        if args.tactile_reward_camera not in preds_last:
                            preds_tactile = predict_all_cameras(
                                wm_model, emb_histories, wm_cameras_list,
                                state_vec, action_chunks, norm_stats, device,
                                use_amp=args.amp, temporal_mode=args.temporal_mode,
                            )
                            preds_last[args.tactile_reward_camera] = preds_tactile[args.tactile_reward_camera]
                    else:
                        preds_last = predict_all_cameras(
                            wm_model, emb_histories, wm_cameras_list,
                            state_vec, action_chunks, norm_stats, device,
                            use_amp=args.amp, temporal_mode=args.temporal_mode,
                        )
                    logger.info("WM forward: %.3fs", time.time() - t0)

                    # ── Tactile reward (CLIP cosine, phase-aware, optionally contrastive)
                    pred_tactile_emb = preds_last[args.tactile_reward_camera]  # (N, P, 512)
                    tactile_rewards  = compute_tactile_rewards(
                        pred_tactile_emb, text_embeds_active, text_embeds_neg_active,
                    )
                    logger.info(
                        "Tactile rewards (phase %d): %s  std=%.5f",
                        current_phase,
                        np.round(tactile_rewards, 4).tolist(), float(np.std(tactile_rewards)),
                    )
                    data["tactile_rewards_log"].append(tactile_rewards.copy())

                    # ── Vision reward (phased robometer on camera_0) ──────────
                    real_vis_frame      = obs_dict[args.vision_reward_camera].copy()
                    robometer_ref_shape = real_vis_frame.shape[:2]

                    # Create session after warmup, using current phase task
                    if warmup_step > 3 and robometer_session_id is None:
                        try:
                            robometer_session_id = stream_create(
                                args.eval_server_url, current_task,
                                timeout_s=args.robometer_timeout_s,
                            )
                            logger.info(
                                "Robometer session created (phase %d, task='%s'): %s",
                                current_phase, current_task, robometer_session_id,
                            )
                        except Exception as e:
                            logger.warning("Could not create robometer session: %s", e)

                    if warmup_step > 3 and robometer_session_id is not None:
                        try:
                            _hist_r, _ = stream_append_frame(
                                args.eval_server_url, robometer_session_id, real_vis_frame,
                                timeout_s=args.robometer_timeout_s,
                            )
                            logger.info("Robometer frame appended, progress=%.4f", _hist_r)
                        except Exception as e:
                            logger.warning("stream_append_frame failed: %s", e)

                    pred_vis_emb     = preds_last[args.vision_reward_camera]   # (N, P, 768)
                    t0 = time.time()
                    predicted_frames = decode_vision_frames(vision_decoder, pred_vis_emb, device)
                    logger.info("Decode: %.3fs", time.time() - t0)

                    if warmup_step > 3 and robometer_session_id is not None:
                        t0 = time.time()
                        vision_rewards = compute_vision_rewards(
                            args.eval_server_url, robometer_session_id,
                            predicted_frames, robometer_ref_shape,
                            timeout_s=args.robometer_timeout_s,
                        )
                        logger.info(
                            "Robometer (phase %d, %.3fs): %s  std=%.5f",
                            current_phase, time.time() - t0,
                            np.round(vision_rewards, 4).tolist(), float(np.std(vision_rewards)),
                        )

                        # ── Phase tracking (from run_dp_tactile_sampling_vision_robometer_log) ──
                        if np.max(vision_rewards) > args.phase_threshold and not start:
                            start = True
                            text_embeds_active = text_embeds_phase2
                            if args.mixed_reward:
                                text_embeds_neg_active = text_embeds  # flip contrastive
                            input("Press Enter to continue...")

                        if start:
                            t_count += 1

                        if (args.separate_phase
                                and current_phase == 1
                                and (args.reward_task_phase2 or args.reward_text_phase2)
                                and t_count > 4):
                            logger.info(
                                "Phase 1 max reward %.4f > threshold %.4f — "
                                "switching to phase 2 (vision task: '%s', tactile text: '%s')",
                                float(np.max(vision_rewards)), args.phase_threshold,
                                args.reward_task_phase2, args.reward_text_phase2,
                            )
                            input("Press Enter to continue...")
                            # Delete phase-1 robometer session
                            try:
                                stream_delete(
                                    args.eval_server_url, robometer_session_id,
                                    timeout_s=args.robometer_timeout_s,
                                )
                                logger.info("Phase-1 session deleted: %s", robometer_session_id)
                            except Exception as e:
                                logger.warning("Could not delete phase-1 session: %s", e)
                            robometer_session_id = None

                            # Switch both vision task and tactile text to phase 2
                            current_phase = 2
                            current_task  = args.reward_task_phase2

                            # Create phase-2 robometer session
                            try:
                                robometer_session_id = stream_create(
                                    args.eval_server_url, current_task,
                                    timeout_s=args.robometer_timeout_s,
                                )
                                stream_append_frame(
                                    args.eval_server_url, robometer_session_id,
                                    real_vis_frame, timeout_s=args.robometer_timeout_s,
                                )
                                logger.info("Phase-2 robometer session created: %s", robometer_session_id)
                            except Exception as e:
                                logger.warning("Could not create phase-2 session: %s", e)
                                robometer_session_id = None
                    else:
                        vision_rewards = np.zeros(args.n_samples, dtype=np.float32)

                    data["vision_rewards_log"].append(vision_rewards.copy())
                    data["phase_log"].append(current_phase)

                    # ── Combined selection (normalized vision + tactile) ──────
                    selected, combined_rewards = select_action_combined(
                        action_chunks, vision_rewards, tactile_rewards,
                        mode=args.action_select, temperature=args.softmax_temp,
                    )
                    print(
                        f"[query] phase={current_phase}  "
                        f"vision_rewards={np.round(vision_rewards, 4).tolist()}  "
                        f"tactile_rewards={np.round(tactile_rewards, 4).tolist()}  "
                        f"combined={np.round(combined_rewards, 4).tolist()}  "
                        f"best_idx={int(np.argmax(combined_rewards))}"
                    )
                    action_chunk_queue.extend(selected)

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
                    action[3:6] = 0.0

                print(f"Step {i}  action={np.round(action, 4).tolist()}")
                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )

                if i > 1000:
                    break

        except KeyboardInterrupt:
            print("\nKeyboard interrupt — saving trajectory …")
            interrupted = True

        if robometer_session_id is not None:
            try:
                stream_delete(args.eval_server_url, robometer_session_id,
                              timeout_s=args.robometer_timeout_s)
                logger.info("Robometer session deleted: %s", robometer_session_id)
            except Exception as e:
                logger.warning("Could not delete robometer session: %s", e)
            robometer_session_id = None

        reset_joints_to(robot_interface, init_joints)
        time.sleep(1)

        label = input("Label (0=left success, 1=left failure, 2=right success, 3=right failure): ")
        episode_id = len(list(Path(args.folder).glob("traj_*.hdf5")))

        # ── Save per-camera videos (like run_dp_model_tactile_liquid_log.py) ─
        episode_video_folder = args.folder / "videos" / f"episode_{episode_id:04d}"
        episode_video_folder.mkdir(parents=True, exist_ok=True)
        for cid in camera_ids:
            frames = data[f"camera_{cid}"]
            if frames:
                video_path = episode_video_folder / f"camera_{cid}.mp4"
                imageio.mimsave(str(video_path), frames, fps=10)
        print(f"Videos saved to {episode_video_folder}")

        # ── Save HDF5 (camera frames replaced by video_path) ─────────────────
        with h5py.File(args.folder / f"traj_{episode_id:04d}.hdf5", "w") as f:
            grp = f.create_group("data")
            grp.attrs["label"]       = label
            grp.attrs["reward_task"] = args.reward_task
            grp.create_dataset("video_path",
                               data=np.bytes_(episode_video_folder.as_posix()))

            skip = {
                "tactile_rewards_log", "vision_rewards_log",
                "action_chunks_log", "decision_ee_xyz", "policy_query_frame_idx",
                "proprio", "phase_log",
                "camera_0", "camera_1", "camera_2",
            }
            for key, value in data.items():
                if key in skip or not value:
                    continue
                grp.create_dataset(key, data=value)

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
            if data["tactile_rewards_log"]:
                grp.create_dataset("tactile_rewards_log",
                                   data=np.stack(data["tactile_rewards_log"], axis=0))
            if data["vision_rewards_log"]:
                grp.create_dataset("vision_rewards_log",
                                   data=np.stack(data["vision_rewards_log"], axis=0))
            if data.get("phase_log"):
                grp.create_dataset("phase_log",
                                   data=np.asarray(data["phase_log"], dtype=np.int32))
            if args.separate_phase:
                grp.attrs["reward_task_phase2"] = args.reward_task_phase2
                grp.attrs["reward_text_phase2"] = args.reward_text_phase2
                grp.attrs["phase_threshold"]    = args.phase_threshold

        print(f"Saved traj_{episode_id:04d}.hdf5  (label={label})")

        if interrupted:
            robot_interface.close()
            break


if __name__ == "__main__":
    main()
