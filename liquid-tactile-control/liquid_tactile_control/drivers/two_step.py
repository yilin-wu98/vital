# Publication cleanup (2026-10-27): adapted from deoxys_control/deoxys/examples/run_dp_two_step_visual_tactile_guidance_liquid_log.py.
"""
run_dp_two_step_visual_tactile_guidance.py

For each policy query:
1. Sample N action sequence pairs (chunk1, chunk2) from the diffusion policy
   by splitting the full ac_chunk prediction at ACT_HORIZON.
2. Two-step WM forward prediction (camera_0 + camera_2 heads):
     Step 1: predict embeddings at end of chunk1 using current obs history.
     Step 2: shift history window with step-1 predictions, predict embeddings
             at end of chunk2.
3. Decode the step-2 camera_0 prediction → RGB → robometer streaming reward
   → select the sample with the highest reward.
4. Re-noise the selected chunk1 for `renoise_steps` DDIM forward-diffusion
   steps to get a partially noised x_t, then run `renoise_steps` guided
   denoising steps with tactile CLIP reward gradient (camera_2 head) to
   produce the final action chunk.
5. Execute the guided chunk1.

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
    parser.add_argument("--use_tactile", action="store_true", default=False)

    # Shared multi-head world model
    parser.add_argument("--wm_checkpoint", type=str, required=True,
                        help="Multi-head WM checkpoint predicting camera_0 and camera_2")
    parser.add_argument("--wm_cameras", type=str, default="camera_0,camera_1,camera_2",
                        help="Cameras the WM was trained on (comma-separated)")
    parser.add_argument("--norm_stats_json", type=str, default=None)
    parser.add_argument("--wm_arch", type=str, choices=["video", "k_step"], default="k_step")
    parser.add_argument("--temporal_mode", type=str, choices=["lpb", "consecutive"], default="lpb")
    parser.add_argument("--num_hist", type=int, default=None)
    parser.add_argument("--num_pred", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=8)

    # Camera heads
    parser.add_argument("--vision_reward_camera", type=str, default="camera_0",
                        help="WM head decoded for robometer visual reward")
    parser.add_argument("--tactile_reward_camera", type=str, default="camera_2",
                        help="WM head used for tactile CLIP guidance reward")

    # Tactile encoder + CLIP (for guidance reward)
    parser.add_argument("--anytouch_checkpoint", type=str, required=True)
    parser.add_argument("--reward_text", type=str, default="wiping the board",
                        help="Tactile CLIP text for phase 1")
    parser.add_argument("--reward_text_phase2", type=str, default="",
                        help="Tactile CLIP text for phase 2 (optional)")

    # DINOv3 (for RGB camera embeddings)
    parser.add_argument("--dinov3_hub_dir", type=str,
                        default=None)
    parser.add_argument("--dinov3_weights", type=str,
                        default=None)

    # VQVAE decoder + robometer
    parser.add_argument("--decoder_camera0", type=str, required=True,
                        help="VQVAE decoder checkpoint for vision_reward_camera (768-dim)")
    parser.add_argument("--eval_server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--reward_task", type=str, default="",
                        help="Task description for robometer phase 1 (required)")
    parser.add_argument("--robometer_timeout_s", type=float, default=30.0)

    # Two-phase reward switching
    parser.add_argument("--separate_phase", action="store_true", default=False,
                        help="Enable two-phase reward: switch task text when reward exceeds threshold")
    parser.add_argument("--mixed_reward", action="store_true", default=False,
                        help="Use a contrastive tactile reward: r(pos) - r(neg). "
                             "Phase 1: pos=--reward_text, neg=--reward_text_phase2. "
                             "Phase 2 (after phase switch): flipped. "
                             "Requires --reward_text_phase2.")
    parser.add_argument("--reward_task_phase2", type=str, default="",
                        help="Robometer task description for phase 2")
    parser.add_argument("--phase_threshold", type=float, default=0.64,
                        help="Vision reward threshold to trigger phase 1 → phase 2 switch")

    # Partial re-noise + guided denoising
    parser.add_argument("--renoise_steps", type=int, default=4,
                        help="Number of DDIM steps to add noise (and then denoise with guidance)")
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--ss", type=int, default=4,
                        help="Stochastic-sampling repetitions per guided denoising step")
    parser.add_argument("--guidance_min_alpha", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--normalize_grad", action="store_true")
    parser.add_argument("--lpb_scale", action="store_true")

    # Robometer
    parser.add_argument("--server_max_frames", type=int, default=32)

    # Misc
    parser.add_argument("--no_compile", action="store_true", default=True,
                        help="Disable torch.compile on the WM (default: True — eager is "
                             "required for autograd-based guidance; pass --compile_wm to override)")
    parser.add_argument("--compile_wm", action="store_true",
                        help="Enable torch.compile(reduce-overhead) on the WM. "
                             "Safe because stop-gradient eliminates the UNet backward.")
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
    return int(getattr(m, "num_hist")) if _is_k_step(m) else int(m.temp_embedding.shape[1])


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
    return torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).contiguous().float().mul_(1.0 / 255.0)


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
        logger.info("torch.compile applied (arch=%s)", wm_arch)
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
    return F.normalize(out[:, 6:, :], dim=-1)  # (1, P, 512)


def get_vision_embedding(encoder, img_np, camera_id, device):
    img_pil = PILImage.fromarray(np.uint8(img_np)).convert("RGB")
    transform = _DINO_TRANSFORM_CAM0 if camera_id == 0 else _DINO_TRANSFORM_CAM1
    img_t = transform(img_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        return encoder.forward_features(img_t)["x_norm_patchtokens"]  # (1, 196, 768)


# ─────────────────────────────────────────────────────────────────────────────
# WM context helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_wm_context(history, h_ctx, device, fallback=None):
    """Stack embedding history into (1, h_ctx, P, D), left-padding if short."""
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
    if inp.shape[2] != PATCH_SIDE_DINOV3 ** 2:
        inp = ensure_patch_grid(inp, PATCH_SIDE_DINOV3, device)
    return inp


def _k_step_collapsed_actions(ac, num_hist, fs, temporal_mode, chunk_T, device):
    """(N, chunk_T, 7) → (N, num_hist, fs*7) collapsed at rollout_step=0."""
    N = ac.shape[0]
    out = torch.zeros((N, num_hist, fs * 7), device=device, dtype=torch.float32)
    for j in range(num_hist):
        i0 = (-(num_hist - 1 - j) * fs) if temporal_mode == "lpb" else (j - num_hist + 1)
        i0 = max(0, min(i0, chunk_T - 1))
        for k in range(fs):
            idx = min(i0 + k, chunk_T - 1)
            out[:, j, k * 7:(k + 1) * 7] = ac[:, idx, :]
    return out


def _normalize_collapsed(acs, norm_stats):
    shape = acs.shape
    return norm_stats.normalize_acs(acs.reshape(-1, 7)).view(shape)


def _video_action_windows(ac, steps, h_ctx, device):
    N, T, _ = ac.shape
    def window(s):
        idxs = [min(s + j, T - 1) for j in range(h_ctx)]
        return torch.cat([ac[:, i:i+1] for i in idxs], dim=1).to(device=device, dtype=torch.float32)
    return torch.stack([window(s) for s in range(steps)], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Two-step WM forward prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_two_steps(
    wm_model,
    emb_histories,       # dict: cam_name -> deque of (1, P, D)
    wm_cameras,
    state,               # (8,) float32 numpy
    chunk1_actions,      # (N, ACT_HORIZON, 7) torch tensor on device
    chunk2_actions,      # (N, ACT_HORIZON, 7) torch tensor on device
    norm_stats,
    device,
    use_amp,
    temporal_mode,
):
    """
    Run the WM twice:
      Step 1 — current history + chunk1 → predicted embeddings at end of chunk1
      Step 2 — shift context (insert step-1 predictions) + chunk2 → predicted
               embeddings at end of chunk2

    Returns:
        preds_step1: dict {cam: (N, P, D)} — predictions after chunk1
        preds_step2: dict {cam: (N, P, D)} — predictions after chunk2
    """
    N = chunk1_actions.shape[0]
    chunk_T = int(chunk1_actions.shape[1])
    autocast_ctx = torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda"))

    if _is_k_step(wm_model):
        num_hist = int(wm_model.num_hist)
        fs = int(wm_model.frameskip)

        # Build step-1 inputs
        inputs_1 = {}
        for cam in wm_cameras:
            hist = emb_histories.get(cam, deque())
            fallback = list(hist)[-1] if hist else None
            inputs_1[cam] = build_wm_context(hist, num_hist, device, fallback).expand(N, -1, -1, -1).contiguous()
        st_1 = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, 8).expand(N, num_hist, 8).contiguous()

        acs_1 = _normalize_collapsed(
            _k_step_collapsed_actions(chunk1_actions, num_hist, fs, temporal_mode, chunk_T, device),
            norm_stats,
        )
        with torch.inference_mode():
            with autocast_ctx:
                preds_1, pred_state_1 = wm_model(inputs_1, st_1, acs_1)
                # Clone immediately — torch.compile CUDA graphs reuse output buffers
                preds_1 = {cam: t.clone() for cam, t in preds_1.items()}
                pred_state_1 = pred_state_1.clone()

        # Build step-2 inputs: shift history left by 1, append step-1 prediction
        inputs_2 = {}
        for cam in wm_cameras:
            # preds_1[cam]: (N, num_pred, P, D) — take last pred slot
            new_frame = preds_1[cam][:, -1:, :, :]          # (N, 1, P, D)
            inputs_2[cam] = torch.cat([inputs_1[cam][:, 1:], new_frame], dim=1).contiguous()
        # Shift state context too
        st_2 = torch.cat([st_1[:, 1:], pred_state_1[:, -1:]], dim=1).contiguous()

        acs_2 = _normalize_collapsed(
            _k_step_collapsed_actions(chunk2_actions, num_hist, fs, temporal_mode, chunk_T, device),
            norm_stats,
        )
        with torch.inference_mode():
            with autocast_ctx:
                preds_2, _ = wm_model(inputs_2, st_2, acs_2)

    else:  # video WM — autoregressive
        h_ctx = _ctx_len(wm_model)

        def _build_inputs(h_ctx):
            out = {}
            for cam in wm_cameras:
                hist = emb_histories.get(cam, deque())
                fallback = list(hist)[-1] if hist else None
                out[cam] = build_wm_context(hist, h_ctx, device, fallback).expand(N, -1, -1, -1).contiguous()
            return out

        inputs_1 = _build_inputs(h_ctx)
        st_1 = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, 8).expand(N, h_ctx, 8).contiguous()
        acs_all_1 = norm_stats.normalize_acs(_video_action_windows(chunk1_actions, ACT_HORIZON, h_ctx, device))

        inputs_cur = {cam: v.clone() for cam, v in inputs_1.items()}
        st_cur = st_1.clone()
        with torch.inference_mode():
            for step in range(ACT_HORIZON):
                with autocast_ctx:
                    preds_1, pred_state_1 = wm_model(inputs_cur, st_cur, acs_all_1[step])
                for cam in wm_cameras:
                    if cam in preds_1:
                        inputs_cur[cam] = torch.cat([inputs_cur[cam][:, 1:], preds_1[cam][:, -1:]], dim=1)
                st_cur = torch.cat([st_cur[:, 1:], pred_state_1[:, -1:]], dim=1)

        # Step 2 starts from where step 1 left off
        acs_all_2 = norm_stats.normalize_acs(_video_action_windows(chunk2_actions, ACT_HORIZON, h_ctx, device))
        with torch.inference_mode():
            for step in range(ACT_HORIZON):
                with autocast_ctx:
                    preds_2, pred_state_2 = wm_model(inputs_cur, st_cur, acs_all_2[step])
                for cam in wm_cameras:
                    if cam in preds_2:
                        inputs_cur[cam] = torch.cat([inputs_cur[cam][:, 1:], preds_2[cam][:, -1:]], dim=1)
                st_cur = torch.cat([st_cur[:, 1:], pred_state_2[:, -1:]], dim=1)

    # Return per-sample predictions at the last temporal slot of each step
    return (
        {cam: preds_1[cam][:, -1, :, :].clone() for cam in preds_1},
        {cam: preds_2[cam][:, -1, :, :].clone() for cam in preds_2},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visual reward via robometer
# ─────────────────────────────────────────────────────────────────────────────

def decode_vision_frames(decoder, pred_emb, device):
    """(N, P, 768) → (N, H, W, 3) uint8"""
    with torch.no_grad():
        pred_img, _ = decoder(pred_emb.unsqueeze(1).to(device))
    images = []
    for img in pred_img.cpu().numpy():
        images.append((np.clip(img.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8))
    return np.stack(images, axis=0)


def compute_vision_rewards(eval_server_url, session_id, predicted_frames,
                           ref_shape, timeout_s=30.0):
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
        logger.warning("Robometer evaluate failed: %s — returning zero rewards", e)
        return np.zeros(N, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Tactile guidance reward function (differentiable, uses camera_2 WM head)
# ─────────────────────────────────────────────────────────────────────────────

def make_tactile_reward_fn(
    wm_model,
    emb_histories,
    wm_cameras,
    state,
    norm_stats,
    text_embeds,
    ac_min_t,      # torch tensor (7,) on device
    ac_max_t,      # torch tensor (7,) on device
    device,
    temporal_mode,
    tactile_reward_camera="camera_2",
    text_embeds_neg=None,
):
    """
    Build a differentiable reward closure that uses the camera_2 head of the
    shared WM.  All context tensors are snapshotted and detached at factory time.

    Supports k_step WM only (the video case is the same structure but with
    a single-frame window h_ctx=1 context for simplicity).
    """
    # The compiled model (reduce-overhead) records its first CUDA graph inside
    # torch.inference_mode() from predict_two_steps.  Calling the same compiled
    # model with grad enabled triggers a second CUDA graph recording that fails
    # because model buffers were captured as inference tensors in the first graph.
    # Solution: use the original uncompiled module for gradient computation.
    # _orig_mod is set by torch.compile; falls back to wm_model if not compiled.
    wm_model = getattr(wm_model, "_orig_mod", wm_model)

    if _is_k_step(wm_model):
        num_hist = int(wm_model.num_hist)
        fs = int(wm_model.frameskip)

        inputs_snap = {}
        for cam in wm_cameras:
            hist = emb_histories.get(cam, deque())
            fallback = list(hist)[-1] if hist else None
            # .clone() outside inference_mode produces a normal (non-inference) tensor,
            # which is required for torch.compile + CUDA graph capture with autograd.
            inputs_snap[cam] = build_wm_context(hist, num_hist, device, fallback).clone()
        st_snap = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, 8).expand(1, num_hist, 8).clone()
    else:
        h_ctx = _ctx_len(wm_model)
        inputs_snap = {}
        for cam in wm_cameras:
            hist = emb_histories.get(cam, deque())
            fallback = list(hist)[-1] if hist else None
            inputs_snap[cam] = build_wm_context(hist, h_ctx, device, fallback).clone()
        st_snap = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, 8).expand(1, h_ctx, 8).clone()
        num_hist, fs = h_ctx, 1

    text_emb = text_embeds.clone()
    text_emb_neg = text_embeds_neg.clone() if text_embeds_neg is not None else None

    def reward_fn(x0_pred):
        # x0_pred: (1, ac_chunk, 7) in [-1, 1] — has grad
        ac_chunk = x0_pred.shape[1]
        ac_raw = (x0_pred + 1.0) / 2.0 * (ac_max_t - ac_min_t) + ac_min_t

        if _is_k_step(wm_model):
            slots = []
            for j in range(num_hist):
                i0 = (-(num_hist - 1 - j) * fs) if temporal_mode == "lpb" else (j - num_hist + 1)
                i0 = max(0, min(i0, ac_chunk - 1))
                frames = [ac_raw[:, min(i0 + k, ac_chunk - 1), :] for k in range(fs)]
                slots.append(torch.cat(frames, dim=-1).unsqueeze(1))
            acs = torch.cat(slots, dim=1)  # (1, num_hist, fs*7)
            acs_n = _normalize_collapsed(acs, norm_stats)
            preds, _ = wm_model(inputs_snap, st_snap, acs_n)
        else:
            ac_win = torch.cat([ac_raw[:, min(j, ac_chunk - 1):min(j, ac_chunk - 1)+1] for j in range(num_hist)], dim=1)
            acs_n = norm_stats.normalize_acs(ac_win)
            preds, _ = wm_model(inputs_snap, st_snap, acs_n)

        pred_e = preds[tactile_reward_camera][:, -1, :, :]  # (1, P, D)
        emb = F.normalize(pred_e.mean(dim=1), dim=-1)
        reward = (emb @ text_emb.T)[:, 0]
        if text_emb_neg is not None:
            reward = reward - (emb @ text_emb_neg.T)[:, 0]
        return reward

    return reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# Partial re-noise + guided denoising
# ─────────────────────────────────────────────────────────────────────────────

def apply_partial_guided_denoise(
    policy,
    obs,
    x0_selected_norm,   # (1, ACT_HORIZON, 7) in policy [-1,1] space, on device
    reward_fn,
    renoise_steps=4,
    guidance_scale=1.5,
    ss=4,
    guidance_min_alpha=0.0,
    grad_clip=None,
    normalize_grad=False,
    lpb_scale=False,
):
    """
    1. Add Gaussian noise to x0_selected_norm at the noise level corresponding
       to the (renoise_steps)-th-from-last DDIM timestep.
    2. Run renoise_steps guided DDIM denoising steps with reward_fn.
    3. Return the final denoised action (1, ACT_HORIZON, 7) in policy [-1,1] space.
    """
    agent = policy.agent
    images, state = policy._get_images_and_states(obs)
    device = x0_selected_norm.device
    B = 1

    # Obs encoding (no grad, stable FP32)
    with torch.no_grad():
        s_t = agent._shared_forward(images, state)

    # Set up the full DDIM schedule
    eval_steps = agent._eval_diffusion_steps
    agent.diffusion_schedule.set_timesteps(eval_steps)
    agent.diffusion_schedule.alphas_cumprod = agent.diffusion_schedule.alphas_cumprod.to(device)
    alphas_cumprod = agent.diffusion_schedule.alphas_cumprod
    all_timesteps = list(agent.diffusion_schedule.timesteps)  # high→low noise

    # The last renoise_steps timesteps of the denoising schedule = lowest noise levels
    renoise_steps = min(renoise_steps, eval_steps)
    partial_timesteps = all_timesteps[-renoise_steps:]  # still ordered high→low within window

    # Forward diffusion: map x0 → x_t at the noise level of the first partial timestep
    t_start = partial_timesteps[0]
    alpha_bar_start = alphas_cumprod[t_start].to(device)
    noise = torch.randn_like(x0_selected_norm)
    # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
    x_t = alpha_bar_start.sqrt() * x0_selected_norm + (1.0 - alpha_bar_start).sqrt() * noise

    logger.info(
        "Partial re-noise: t_start=%d  alpha_bar=%.4f  x_t_norm=%.4f",
        t_start.item(), alpha_bar_start.item(), x_t.norm().item(),
    )

    # Per-step guidance logs: reward and x_t snapshot after each denoising step
    step_rewards: list[float] = []
    step_xt_norm: list[np.ndarray] = []  # (ACT_HORIZON, 7) in [-1,1] after each step

    # Guided denoising for renoise_steps
    for timestep in partial_timesteps:
        batched_ts = timestep.unsqueeze(0).repeat(B).to(device)
        alpha_prod_t = alphas_cumprod[timestep]
        sqrt_one_minus = (1.0 - alpha_prod_t).sqrt()

        alpha_ok = alpha_prod_t.item() >= guidance_min_alpha
        inner_iters = ss if alpha_ok else 1

        step_reward = float("nan")

        for _ in range(inner_iters):
            # Stop-gradient through noise_net
            with torch.no_grad():
                noise_pred = agent.noise_net(x_t, batched_ts, s_t)

            if alpha_ok:
                with torch.no_grad():
                    x0_raw = ((x_t - sqrt_one_minus * noise_pred) / alpha_prod_t.sqrt()).clamp(-1.0, 1.0)

                x0_pred = x0_raw.detach().requires_grad_(True)
                reward = reward_fn(x0_pred)
                step_reward = reward.item()

                (grad_x0,) = torch.autograd.grad(reward.sum(), x0_pred, allow_unused=True)

                if grad_x0 is not None:
                    if grad_clip is not None:
                        g_norm = grad_x0.norm()
                        if g_norm > grad_clip:
                            grad_x0 = grad_x0 * (grad_clip / g_norm.clamp(min=1e-8))
                    if normalize_grad:
                        grad_x0 = grad_x0 / (grad_x0.norm() + 1e-8)
                    scale_factor = sqrt_one_minus if lpb_scale else sqrt_one_minus / alpha_prod_t.sqrt()
                    noise_pred_guided = noise_pred - scale_factor * guidance_scale * grad_x0
                else:
                    noise_pred_guided = noise_pred
            else:
                noise_pred_guided = noise_pred

            with torch.no_grad():
                x_t = agent.diffusion_schedule.step(
                    model_output=noise_pred_guided,
                    timestep=timestep,
                    sample=x_t,
                ).prev_sample

        step_rewards.append(step_reward)
        step_xt_norm.append(x_t[0].detach().cpu().numpy().astype(np.float32))

    # step_logs contains per-guidance-step data for saving to dataset
    step_logs = {
        "step_rewards": np.array(step_rewards, dtype=np.float32),          # (renoise_steps,)
        "step_xt_norm": np.stack(step_xt_norm, axis=0).astype(np.float32), # (renoise_steps, ACT_HORIZON, 7)
    }

    return x_t, step_logs  # x_t: (1, ACT_HORIZON, 7) in [-1,1]


# ─────────────────────────────────────────────────────────────────────────────
# Policy helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_action_samples_two_chunks(policy, obs, N, binary_gripper=False):
    """
    Sample N full action sequences and split into two ACT_HORIZON-length chunks.

    Returns:
        chunk1: (N, ACT_HORIZON, 7) unnormalized
        chunk2: (N, ACT_HORIZON, 7) unnormalized  (indices ACT_HORIZON:2*ACT_HORIZON)
        chunk1_norm: (N, ACT_HORIZON, 7) in policy [-1,1] space (before unnorm)
    """
    images, state = policy._get_images_and_states(obs)
    batch_images = {k: v.repeat(N, *([1] * (v.dim() - 1))) for k, v in images.items()}
    batch_state  = state.repeat(N, 1)

    with torch.no_grad():
        actions_norm = policy.agent.get_actions(batch_images, batch_state)  # (N, ac_chunk, 7)

    ac_chunk = actions_norm.shape[1]
    end2 = min(2 * ACT_HORIZON, ac_chunk)

    # Normalized (policy space) chunk1 — needed for re-noising
    chunk1_norm = actions_norm[:, :ACT_HORIZON, :].clone()  # (N, ACT_HORIZON, 7) in [-1,1]

    # Unnormalized chunks
    ac = actions_norm[:, :end2, :].cpu().numpy().astype(np.float32)
    ac = (ac + 1.0) / 2.0
    ac = ac * (policy.ac_max - policy.ac_min) + policy.ac_min
    if binary_gripper:
        ac[:, :, -1] = np.sign(ac[:, :, -1])

    chunk1 = ac[:, :ACT_HORIZON, :]
    chunk2 = ac[:, ACT_HORIZON:end2, :] if end2 > ACT_HORIZON else ac[:, :ACT_HORIZON, :]

    return chunk1, chunk2, chunk1_norm


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
        raise ValueError("--reward_task is required")

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
        cr = CameraRedisSubInterface(
            camera_info=camera_info, camera_id=cid,
            **({} if "tactile" not in camera_types[cid] else {"use_color": False, "use_tactile": True}),
        )
        cr.start()
        cr_interfaces[cid] = cr

    controller_type = args.controller_type
    controller_cfg  = YamlConfig(os.path.join(config_root, args.controller_cfg)).as_easydict()
    args.folder.mkdir(parents=True, exist_ok=True)

    # ── World model ──────────────────────────────────────────────────────────
    nh = args.num_hist if args.num_hist is not None else 1
    wm_model = load_world_model(
        args.wm_checkpoint, wm_cameras_list,
        wm_arch=args.wm_arch, num_hist=nh, num_pred=args.num_pred,
        frameskip=args.frameskip, device=device,
        use_compile=args.compile_wm,
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
        text_embeds = F.normalize(clip_model.get_text_features(**tok_out), dim=-1)  # (1, D)
    logger.info("Tactile CLIP reward text (phase 1): '%s'", args.reward_text)

    text_embeds_phase2 = None
    if args.reward_text_phase2:
        tok_out2 = tokenizer([args.reward_text_phase2], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeds_phase2 = F.normalize(clip_model.get_text_features(**tok_out2), dim=-1)
        logger.info("Tactile CLIP reward text (phase 2): '%s'", args.reward_text_phase2)

    # ac_min / ac_max tensors for differentiable unnorm
    ac_min_t = torch.tensor(policy.ac_min, dtype=torch.float32, device=device)
    ac_max_t = torch.tensor(policy.ac_max, dtype=torch.float32, device=device)

    # ── DINOv3 for RGB cameras ───────────────────────────────────────────────
    rgb_cams = [c for c in wm_cameras_list if c in ("camera_0", "camera_1")]
    vision_encoder = None
    if rgb_cams:
        vision_encoder = load_vision_encoder(args.dinov3_hub_dir, args.dinov3_weights, device)
        logger.info("DINOv3 encoder loaded for %s", rgb_cams)

    # ── VQVAE decoder ───────────────────────────────────────────────────────
    vis_cam_dim = CAMERA_EMB_DIMS.get(args.vision_reward_camera, 768)
    vision_decoder = load_vqvae_decoder(args.decoder_camera0, vis_cam_dim, device)
    logger.info("VQVAE decoder loaded (emb_dim=%d) for %s", vis_cam_dim, args.vision_reward_camera)
    logger.info("Robometer: %s  task: '%s'", args.eval_server_url, args.reward_task)
    logger.info(
        "Partial guidance: renoise_steps=%d  guidance_scale=%.2f  ss=%d",
        args.renoise_steps, args.guidance_scale, args.ss,
    )

    wm_h_ctx = _history_maxlen(wm_model, args.temporal_mode)

    # init_joints = [-0.18941418455646322, 0.3986268416401311, 0.37023725867044477,
    #                -1.8986898980216536, -0.1220696096038897, 2.1729692311931763,
    #                -2.1428127053565933]
    init_joints = [-0.14091554, 0.02724072, 0.33824686, -2.48239114, -0.12527768, 2.53220921, -2.87378524]

    while True:
        data: dict = {
            "actions": [], "ee_states": [], "joint_states": [],
            "gripper_states": [], "proprio": [],
            "vision_rewards_log": [], "action_chunks_log": [],
            "decision_ee_xyz": [], "policy_query_frame_idx": [],
            "chunk_before_guidance": [],   # (query, ACT_HORIZON, 7) unnorm, selected before guidance
            "chunk_after_guidance": [],    # (query, ACT_HORIZON, 7) unnorm, after guided denoising
            "guidance_step_rewards": [],   # (query, renoise_steps) reward at each guidance step
            "guidance_step_xt": [],        # (query, renoise_steps, ACT_HORIZON, 7) norm-space x_t per step
        }
        for cid in camera_ids:
            data[f"camera_{cid}"] = []

        i = 0
        action_chunk_queue = []
        tactile_buffer     = deque(maxlen=4)
        emb_histories: dict[str, deque] = {cam: deque(maxlen=wm_h_ctx) for cam in wm_cameras_list}

        robometer_session_id:  Optional[str]   = None
        robometer_ref_shape:   Optional[tuple] = None
        episode_tactile_bg = None

        # Phase state (mirrors run_dp_tactile_vision_combined_liquid_log.py)
        current_phase       = 1
        current_task        = args.reward_task
        text_embeds_active  = text_embeds
        # Mixed reward: phase 1 → pos=text_embeds, neg=text_embeds_phase2; phase 2 flips.
        text_embeds_neg_active = text_embeds_phase2 if (args.mixed_reward and text_embeds_phase2 is not None) else None
        start               = False
        t_count             = 0

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

                for cid in camera_ids:
                    raw = cr_interfaces[cid].get_img()
                    if cid in (0, 1):
                        color_img = raw["color"]
                        obs_img   = cv2.resize(color_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    else:
                        color_img = raw["tactile"]
                        obs_img   = color_img
                    data[f"camera_{cid}"].append(color_img)   # full-res for video
                    if not args.use_tactile and cid == 2:
                        continue
                    obs_dict[f"camera_{cid}"] = obs_img        # resized for policy
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
                    state_vec = obs_dict["state"].astype(np.float32)

                    # Sample N pairs of action chunks
                    chunk1, chunk2, chunk1_norm = get_action_samples_two_chunks(
                        policy, obs_dict, args.n_samples, binary_gripper=False
                    )

                    data["policy_query_frame_idx"].append(int(len(data["camera_0"]) - 1))
                    data["decision_ee_xyz"].append(np.asarray(obs_dict["state"][:3], dtype=np.float32))
                    data["action_chunks_log"].append(np.asarray(chunk1, dtype=np.float32))

                    # ── Two-step WM forward ──────────────────────────────────
                    t0 = time.time()
                    ac1_t = torch.tensor(chunk1, dtype=torch.float32, device=device)
                    ac2_t = torch.tensor(chunk2, dtype=torch.float32, device=device)

                    preds_step1, preds_step2 = predict_two_steps(
                        wm_model, emb_histories, wm_cameras_list,
                        state_vec, ac1_t, ac2_t, norm_stats,
                        device, use_amp=args.amp, temporal_mode=args.temporal_mode,
                    )
                    logger.info("Two-step WM forward: %.3fs", time.time() - t0)

                    # ── Visual reward from step-2 camera_0 prediction ────────
                    real_vis_frame      = obs_dict[args.vision_reward_camera].copy()
                    robometer_ref_shape = real_vis_frame.shape[:2]

                    if robometer_session_id is None:
                        try:
                            robometer_session_id = stream_create(
                                args.eval_server_url, current_task,
                                timeout_s=args.robometer_timeout_s,
                                max_frames=args.server_max_frames,
                            )
                            logger.info("Robometer session created (phase %d, task='%s'): %s",
                                        current_phase, current_task, robometer_session_id)
                        except Exception as e:
                            logger.warning("Could not create robometer session: %s", e)

                    if robometer_session_id is not None:
                        try:
                            _hist_r, _ = stream_append_frame(
                                args.eval_server_url, robometer_session_id, real_vis_frame,
                                timeout_s=args.robometer_timeout_s,
                            )
                            logger.debug("Robometer frame appended, progress=%.4f", _hist_r)
                        except Exception as e:
                            logger.warning("stream_append_frame failed: %s", e)

                    t0 = time.time()
                    # Use step-2 prediction (end of 2nd chunk) as the visual reward signal
                    pred_vis_step2  = preds_step2[args.vision_reward_camera]   # (N, P, 768)
                    decoded_step2   = decode_vision_frames(vision_decoder, pred_vis_step2, device)
                    logger.info("Decode step-2: %.3fs", time.time() - t0)

                    if robometer_session_id is not None:
                        t0 = time.time()
                        vision_rewards = compute_vision_rewards(
                            args.eval_server_url, robometer_session_id,
                            decoded_step2, robometer_ref_shape,
                            timeout_s=args.robometer_timeout_s,
                        )
                        logger.info(
                            "Robometer (phase %d, %.3fs): %s  best=%d (%.4f)",
                            current_phase, time.time() - t0,
                            np.round(vision_rewards, 4).tolist(),
                            int(np.argmax(vision_rewards)), float(np.max(vision_rewards)),
                        )

                        # ── Phase switching (mirrors run_dp_tactile_vision_combined_liquid_log) ──
                        if np.max(vision_rewards) > args.phase_threshold and not start:
                            start = True
                            if text_embeds_phase2 is not None:
                                text_embeds_active = text_embeds_phase2
                            input("Press Enter to continue...")

                        if start:
                            t_count += 1

                        if (args.separate_phase
                                and current_phase == 1
                                and (args.reward_task_phase2 or args.reward_text_phase2)
                                and t_count > 4):
                            logger.info(
                                "Phase 1 max reward %.4f > threshold %.4f — switching to phase 2 "
                                "(vision task: '%s', tactile text: '%s')",
                                float(np.max(vision_rewards)), args.phase_threshold,
                                args.reward_task_phase2, args.reward_text_phase2,
                            )
                            input("Press Enter to continue...")
                            # Delete phase-1 session
                            try:
                                stream_delete(args.eval_server_url, robometer_session_id,
                                              timeout_s=args.robometer_timeout_s)
                                logger.info("Phase-1 session deleted: %s", robometer_session_id)
                            except Exception as e:
                                logger.warning("Could not delete phase-1 session: %s", e)
                            robometer_session_id = None

                            current_phase = 2
                            current_task  = args.reward_task_phase2
                            if text_embeds_phase2 is not None:
                                text_embeds_active = text_embeds_phase2
                            # Mixed reward phase 2: pos=phase2 text, neg=phase1 text
                            if args.mixed_reward:
                                text_embeds_neg_active = text_embeds

                            # Create phase-2 session
                            try:
                                robometer_session_id = stream_create(
                                    args.eval_server_url, current_task,
                                    timeout_s=args.robometer_timeout_s,
                                    max_frames=args.server_max_frames,
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

                    # ── Select best sample by visual reward ──────────────────
                    best_idx = int(np.argmax(vision_rewards))
                    logger.info("Selected sample idx=%d  vision_reward=%.4f",
                                best_idx, float(vision_rewards[best_idx]))

                    # ── Partial re-noise + tactile-guided denoising ──────────
                    # Snapshot histories BEFORE reward function factory
                    reward_fn = make_tactile_reward_fn(
                        wm_model, emb_histories, wm_cameras_list,
                        state_vec, norm_stats, text_embeds_active,
                        ac_min_t, ac_max_t, device,
                        temporal_mode=args.temporal_mode,
                        tactile_reward_camera=args.tactile_reward_camera,
                        text_embeds_neg=text_embeds_neg_active,
                    )

                    # selected chunk1 in policy [-1,1] space: (1, ACT_HORIZON, 7)
                    x0_sel = chunk1_norm[best_idx:best_idx + 1].to(device)

                    # Unnormalize selected chunk before guidance for logging
                    ac_before = x0_sel[0].detach().cpu().numpy().astype(np.float32)
                    ac_before = (ac_before + 1.0) / 2.0
                    ac_before = ac_before * (policy.ac_max - policy.ac_min) + policy.ac_min
                    # ac_before[:, -1] = np.sign(ac_before[:, -1])
                    data["chunk_before_guidance"].append(ac_before.copy())

                    t0 = time.time()
                    x0_guided_norm, guidance_step_logs = apply_partial_guided_denoise(
                        policy, obs_dict, x0_sel,
                        reward_fn,
                        renoise_steps=args.renoise_steps,
                        guidance_scale=args.guidance_scale,
                        ss=args.ss,
                        guidance_min_alpha=args.guidance_min_alpha,
                        grad_clip=args.grad_clip,
                        normalize_grad=args.normalize_grad,
                        lpb_scale=args.lpb_scale,
                    )
                    logger.info("Partial guidance: %.3fs", time.time() - t0)

                    data["guidance_step_rewards"].append(guidance_step_logs["step_rewards"].copy())
                    data["guidance_step_xt"].append(guidance_step_logs["step_xt_norm"].copy())

                    # Unnormalize guided chunk1
                    ac_guided = x0_guided_norm[0].detach().cpu().numpy().astype(np.float32)
                    ac_guided = (ac_guided + 1.0) / 2.0
                    ac_guided = ac_guided * (policy.ac_max - policy.ac_min) + policy.ac_min
                    # ac_guided[:, -1] = np.sign(ac_guided[:, -1])  # binarize gripper
                    data["chunk_after_guidance"].append(ac_guided.copy())

                    action_chunk_queue.extend([ac_guided[t] for t in range(ACT_HORIZON)])

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

                if i > 900:
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

        # ── Save per-camera videos ────────────────────────────────────────────
        episode_video_folder = args.folder / "videos" / f"episode_{episode_id:04d}"
        episode_video_folder.mkdir(parents=True, exist_ok=True)
        for cid in camera_ids:
            frames = data[f"camera_{cid}"]
            if frames:
                imageio.mimsave(str(episode_video_folder / f"camera_{cid}.mp4"), frames, fps=10)
        print(f"Videos saved → {episode_video_folder}")

        with h5py.File(args.folder / f"traj_{episode_id:04d}.hdf5", "w") as f:
            grp = f.create_group("data")
            grp.attrs["label"]         = label
            grp.attrs["reward_task"]   = args.reward_task
            grp.attrs["renoise_steps"] = args.renoise_steps
            grp.attrs["video_path"]    = episode_video_folder.as_posix()
            if args.separate_phase:
                grp.attrs["reward_task_phase2"] = args.reward_task_phase2
                grp.attrs["reward_text_phase2"] = args.reward_text_phase2
                grp.attrs["phase_threshold"]    = args.phase_threshold

            skip = {"vision_rewards_log", "action_chunks_log",
                    "decision_ee_xyz", "policy_query_frame_idx", "proprio",
                    "chunk_before_guidance", "chunk_after_guidance",
                    "guidance_step_rewards", "guidance_step_xt",
                    *(f"camera_{cid}" for cid in camera_ids)}
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
            if data["vision_rewards_log"]:
                grp.create_dataset("vision_rewards_log",
                                   data=np.stack(data["vision_rewards_log"], axis=0))
            if data["chunk_before_guidance"]:
                grp.create_dataset("chunk_before_guidance",
                                   data=np.stack(data["chunk_before_guidance"], axis=0))
            if data["chunk_after_guidance"]:
                grp.create_dataset("chunk_after_guidance",
                                   data=np.stack(data["chunk_after_guidance"], axis=0))
            if data["guidance_step_rewards"]:
                grp.create_dataset("guidance_step_rewards",
                                   data=np.stack(data["guidance_step_rewards"], axis=0))
            if data["guidance_step_xt"]:
                grp.create_dataset("guidance_step_xt",
                                   data=np.stack(data["guidance_step_xt"], axis=0))

        print(f"Saved traj_{episode_id:04d}.hdf5  (label={label})")

        if interrupted:
            robot_interface.close()
            break


if __name__ == "__main__":
    main()
