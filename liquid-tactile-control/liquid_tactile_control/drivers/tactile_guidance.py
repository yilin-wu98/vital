# Publication cleanup (2026-10-27): adapted from deoxys_control/deoxys/examples/run_model_tactile_guidance_liquid_log.py.
"""Diffusion policy with classifier-based tactile-language reward guidance.

Unlike run_dp_model_tactile_sampling.py (which samples N chunks and selects the best),
this script modifies the DDIM denoising loop itself: at each denoising step the gradient
∇_{x_t} r(x̂₀(x_t)) is computed via autograd and added to x_t, steering denoising toward
high-reward (tactile-text aligned) action trajectories.

Reward function:  r(a_{t:t+H}) = cosine_sim(WM(o_t, a_{t:t+H}), text_embed)
                  where WM predicts a future tactile embedding from the current observation
                  and the proposed action chunk.
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import h5py
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as _tv_transforms
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

# ── latent-tactile path ──────────────────────────────────────────────────────

from liquid_tactile_control.robot_config import robot_config_parse_args
from liquid_tactile_control.policy import RealPolicy

from dino_wm.dino_models import TactileVideoTransformer, NormStats
from dino_wm.dino_models_k_step import TactileKStepTransformer

# ── robometer client helpers ─────────────────────────────────────────────────

from liquid_tactile_control.robometer_client import (
    stream_create,
    stream_append_frame,
    stream_delete,
)

# ── Import shared utilities from the sampling script ─────────────────────────
from liquid_tactile_control.sampling_context import (
    ensure_patch_grid,
    _strip_compile_prefix,
    load_tactile_encoder,
    get_tactile_embedding_from_image,
    build_tactile_wm_context,
    build_tactile_k_step_context,
    build_k_step_state_tensor,
    build_k_step_inp_and_st_from_logged,
    _is_k_step_wm,
    _wm_context_length,
    _tactile_emb_history_maxlen,
    eef_pose_to_state,
    PATCH_SIDE_DINOV3,
    CAMERA_EMB_DIMS,
)
from anytouch_reward.loading import load_text_encoder as load_clip_for_reward

import logging
logger = logging.getLogger(__name__)

ACT_HORIZON = 8

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


# ─────────────────────────────────────────────────────────────────────────────
# World-model loading  (always without torch.compile to keep autograd working)
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_compile(model, use_compile: bool):
    """
    Optionally wrap model with torch.compile (mode="reduce-overhead").

    torch.compile is compatible with autograd in PyTorch ≥ 2.0, but with the
    stop-gradient optimization (Opt-1) the WM backward graph is the only one
    that runs, so compiled WM forward + backward is safe and substantially faster.

    If compilation fails at runtime (e.g. dynamic shapes, unsupported ops), fall
    back gracefully to the eager model.
    """
    if not use_compile or not hasattr(torch, "compile"):
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        logger.info("WM compiled with torch.compile(mode='reduce-overhead')")
        return compiled
    except Exception as exc:
        logger.warning("torch.compile failed (%s); falling back to eager mode.", exc)
        return model


def load_world_model_for_guidance(
    checkpoint_path,
    device="cuda",
    wm_arch="video",
    num_hist=1,
    num_pred=1,
    frameskip=8,
    use_compile=False,
    cameras=None,
):
    """
    Load WM for gradient-based guidance.

    By default (use_compile=False) the model is loaded in eager mode, which is
    the safest option.  Pass use_compile=True (--compile_wm) to apply
    torch.compile on the WM forward pass; this is compatible with the
    stop-gradient optimization (Opt-1) because the UNet backward is eliminated
    and only the WM backward runs.

    Pass cameras=["camera_0","camera_1","camera_2"] when the checkpoint was
    trained with multiple heads; the reward function will zero-fill the
    non-tactile cameras and only use the camera_2 head for reward.
    """
    if cameras is None:
        cameras = ["camera_2"]
    camera_dims = {c: CAMERA_EMB_DIMS.get(c, 512) for c in cameras}

    if wm_arch == "k_step":
        transition = TactileKStepTransformer(
            cameras=cameras,
            camera_dims=camera_dims,
            condition_cameras=cameras,
            predict_cameras=cameras,
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
    else:
        transition = TactileVideoTransformer(
            cameras=cameras,
            camera_dims=camera_dims,
            condition_cameras=cameras,
            predict_cameras=cameras,
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

    compile_note = "with torch.compile" if use_compile else "eager (no compile)"
    logger.info("WM (%s) loaded — %s", wm_arch, compile_note)
    return _maybe_compile(transition, use_compile)


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable reward-function factories
# ─────────────────────────────────────────────────────────────────────────────

def _unnorm_policy_to_raw(x0_pred, ac_min_t, ac_max_t):
    """Convert policy-normalized actions [-1,1] to raw action space (differentiable)."""
    return (x0_pred + 1.0) / 2.0 * (ac_max_t - ac_min_t) + ac_min_t


def _norm_raw_for_wm_differentiable(ac_raw, norm_stats):
    """Normalize raw 7d actions for the world model (differentiable linear op)."""
    shape = ac_raw.shape
    flat = ac_raw.reshape(-1, 7)
    flat = norm_stats.normalize_acs(flat)
    return flat.reshape(shape)


def make_reward_fn_video(
    wm_model,
    tactile_emb_history,
    state,
    h_ctx,
    norm_stats,
    text_embeds,
    ac_min_t,
    ac_max_t,
    device,
    tactile_emb_fallback=None,
    wm_cameras=None,
    text_embeds_neg=None,
):
    """
    Build a differentiable reward closure for the video (autoregressive) world model.

    The returned function accepts x0_pred (1, ac_chunk, 7) in policy [-1,1] space and
    returns a scalar reward tensor with grad attached.

    All context tensors (tactile, state) are snapshotted and detached at factory time so
    that repeated calls with different x0_pred are cheap and only differentiate through the
    action path.
    """
    if wm_cameras is None:
        wm_cameras = ["camera_2"]
    inp = build_tactile_wm_context(
        tactile_emb_history, h_ctx, device, tactile_emb_fallback=tactile_emb_fallback
    ).clone()  # (1, h_ctx, N_patches, 512) — clone for non-inference tensor
    st = (
        torch.tensor(state, dtype=torch.float32, device=device)
        .view(1, 1, 8)
        .expand(1, h_ctx, 8)
        .clone()
    )  # (1, h_ctx, 8)
    text_emb = text_embeds.clone()  # (1, D)
    text_emb_neg = text_embeds_neg.clone() if text_embeds_neg is not None else None
    # Zero-fill snapshots for non-tactile cameras (needed when WM has multiple heads)
    other_snaps = {
        cam: torch.zeros(1, h_ctx, PATCH_SIDE_DINOV3 ** 2, CAMERA_EMB_DIMS.get(cam, 768),
                         device=device, dtype=torch.float32)
        for cam in wm_cameras if cam != "camera_2"
    }

    def reward_fn(x0_pred):
        # x0_pred: (1, ac_chunk, 7) in [-1, 1] — has grad
        ac_chunk = x0_pred.shape[1]
        ac_raw = _unnorm_policy_to_raw(x0_pred, ac_min_t, ac_max_t)  # (1, ac_chunk, 7)

        # Build (1, h_ctx, 7) action window: clamp indices to ac_chunk
        idxs = [min(j, ac_chunk - 1) for j in range(h_ctx)]
        ac_window = torch.cat([ac_raw[:, i : i + 1, :] for i in idxs], dim=1)  # (1, h_ctx, 7)

        ac_normed = _norm_raw_for_wm_differentiable(ac_window, norm_stats)  # (1, h_ctx, 7)

        camera_inputs = {cam: snap for cam, snap in other_snaps.items()}
        camera_inputs["camera_2"] = inp
        preds, _ = wm_model(camera_inputs, st, ac_normed)
        pred_e = preds["camera_2"][:, -1, :, :]  # (1, N_patches, 512)

        emb = pred_e.mean(dim=1)  # (1, 512)
        emb = F.normalize(emb, dim=-1)
        reward = (emb @ text_emb.T)[:, 0]  # (1,)
        if text_emb_neg is not None:
            reward = reward - (emb @ text_emb_neg.T)[:, 0]
        return reward

    return reward_fn


def make_reward_fn_k_step(
    wm_model,
    tactile_emb_history,
    state,
    norm_stats,
    text_embeds,
    ac_min_t,
    ac_max_t,
    device,
    temporal_mode="lpb",
    tactile_emb_fallback=None,
    state_history=None,
    logged_cam2=None,
    logged_proprio=None,
    tactile_encoder=None,
    sensor_id=0,
    use_amp=False,
    wm_cameras=None,
    vision_emb_histories=None,
    text_embeds_neg=None,
):
    """
    Build a differentiable reward closure for the k-step world model.

    Supports both lpb and consecutive temporal modes, and optional full-episode logs
    for num_hist > 1 (same as the sampling script).
    """
    num_hist = int(wm_model.num_hist)
    fs = int(wm_model.frameskip)

    use_logged = (
        num_hist > 1
        and logged_cam2 is not None
        and logged_proprio is not None
        and len(logged_cam2) >= 1
        and tactile_encoder is not None
    )

    if wm_cameras is None:
        wm_cameras = ["camera_2"]

    if use_logged:
        inp_one, st_one = build_k_step_inp_and_st_from_logged(
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
        inp = inp_one.clone()
        st = st_one.clone()
    else:
        inp = build_tactile_k_step_context(
            tactile_emb_history, num_hist, fs, temporal_mode, device,
            tactile_emb_fallback=tactile_emb_fallback
        ).clone()
        st = build_k_step_state_tensor(
            state_history, num_hist, fs, temporal_mode, state, device, 1
        ).clone()

    text_emb = text_embeds.clone()
    text_emb_neg = text_embeds_neg.clone() if text_embeds_neg is not None else None
    # Build context tensors for non-tactile cameras using real embeddings when available.
    other_snaps = {}
    for cam in wm_cameras:
        if cam == "camera_2":
            continue
        cam_dim = CAMERA_EMB_DIMS.get(cam, 768)
        hist = vision_emb_histories.get(cam, deque()) if vision_emb_histories is not None else deque()
        if hist:
            fallback = list(hist)[-1].detach()
            other_snaps[cam] = build_tactile_k_step_context(
                hist, num_hist, fs, temporal_mode, device,
                tactile_emb_fallback=fallback,
            ).clone()
        else:
            other_snaps[cam] = torch.zeros(
                1, num_hist, PATCH_SIDE_DINOV3 ** 2, cam_dim,
                device=device, dtype=torch.float32,
            )

    def reward_fn(x0_pred):
        # x0_pred: (1, ac_chunk, 7) in [-1, 1] — has grad
        ac_chunk = x0_pred.shape[1]
        ac_raw = _unnorm_policy_to_raw(x0_pred, ac_min_t, ac_max_t)  # (1, ac_chunk, 7)

        # Build (1, num_hist, fs*7) collapsed actions for WM anchor at rollout_step=0
        slots = []
        for j in range(num_hist):
            if temporal_mode == "consecutive":
                idx_start = 0 - num_hist + 1 + j
            else:
                # lpb: idx_start = 0 - (num_hist - 1 - j) * fs
                idx_start = 0 - (num_hist - 1 - j) * fs
            idx_start = max(0, min(idx_start, ac_chunk - 1))

            frames = []
            for k in range(fs):
                idx = min(idx_start + k, ac_chunk - 1)
                frames.append(ac_raw[:, idx, :])  # (1, 7)
            slot = torch.cat(frames, dim=-1).unsqueeze(1)  # (1, 1, fs*7)
            slots.append(slot)

        acs_collapsed = torch.cat(slots, dim=1)  # (1, num_hist, fs*7)
        acs_normed = _norm_raw_for_wm_differentiable(acs_collapsed, norm_stats)  # (1, num_hist, fs*7)

        camera_inputs = {cam: snap for cam, snap in other_snaps.items()}
        camera_inputs["camera_2"] = inp
        preds, _ = wm_model(camera_inputs, st, acs_normed)
        pred_e = preds["camera_2"][:, -1, :, :]  # (1, N_patches, 512)

        emb = pred_e.mean(dim=1)  # (1, 512)
        emb = F.normalize(emb, dim=-1)
        reward = (emb @ text_emb.T)[:, 0]  # (1,)
        if text_emb_neg is not None:
            reward = reward - (emb @ text_emb_neg.T)[:, 0]
        return reward

    return reward_fn


def make_reward_fn(
    wm_model,
    tactile_emb_history,
    state,
    norm_stats,
    text_embeds,
    ac_min_t,
    ac_max_t,
    device,
    temporal_mode="lpb",
    tactile_emb_fallback=None,
    state_history=None,
    logged_cam2=None,
    logged_proprio=None,
    tactile_encoder=None,
    sensor_id=0,
    use_amp=False,
    wm_cameras=None,
    vision_emb_histories=None,
    text_embeds_neg=None,
):
    """Dispatch to video or k-step reward function factory."""
    if _is_k_step_wm(wm_model):
        return make_reward_fn_k_step(
            wm_model, tactile_emb_history, state, norm_stats, text_embeds,
            ac_min_t, ac_max_t, device, temporal_mode, tactile_emb_fallback,
            state_history, logged_cam2, logged_proprio, tactile_encoder, sensor_id, use_amp,
            wm_cameras=wm_cameras,
            vision_emb_histories=vision_emb_histories,
            text_embeds_neg=text_embeds_neg,
        )
    h_ctx = _wm_context_length(wm_model)
    return make_reward_fn_video(
        wm_model, tactile_emb_history, state, h_ctx, norm_stats, text_embeds,
        ac_min_t, ac_max_t, device, tactile_emb_fallback,
        wm_cameras=wm_cameras,
        text_embeds_neg=text_embeds_neg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guidance visualization  (DynaGuide-style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_guidance_visualization(
    corrections_list,
    guidance_list,
    diffusion_list,
    rewards_list,
    out_path,
    ac_dim_labels=None,
    query_idx=None,
    camera_images=None,
):
    """
    Visualize one guided-denoising call across all T×ss inner iterations.

    Mirrors DynaGuide's per-step analysis:
      - corrections_list  : x̂₀ predicted clean action at each inner iter  (T×ss entries)
      - guidance_list     : applied guidance √(1−ᾱₜ)·scale·∇r              (T×ss entries)
      - diffusion_list    : unguided diffusion direction √(1−ᾱₜ)·ε          (T×ss entries)
      - rewards_list      : scalar reward at each inner iter                 (T×ss entries)
      - camera_images     : optional list of (title_str, H×W×3 uint8 RGB ndarray)
                            shown as a row of images below the analysis plots.

    Produces a figure with:
      Top 2×2 analysis plots:
        [0,0] Reward vs iteration        — shows guidance effectiveness
        [0,1] Guidance vs diffusion norm — shows relative steering strength
        [1,0] x̂₀ XYZ dims vs iteration  — shows how the action chunk evolves
        [1,1] x̂₀ all dims heatmap       — full action chunk trajectory
      Bottom row (when camera_images is provided):
        One axis per camera, shown side-by-side with axis labels off.
    """
    n = len(rewards_list)
    if n == 0:
        return
    iters = np.arange(n)

    rewards = np.array(rewards_list, dtype=np.float32)

    # Norms — flatten (B, T, D) → scalar per entry
    g_norms = np.array([
        g.float().norm().item() for g in guidance_list
    ], dtype=np.float32)
    d_norms = np.array([
        d.float().norm().item() for d in diffusion_list
    ], dtype=np.float32)

    # x̂₀ mean over the action horizon: (n_iters, ac_dim)
    x0_mean = np.array([
        c[0].float().mean(dim=0).cpu().numpy() for c in corrections_list
    ], dtype=np.float32)  # (n, ac_dim)

    if ac_dim_labels is None:
        ac_dim_labels = ["x", "y", "z", "rx", "ry", "rz", "g"]

    title_suffix = f"  (query {query_idx})" if query_idx is not None else ""

    # ── Figure layout ──────────────────────────────────────────────────────
    has_cams = camera_images is not None and len(camera_images) > 0
    n_cams = len(camera_images) if has_cams else 0

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # Two-section layout: analysis (2×2) on top, cameras on bottom.
    # height_ratios makes the camera row ~40% the height of one analysis row.
    if has_cams:
        fig = plt.figure(figsize=(14, 12))
        outer = GridSpec(2, 1, figure=fig, height_ratios=[4, 1.2], hspace=0.38)
        inner_top = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0.40, wspace=0.30)
        inner_bot = GridSpecFromSubplotSpec(1, n_cams, subplot_spec=outer[1], wspace=0.06)
    else:
        fig = plt.figure(figsize=(14, 9))
        outer = GridSpec(1, 1, figure=fig)
        inner_top = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0.40, wspace=0.30)

    fig.suptitle(f"Guidance visualization{title_suffix}", fontsize=13, y=0.995)

    ax00 = fig.add_subplot(inner_top[0, 0])
    ax01 = fig.add_subplot(inner_top[0, 1])
    ax10 = fig.add_subplot(inner_top[1, 0])
    ax11 = fig.add_subplot(inner_top[1, 1])

    # ── [0,0] Reward vs iteration ──────────────────────────────────────────
    ax00.plot(iters, rewards, color="tab:blue", linewidth=2)
    ax00.set_xlabel("Denoising iteration (T × ss)")
    ax00.set_ylabel("Reward  (cosine sim)")
    ax00.set_title("Reward vs iteration")
    ax00.grid(True, alpha=0.3)

    # ── [0,1] Guidance norm vs diffusion norm ─────────────────────────────
    ax01.plot(iters, g_norms, color="tab:orange", linewidth=2, label="guidance  √(1−ᾱ)·scale·∇r")
    ax01.plot(iters, d_norms, color="tab:gray",   linewidth=2, label="diffusion  √(1−ᾱ)·ε", linestyle="--")
    ax01.set_xlabel("Denoising iteration (T × ss)")
    ax01.set_ylabel("L2 norm")
    ax01.set_title("Guidance vs diffusion norm")
    ax01.legend(fontsize=8)
    ax01.grid(True, alpha=0.3)

    # ── [1,0] x̂₀ XYZ + gripper vs iteration ─────────────────────────────
    colors_xyz = ["tab:red", "tab:green", "tab:blue"]
    for j in range(min(3, x0_mean.shape[1])):
        lbl = ac_dim_labels[j] if j < len(ac_dim_labels) else f"dim{j}"
        ax10.plot(iters, x0_mean[:, j], color=colors_xyz[j], label=lbl, linewidth=2)
    # Gripper (dim 6) on a twin y-axis so its [-1, 1] scale doesn't compress XYZ
    gripper_idx = 6
    if x0_mean.shape[1] > gripper_idx:
        ax10_g = ax10.twinx()
        lbl_g = ac_dim_labels[gripper_idx] if gripper_idx < len(ac_dim_labels) else "g"
        g_vals = x0_mean[:, gripper_idx]
        ax10_g.plot(iters, g_vals, color="tab:purple",
                    label=lbl_g, linewidth=2, linestyle="--")
        ax10_g.set_ylabel("Gripper", color="tab:purple", fontsize=8)
        ax10_g.tick_params(axis="y", labelcolor="tab:purple", labelsize=7)
        ax10_g.set_ylim(-1.1, 1.1)
        # Annotate start and end gripper values
        ax10_g.annotate(f"{g_vals[0]:.3f}", xy=(iters[0], g_vals[0]),
                        xytext=(4, 4), textcoords="offset points",
                        color="tab:purple", fontsize=7, ha="left")
        ax10_g.annotate(f"{g_vals[-1]:.3f}", xy=(iters[-1], g_vals[-1]),
                        xytext=(-4, 4), textcoords="offset points",
                        color="tab:purple", fontsize=7, ha="right")
        # Merge legends
        handles_l, labels_l = ax10.get_legend_handles_labels()
        handles_g, labels_g = ax10_g.get_legend_handles_labels()
        ax10.legend(handles_l + handles_g, labels_l + labels_g, fontsize=8)
    else:
        ax10.legend(fontsize=8)
    ax10.set_xlabel("Denoising iteration (T × ss)")
    ax10.set_ylabel("Mean x̂₀ over action horizon")
    ax10.set_title("Predicted clean action XYZ + gripper evolution")
    ax10.grid(True, alpha=0.3)

    # ── [1,1] Full x̂₀ heatmap (all dims, all iters) ──────────────────────
    im = ax11.imshow(
        x0_mean.T,                   # (ac_dim, n_iters)
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-1.0, vmax=1.0,
    )
    ax11.set_xlabel("Denoising iteration (T × ss)")
    ax11.set_ylabel("Action dim")
    ax11.set_title("x̂₀ heatmap (all dims)")
    tick_positions = list(range(min(len(ac_dim_labels), x0_mean.shape[1])))
    ax11.set_yticks(tick_positions)
    ax11.set_yticklabels(ac_dim_labels[: len(tick_positions)], fontsize=8)
    plt.colorbar(im, ax=ax11, fraction=0.046, pad=0.04)

    # ── Bottom row: camera images ──────────────────────────────────────────
    if has_cams:
        for j, (cam_title, img_rgb) in enumerate(camera_images):
            ax = fig.add_subplot(inner_bot[j])
            ax.imshow(img_rgb)
            ax.set_title(cam_title, fontsize=9, pad=3)
            ax.axis("off")

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diffusion policy with classifier-guided denoising via tactile-language reward"
    )
    robot_config_parse_args(parser)

    # Policy checkpoint
    parser.add_argument(
        "--ckpt_dir", type=str,
        default=None,
    )
    parser.add_argument("--ckpt_iter", type=str, default=23999)

    # World model
    parser.add_argument("--use_tactile", action="store_true", default=False)
    parser.add_argument("--wm_checkpoint", type=str, default=None)
    parser.add_argument("--wm_cameras", type=str, default="camera_2",
                        help="Comma-separated list of cameras the WM was trained on. "
                             "Use 'camera_0,camera_1,camera_2' for a 3-head checkpoint; "
                             "non-tactile cameras receive zero embeddings in the reward fn.")
    parser.add_argument("--anytouch_checkpoint", type=str, default=None)
    parser.add_argument("--reward_text", type=str, default="wiping the board")
    parser.add_argument("--norm_stats_json", type=str, default=None)
    parser.add_argument(
        "--wm_arch", type=str, choices=["video", "k_step"], default="video"
    )
    parser.add_argument("--temporal_mode", type=str, choices=["lpb", "consecutive"], default="lpb")
    parser.add_argument("--num_hist", type=int, default=None)
    parser.add_argument("--num_pred", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=8)

    # Guidance hyperparameters  (defaults from DynaGuide, Du & Song 2025)
    parser.add_argument(
        "--guidance_scale", type=float, default=1.5,
        help=(
            "Multiplier on the reward gradient applied to the noise prediction at each "
            "denoising step. DynaGuide paper default: 1.5."
        ),
    )
    parser.add_argument(
        "--ss", type=int, default=4,
        help=(
            "Stochastic-sampling repetitions per denoising timestep. "
            "At each noise level the guidance + DDIM step is repeated ss times without "
            "decrementing the timestep index, amplifying the guidance effect. "
            "DynaGuide paper default: 4. Set to 1 for standard single-pass guidance."
        ),
    )
    parser.add_argument(
        "--guidance_every", type=int, default=1,
        help=(
            "Apply WM reward guidance at every this many DDIM timesteps (default 1 = "
            "every step). guidance_every=2 halves WM calls; non-guided steps take one "
            "unguided DDIM step (ss not repeated)."
        ),
    )
    parser.add_argument(
        "--guidance_min_alpha", type=float, default=0.0,
        help=(
            "Skip reward guidance when ᾱₜ < this value (default 0.0 = apply at all steps). "
            "At early denoising steps ᾱ is tiny so x̂₀ reconstruction is unreliable and the "
            "effective gradient step size (1−ᾱ)/ᾱ × scale is enormous, causing reward to "
            "drop. Setting 0.1 skips roughly the first half of steps with the squaredcos "
            "schedule (where ᾱ < 0.1) and gives much more monotonic reward curves."
        ),
    )
    parser.add_argument(
        "--grad_clip", type=float, default=None,
        help=(
            "Clip the reward gradient L2 norm to at most this value before applying "
            "guidance (default None = no clipping). Suggested: 1.0. Prevents a single "
            "large gradient from overshooting and crashing reward at any step."
        ),
    )
    parser.add_argument(
        "--normalize_grad", action="store_true",
        help=(
            "Normalize ∇_{x̂₀} r to unit norm before applying guidance (FGSM / signed-"
            "gradient style). Decouples guidance_scale from reward gradient magnitude: "
            "guidance_scale then directly sets the step size in x̂₀ ∈ [−1,1] space. "
            "Useful when cosine-similarity gradients are tiny and guidance has no visible "
            "effect. With this flag, use guidance_scale ≈ 0.05–0.2 (5–20%% of action "
            "range per step). grad_clip and normalize_grad are redundant; prefer this flag alone."
        ),
    )
    parser.add_argument(
        "--lpb_scale", action="store_true",
        help=(
            "Use LPB-style scale factor √(1−ᾱₜ) instead of the default DynaGuide-style "
            "√(1−ᾱₜ)/√ᾱₜ. The LPB scale is bounded in [0,1] at all noise levels so it "
            "never explodes at early denoising steps — guidance_min_alpha is therefore "
            "unnecessary when this flag is set. The effective gradient is smaller by √ᾱₜ "
            "so guidance_scale typically needs to be increased (e.g. 5–10×)."
        ),
    )
    parser.add_argument(
        "--guidance_start_step", type=int, default=0,
        help=(
            "Only apply reward guidance when step_idx >= this value (default 0 = guide "
            "from the very first denoising step). step_idx counts from 0 (highest noise, "
            "t=T) to eval_steps-1 (lowest noise, t≈0), so setting this > 0 skips the "
            "noisiest early steps — equivalent to LPB's guidance_start_timestep. "
            "Complementary to --guidance_min_alpha; both can be used simultaneously."
        ),
    )


    # DINOv3 for RGB camera embeddings (camera_0, camera_1)
    parser.add_argument("--dinov3_hub_dir", type=str,
                        default=None)
    parser.add_argument("--dinov3_weights", type=str,
                        default=None)

    parser.add_argument("--amp", action="store_true", help="Use FP16 for tactile encoder inference.")
    parser.add_argument(
        "--compile_wm", action="store_true",
        help=(
            "Apply torch.compile(mode='reduce-overhead') to the world model. "
            "Safe with the stop-gradient optimization (Opt-1): only the WM backward "
            "runs, so compile accelerates both the WM forward and backward. "
            "Adds ~30–60 s one-time JIT compilation on first query; subsequent "
            "queries are 2–3× faster. Requires PyTorch ≥ 2.0."
        ),
    )
    parser.add_argument(
        "--visualize_guidance", action="store_true",
        help=(
            "Save a DynaGuide-style guidance visualization (reward, gradient norm, x̂₀ "
            "evolution) for every new policy query. Plots saved to <folder>/guidance_vis/."
        ),
    )

    # ── Robometer-based two-phase text switching ──────────────────────────────
    parser.add_argument(
        "--eval_server_url", type=str, default="http://localhost:8000",
        help="Robometer streaming eval server URL (default: port 8000). "
             "Start with: uv run python robometer/evals/eval_server_streaming.py ...",
    )
    parser.add_argument(
        "--robometer_timeout_s", type=float, default=30.0,
        help="HTTP timeout for robometer calls (default: 30 s)",
    )
    parser.add_argument(
        "--separate_phase", action="store_true", default=False,
        help="Enable two-phase guidance: robometer evaluates real camera_0 frames; "
             "when the reward exceeds --phase_threshold the guidance text switches to "
             "--reward_text_phase2.",
    )
    parser.add_argument(
        "--mixed_reward", action="store_true", default=False,
        help="Use a contrastive reward: r(text_pos, tactile) - r(text_neg, tactile). "
             "Phase 1: text_pos=--reward_text, text_neg=--reward_text_phase2. "
             "Phase 2 (after phase switch): flipped. "
             "Requires --reward_text_phase2 to be set.",
    )
    parser.add_argument(
        "--reward_text_phase2", type=str, default="",
        help="CLIP reward text used for phase-2 guidance (used when --separate_phase is set)",
    )
    parser.add_argument(
        "--phase_threshold", type=float, default=0.64,
        help="Robometer reward threshold on real observations that triggers the "
             "phase-1 → phase-2 text switch (default: 0.64)",
    )

    return parser.parse_args()


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

    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg), gripper_mode='continuous_smooth')
    policy = RealPolicy(["state"], args.ckpt_dir, args.ckpt_iter)

    camera_ids = [0, 1, 2]
    camera_types = ["rs", "zed", "tactile"]
    camera_info = EasyDict({
        "camera_id": camera_ids,
        "camera_type": camera_types,
        "camera_name": [
            f"camera_{camera_types[cid]}_{cid}" for cid in camera_ids
        ],
    })
    cr_interfaces = {}
    for cid in camera_ids:
        if "tactile" in camera_types[cid]:
            cr_iface = CameraRedisSubInterface(
                camera_info=camera_info, camera_id=cid, use_color=False, use_tactile=True
            )
        else:
            cr_iface = CameraRedisSubInterface(camera_info=camera_info, camera_id=cid)
        cr_iface.start()
        cr_interfaces[cid] = cr_iface

    controller_type = args.controller_type
    controller_cfg = YamlConfig(os.path.join(config_root, args.controller_cfg)).as_easydict()

    args.folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {args.folder}")


    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── World model, tactile encoder, CLIP ───────────────────────────────────
    wm_model = None
    tactile_encoder = None
    sensor_id = 0
    text_embeds = None
    norm_stats = None
    ac_min_t = None
    ac_max_t = None

    if args.use_tactile and args.wm_checkpoint and args.anytouch_checkpoint:
        nh = args.num_hist if args.num_hist is not None else 1
        npred = args.num_pred if args.num_pred is not None else 1
        fs = args.frameskip if args.frameskip is not None else 8

        logger.info(
            "Loading WM (arch=%s, compile_wm=%s) ...", args.wm_arch, args.compile_wm
        )
        if args.compile_wm:
            logger.info(
                "torch.compile enabled — first query will be slow (~30–60 s JIT); "
                "subsequent queries will be 2–3× faster."
            )

        wm_cameras_list = [c.strip() for c in args.wm_cameras.split(",")]
        wm_model = load_world_model_for_guidance(
            args.wm_checkpoint, device,
            wm_arch=args.wm_arch,
            num_hist=nh, num_pred=npred, frameskip=fs,
            use_compile=args.compile_wm,
            cameras=wm_cameras_list,
        )
        logger.info(
            "WM: arch=%s temporal_mode=%s (k_step: num_hist=%s num_pred=%s frameskip=%s)",
            args.wm_arch, args.temporal_mode, nh, npred, fs,
        )

        tactile_encoder, sensor_id = load_tactile_encoder(
            args.anytouch_checkpoint, device=str(device)
        )
        clip_model, tokenizer = load_clip_for_reward(args.anytouch_checkpoint, device)

        norm_stats_path = args.norm_stats_json
        if norm_stats_path is None and args.wm_checkpoint:
            default_path = os.path.join(
                os.path.dirname(os.path.abspath(args.wm_checkpoint)), "norm_stats.json"
            )
            if os.path.isfile(default_path):
                norm_stats_path = default_path
        norm_stats = NormStats(norm_stats_path, device=str(device))

        inputs = tokenizer(
            [args.reward_text], padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            text_embeds = clip_model.get_text_features(**inputs)
            text_embeds = F.normalize(text_embeds, dim=-1)
        logger.info(
            "Reward text: '%s'  guidance_scale=%.3f  ss=%d  every=%d  min_alpha=%.2f  "
            "grad_clip=%s  scale=%s  start_step=%d  normalize_grad=%s",
            args.reward_text, args.guidance_scale, args.ss,
            args.guidance_every, args.guidance_min_alpha,
            f"{args.grad_clip:.2f}" if args.grad_clip is not None else "off",
            "lpb(√(1−ᾱ))" if args.lpb_scale else "dynaguide(√(1−ᾱ)/√ᾱ)",
            args.guidance_start_step,
            args.normalize_grad,
        )

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

        # Convert policy normalization constants to tensors for differentiable unnorm
        ac_min_t = torch.tensor(policy.ac_min, dtype=torch.float32, device=device)
        ac_max_t = torch.tensor(policy.ac_max, dtype=torch.float32, device=device)

    wm_h_ctx = _tactile_emb_history_maxlen(wm_model, args.temporal_mode) if wm_model is not None else None

    # ── DINOv3 for RGB cameras (camera_0, camera_1) ───────────────────────────
    rgb_cams = [c for c in wm_cameras_list if c in ("camera_0", "camera_1")] if wm_model is not None else []
    vision_encoder = None
    if rgb_cams:
        vision_encoder = load_vision_encoder(args.dinov3_hub_dir, args.dinov3_weights, device)
        logger.info("DINOv3 encoder loaded for %s", rgb_cams)

    # ── Robot reset ───────────────────────────────────────────────────────────
    # init_joints = [
        # -0.18941418455646322, 0.3986268416401311, 0.37023725867044477,
        # -1.8986898980216536, -0.1220696096038897, 2.1729692311931763, -2.1428127053565933,
    # ]
    init_joints = [-0.14091554, 0.02724072, 0.33824686, -2.48239114, -0.12527768, 2.53220921, -2.87378524]

    episode_id = 0

    while True:
        data = {
            "actions": [],
            "ee_states": [],
            "joint_states": [],
            "gripper_states": [],
            "proprio": [],
            "rewards_log": [],
            "decision_ee_xyz": [],
            "policy_query_frame_idx": [],
            "phase_log": [],
        }
        for cid in camera_ids:
            data[f"camera_{cid}"] = []

        i = 0
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
        # Reset the policy's internal action queue at episode start
        policy.act_history = []

        # ── Two-phase state ───────────────────────────────────────────────────
        current_phase = 1
        current_task = args.reward_text
        text_embeds_active = text_embeds
        # Mixed reward: phase 1 → pos=text_embeds, neg=text_embeds_phase2; phase 2 flips.
        text_embeds_neg_active = text_embeds_phase2 if (args.mixed_reward and text_embeds_phase2 is not None) else None
        robometer_session_id: Optional[str] = None

        reset_joints_to(robot_interface, init_joints)
        time.sleep(1)
        print("Beginning Trajectory")
        interrupted = False
        warmup_step = 0
         # ── Visualization directory ───────────────────────────────────────────────
        vis_dir = None
        if args.visualize_guidance:
            ### add the traj_id to the guidance_vis directory
            traj_id = len(list(Path(args.folder).glob("traj_*.hdf5")))
            vis_dir = args.folder / "guidance_vis" / f"traj_{traj_id:04d}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Guidance visualizations → {vis_dir}")


        try:
            while True:
                last_state = robot_interface._state_buffer[-1]
                last_gripper_state = robot_interface._gripper_state_buffer[-1]

                obs_dict = {}
                obs_dict["state"] = eef_pose_to_state(
                    np.array(last_state.O_T_EE).reshape(4, 4).T,
                    np.array(last_gripper_state.width),
                )
                data["proprio"].append(np.asarray(obs_dict["state"], dtype=np.float32).copy())
                data["ee_states"].append(np.array(last_state.O_T_EE))
                data["joint_states"].append(np.array(last_state.q))
                data["gripper_states"].append(np.array(last_gripper_state.width))

                for cid in camera_ids:
                    img = cr_interfaces[cid].get_img()
                    if cid == 0:
                        img = cv2.resize(img["color"], (256, 256), interpolation=cv2.INTER_LINEAR)
                    elif cid == 1:
                        img = cv2.resize(img["color"], (256, 256), interpolation=cv2.INTER_LINEAR)
                    elif cid == 2:
                        img = img["tactile"]
                    data[f"camera_{cid}"].append(img)

                    if not args.use_tactile and cid == 2:
                        continue
                    obs_dict[f"camera_{cid}"] = img
                    if args.use_tactile and cid == 2:
                        tactile_buffer.append(img.copy())

                if args.use_tactile and episode_tactile_bg is None and obs_dict.get("camera_2") is not None:
                    episode_tactile_bg = np.asarray(obs_dict["camera_2"]).copy()

                # ── Encode current tactile frame ──────────────────────────────
                current_tactile_emb = None
                if tactile_emb_history is not None and tactile_encoder is not None and tactile_buffer:
                    current_tactile_emb = get_tactile_embedding_from_image(
                        tactile_encoder, tactile_buffer, sensor_id, device,
                        use_amp=args.amp, bg_image=episode_tactile_bg,
                    )
                    tactile_emb_history.append(current_tactile_emb.detach())
                    if state_history is not None:
                        state_history.append(np.asarray(obs_dict["state"], dtype=np.float32).copy())

                # ── Encode RGB camera frames (camera_0, camera_1) with DINOv3 ──
                if vision_encoder is not None and vision_emb_histories is not None:
                    for cam_name in rgb_cams:
                        cid_int = int(cam_name.split("_")[1])
                        v_emb = get_vision_embedding(vision_encoder, obs_dict[cam_name], cid_int, device)
                        vision_emb_histories[cam_name].append(v_emb.detach())

                # ── Log query metadata when act_history is about to be refilled ──
                is_new_query = not policy.act_history
                if is_new_query:
                    data["decision_ee_xyz"].append(
                        np.asarray(obs_dict["state"][:3], dtype=np.float32)
                    )
                    data["policy_query_frame_idx"].append(int(len(data["camera_0"]) - 1))

                # ── Get action (guided denoising or fallback) ─────────────────
                if wm_model is not None and tactile_encoder is not None and tactile_buffer:
                    state_vec = obs_dict["state"].astype(np.float32)
                    nh_wm = int(getattr(wm_model, "num_hist", 1))

                    # ── Robometer phase switching on real observations ─────────
                    if args.separate_phase and is_new_query and warmup_step > 3:
                        real_cam0 = obs_dict["camera_1"]
                        # Create session on first post-warmup query
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
                        # Append real frame and read progress reward
                        if robometer_session_id is not None:
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
                                # Phase 1 → 2 switch
                                if (current_phase == 1
                                        and text_embeds_phase2 is not None
                                        and hist_reward > args.phase_threshold):
                                    logger.info(
                                        "Phase 1→2 switch: robometer reward %.4f > %.4f — "
                                        "switching guidance text to: '%s'",
                                        hist_reward, args.phase_threshold,
                                        "move the dropper back to the red cup",
                                    )
                                    input("Press Enter to continue...")
                                    try:
                                        stream_delete(
                                            args.eval_server_url,
                                            robometer_session_id,
                                            timeout_s=args.robometer_timeout_s,
                                        )
                                        logger.info("Phase-1 session deleted: %s", robometer_session_id)
                                    except Exception as _e:
                                        logger.warning("Could not delete phase-1 session: %s", _e)
                                    robometer_session_id = None
                                    current_phase = 2
                                    current_task = args.reward_text_phase2
                                    text_embeds_active = text_embeds_phase2
                                    # Mixed reward phase 2: pos=phase2, neg=phase1
                                    if args.mixed_reward:
                                        text_embeds_neg_active = text_embeds
                                    # Seed phase-2 session with current real frame
                                    try:
                                        robometer_session_id = stream_create(
                                            args.eval_server_url,
                                            current_task,
                                            timeout_s=args.robometer_timeout_s,
                                        )
                                        stream_append_frame(
                                            args.eval_server_url,
                                            robometer_session_id,
                                            real_cam0,
                                            timeout_s=args.robometer_timeout_s,
                                        )
                                        logger.info(
                                            "Phase-2 robometer session created: %s",
                                            robometer_session_id,
                                        )
                                    except Exception as _e:
                                        logger.warning("Could not create phase-2 session: %s", _e)
                                        robometer_session_id = None
                            except Exception as _e:
                                logger.warning("stream_append_frame failed: %s", _e)

                    # Build a new reward closure each call — cheap when act_history is
                    # non-empty (closure is built but reward_fn is never evaluated).
                    reward_fn = make_reward_fn(
                        wm_model,
                        tactile_emb_history,
                        state_vec,
                        norm_stats,
                        text_embeds_active,
                        ac_min_t,
                        ac_max_t,
                        device,
                        temporal_mode=args.temporal_mode,
                        tactile_emb_fallback=current_tactile_emb,
                        state_history=state_history,
                        logged_cam2=data["camera_2"] if nh_wm > 1 else None,
                        logged_proprio=data["proprio"] if nh_wm > 1 else None,
                        tactile_encoder=tactile_encoder if nh_wm > 1 else None,
                        sensor_id=sensor_id,
                        use_amp=False,  # FP32 for numerically stable gradients
                        wm_cameras=wm_cameras_list,
                        vision_emb_histories=vision_emb_histories,
                        text_embeds_neg=text_embeds_neg_active,
                    )

                    t0 = time.time()
                    action, _, _ = policy.get_action_with_guidance(
                        obs_dict,
                        reward_fn=reward_fn,
                        guidance_scale=args.guidance_scale,
                        ss=args.ss,
                        guidance_every=args.guidance_every,
                        guidance_min_alpha=args.guidance_min_alpha,
                        grad_clip=args.grad_clip,
                        lpb_scale=args.lpb_scale,
                        guidance_start_step=args.guidance_start_step,
                        normalize_grad=args.normalize_grad,
                        binary_gripper=False
                    )
                    if is_new_query:
                        elapsed = time.time() - t0
                        logger.debug(
                            "Guided denoising took %.3fs (scale=%.2f ss=%d every=%d "
                            "min_alpha=%.2f start_step=%d lpb_scale=%s)",
                            elapsed, args.guidance_scale, args.ss,
                            args.guidance_every, args.guidance_min_alpha,
                            args.guidance_start_step, args.lpb_scale,
                        )
                        # ── DynaGuide-style visualization ─────────────────────
                        if vis_dir is not None and hasattr(policy.agent, "corrections_list"):
                            agent = policy.agent
                            if agent.corrections_list:
                                # Log reward summary to console
                                r_arr = np.array(agent.rewards_list, dtype=np.float32)
                                logger.info(
                                    "Query %d  reward: init=%.4f  final=%.4f  max=%.4f  "
                                    "guidance_norm_mean=%.4f",
                                    i,
                                    r_arr[0] if len(r_arr) else float("nan"),
                                    r_arr[-1] if len(r_arr) else float("nan"),
                                    r_arr.max() if len(r_arr) else float("nan"),
                                    np.array([
                                        g.float().norm().item()
                                        for g in agent.guidance_list
                                    ]).mean() if agent.guidance_list else float("nan"),
                                )
                                # Collect current camera frames for the plot.
                                # camera_0/1 come from OpenCV (BGR) → convert to RGB.
                                # camera_2 is tactile (may be 2-channel or grayscale).
                                _cam_imgs = []
                                for _cid in camera_ids:
                                    _key = f"camera_{_cid}"
                                    if _key not in obs_dict:
                                        continue
                                    _img = np.asarray(obs_dict[_key]).copy()
                                    if _cid in (0, 1):
                                        # BGR → RGB
                                        _img = _img[:, :, ::-1]
                                    else:
                                        # Tactile: ensure 3-channel uint8 for imshow
                                        if _img.ndim == 2:
                                            _img = np.stack([_img] * 3, axis=-1)
                                        elif _img.shape[-1] != 3:
                                            _img = _img[..., :3]
                                    if _img.dtype != np.uint8:
                                        _img = np.clip(_img, 0, 255).astype(np.uint8)
                                    _label = "tactile" if _cid == 2 else f"camera_{_cid}"
                                    _cam_imgs.append((_label, _img))

                                plot_guidance_visualization(
                                    agent.corrections_list,
                                    agent.guidance_list,
                                    agent.diffusion_list,
                                    agent.rewards_list,
                                    str(vis_dir / f"query_{i:05d}.png"),
                                    query_idx=i,
                                    camera_images=_cam_imgs if _cam_imgs else None,
                                )
                else:
                    action, _, _ = policy.get_action(obs_dict)

                if warmup_step <= 3:
                    warmup_step += 1
                    policy.act_history = []  # discard warmup actions
                    continue
                print(f'step {i} action: {action}')
                data["actions"].append(action)
                i += 1

                # Zero unused orientation dims
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
            print("\nKeyboard interrupt — saving trajectory...")
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
                "decision_ee_xyz", "policy_query_frame_idx", "rewards_log", "proprio", "phase_log",
                *(f"camera_{cid}" for cid in camera_ids),
            }
            for key, value in data.items():
                if key in skip_keys or not value:
                    continue
                grp.create_dataset(key, data=value)
            if data["proprio"]:
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
            print("Exiting...")
            break

        robot_interface.close()


if __name__ == "__main__":
    main()
