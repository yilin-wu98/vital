# Publication cleanup (2026-10-27): adapted from deoxys_control/deoxys/examples/run_dp_model_tactile_sampling.py.
"""Context construction shared by liquid tactile guidance."""
import os
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from dino_wm.dino_models_k_step import TactileKStepTransformer

CAMERA_EMB_DIMS = {"camera_0": 768, "camera_1": 768, "camera_2": 512}
PATCH_SIDE_DINOV3 = 14
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

def load_tactile_encoder(checkpoint_path, sensor='gelsight_mini', num_frames=4, device='cuda'):
    """Load AnyTouch tactile encoder for single-frame embedding."""
    from dino_wm.hdf5_to_dataset_tactile import _load_tactile_model
    model, sensor_id = _load_tactile_model(
        checkpoint_path, sensor, num_frames=num_frames, device=str(device)
    )
    return model, sensor_id

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

def eef_pose_to_state(T, gripper):
    # Extract translation
    x, y, z = T[:3, 3]

    # Extract rotation matrix and convert to quaternion
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # Returns [qx, qy, qz, qw]

    # Concatenate into final state
    eef_state = np.concatenate(([x, y, z], quat, [gripper]))
    return eef_state
