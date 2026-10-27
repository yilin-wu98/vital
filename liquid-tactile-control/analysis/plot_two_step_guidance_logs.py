#!/usr/bin/env python3
"""
Plot logs from run_dp_two_step_visual_tactile_guidance.py.

HDF5 ``data/`` group layout (written by the run script):
  action_chunks_log       (T, N, H, 7)    — N sampled chunks per query
  vision_rewards_log      (T, N)           — robometer visual reward per sample
  decision_ee_xyz         (T, 3)           — EE position at each query
  policy_query_frame_idx  (T,)             — camera frame index at each query
  chunk_before_guidance   (T, H, 7)        — selected chunk before tactile guidance (unnorm)
  chunk_after_guidance    (T, H, 7)        — chunk after guided denoising (unnorm)
  guidance_step_rewards   (T, R)           — tactile CLIP reward at each guidance step
  guidance_step_xt        (T, R, H, 7)    — normalized action x_t at each guidance step
  camera_0, camera_1, camera_2            — image time series

Outputs under ``--out-dir``  (default: <hdf5_dir>/plots_two_step_guidance/<stem>/):

  Part 1 — Visual sampling selection (mirrors plot_vision_sampling_logs.py)
    visual_rewards_over_decisions.png
    chunks_3d/chunk_XXXX.png              — per-decision 3D EE fan colored by visual reward
    chunks_composite/chunk_XXXX.png       — 3D + camera panels (when cameras available)
    overview_3d.png
    trajectory_chunks.mp4                 (unless --no-video)

  Part 2 — Tactile guidance process
    guidance_rewards.png                  — reward vs step per query + heatmap
    guidance_summary.png                  — Δreward and action-change summary across queries
    guidance_before_after_3d/query_XXXX.png — 3D before vs after + dim overlay + reward curve
    guidance_xt_heatmap/query_XXXX.png    — action evolution heatmap across guidance steps
    guidance_video.mp4                    (unless --no-video)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ACTION_SCALE = 0.05
ACTION_DIM_LABELS = ["x", "y", "z", "rx", "ry", "rz", "g"]


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 loading
# ─────────────────────────────────────────────────────────────────────────────

def load_hdf5(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        g = f["data"]
        out = {}
        for k in g.keys():
            out[k] = np.asarray(g[k][()])
        if "action_chunks_log" not in out:
            raise KeyError(f"{path} missing data/action_chunks_log")
        if "vision_rewards_log" not in out:
            raise KeyError(f"{path} missing data/vision_rewards_log")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared image / video utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_rgb_camera(img, stored_as_bgr: bool = False) -> np.ndarray:
    if img is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] == 4:
        x = x[..., :3]
    if stored_as_bgr and x.shape[-1] == 3:
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x


def resize_rgb_to_wh(im: np.ndarray, w: int, h: int) -> np.ndarray:
    if im is None or im.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    x = np.asarray(im)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[0] == h and x.shape[1] == w:
        return x
    return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)


def _panel_target_hw(rgb_list: list, target_h: int = 200) -> tuple[int, int]:
    best_wh: tuple[int, int] = (target_h, target_h)
    for im in rgb_list:
        if im is None or im.size == 0:
            continue
        h0, w0 = int(im.shape[0]), int(im.shape[1])
        if h0 <= 0:
            continue
        w = max(1, int(round(target_h * w0 / h0)))
        if w > best_wh[0]:
            best_wh = (w, target_h)
    return best_wh


def figure_to_bgr(fig: plt.Figure, dpi: int = 110) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return img


def write_video_cv2(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        return
    h0, w0 = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, float(fps), (w0, h0))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    for f in frames:
        if f.shape[0] != h0 or f.shape[1] != w0:
            f = cv2.resize(f, (w0, h0), interpolation=cv2.INTER_AREA)
        out.write(f)
    out.release()


# ─────────────────────────────────────────────────────────────────────────────
# 3D trajectory utilities
# ─────────────────────────────────────────────────────────────────────────────

def deltas_to_absolute_xyz(deltas: np.ndarray, p0: np.ndarray) -> np.ndarray:
    d = np.asarray(deltas, dtype=np.float64)
    origin = np.asarray(p0, dtype=np.float64).reshape(1, 3)
    steps = np.cumsum(d[:, :3] * ACTION_SCALE, axis=0)
    return np.vstack([origin, origin + steps])


def apply_3d_view(ax, elev: float = 0.0, azim: float = 0.0) -> None:
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def reward_norm_per_chunk(r: np.ndarray) -> tuple[float, float]:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    rmin, rmax = float(r.min()), float(r.max())
    if rmax - rmin < 1e-12:
        rmin, rmax = rmin - 0.5, rmax + 0.5
    return rmin, rmax


def draw_chunk_3d_on_axes(
    ax,
    action_chunk: np.ndarray,
    rewards_per_sample: np.ndarray,
    p0: np.ndarray,
    reward_norm: tuple[float, float] | None,
    view_elev: float,
    view_azim: float,
) -> cm.ScalarMappable:
    ac = np.asarray(action_chunk, dtype=np.float64)
    N, H, _ = ac.shape
    r = np.asarray(rewards_per_sample, dtype=np.float64).reshape(N)
    rmin, rmax = reward_norm if reward_norm is not None else reward_norm_per_chunk(r)
    norm = plt.Normalize(rmin, rmax)
    cmap = plt.colormaps["coolwarm"] if hasattr(plt, "colormaps") else plt.get_cmap("coolwarm")
    p0_arr = np.asarray(p0, dtype=np.float64).reshape(3)

    ax.scatter([p0_arr[0]], [p0_arr[1]], [p0_arr[2]], color="black", s=60, zorder=5, depthshade=False)
    ax.text(
        float(p0_arr[0]), float(p0_arr[1]), float(p0_arr[2]),
        f"EEF\n({p0_arr[0]:.3f},{p0_arr[1]:.3f},{p0_arr[2]:.3f})",
        fontsize=6, color="black", zorder=6,
    )

    for n in range(N):
        abs_xyz = deltas_to_absolute_xyz(ac[n, :, :3], p0_arr)
        color = cmap(norm(r[n]))
        ax.plot(abs_xyz[:, 0], abs_xyz[:, 1], abs_xyz[:, 2], color=color, linewidth=2.0, alpha=0.9)
        ax.text(
            float(abs_xyz[-1, 0]), float(abs_xyz[-1, 1]), float(abs_xyz[-1, 2]),
            f"{r[n]:.3f}", fontsize=7, color=color,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    apply_3d_view(ax, view_elev, view_azim)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def infer_query_frame_indices(n_queries: int, n_cam_frames: int) -> np.ndarray:
    if n_cam_frames <= 0:
        return np.zeros(n_queries, dtype=np.int64)
    if n_queries == 1:
        return np.array([n_cam_frames - 1], dtype=np.int64)
    return np.round(np.linspace(0, n_cam_frames - 1, n_queries)).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Visual sampling
# ─────────────────────────────────────────────────────────────────────────────

def plot_visual_rewards(vision_rewards_2d: np.ndarray, out_path: Path, title: str | None = None) -> None:
    T, N = vision_rewards_2d.shape
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.2, 1.0])

    im = axes[0].imshow(
        vision_rewards_2d.T, aspect="auto", origin="lower", cmap="viridis",
        interpolation="nearest", extent=[-0.5, T - 0.5, -0.5, N - 0.5],
    )
    axes[0].set_xlabel("Decision index (policy query)")
    axes[0].set_ylabel("Sample index")
    axes[0].set_title("Visual reward per sample (robometer)")
    plt.colorbar(im, ax=axes[0], label="visual reward")

    for n in range(N):
        axes[1].plot(np.arange(T), vision_rewards_2d[:, n],
                     marker="o", markersize=3, alpha=0.85, label=f"s{n}")
    axes[1].set_xlabel("Decision index")
    axes[1].set_ylabel("Visual reward")
    axes[1].set_title("Visual reward vs decision (one curve per sample)")
    axes[1].grid(True, alpha=0.3)
    if N <= 12:
        axes[1].legend(fontsize=7, ncol=2, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_chunk_3d_visual(
    action_chunk: np.ndarray,
    rewards_per_sample: np.ndarray,
    p0: np.ndarray,
    out_path: Path,
    title: str,
    reward_norm: tuple[float, float] | None = None,
    view_elev: float = 0.0,
    view_azim: float = 90.0,
) -> None:
    r = np.asarray(rewards_per_sample, dtype=np.float64).reshape(-1)
    rn = reward_norm if reward_norm is not None else reward_norm_per_chunk(r)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sm = draw_chunk_3d_on_axes(ax, action_chunk, rewards_per_sample, p0, rn, view_elev, view_azim)
    ax.set_title(title)
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.12, label="visual reward")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview_3d_visual(
    ac_log: np.ndarray,
    vision_rewards_2d: np.ndarray,
    decision_ee: np.ndarray | None,
    out_path: Path,
    title: str,
    view_elev: float = 0.0,
    view_azim: float = 0.0,
) -> None:
    ac = np.asarray(ac_log, dtype=np.float64)
    T, N, H = ac.shape[:3]
    r_all = np.asarray(vision_rewards_2d, dtype=np.float64)
    rmin, rmax = float(r_all.min()), float(r_all.max())
    if rmax == rmin:
        rmax = rmin + 1.0
    norm = plt.Normalize(rmin, rmax)
    cmap = plt.colormaps["coolwarm"] if hasattr(plt, "colormaps") else plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    p0_list: list[np.ndarray] = []

    for t in range(T):
        p0 = (
            np.asarray(decision_ee[t], dtype=np.float64).reshape(3)
            if decision_ee is not None
            else np.zeros(3, dtype=np.float64)
        )
        p0_list.append(p0)
        best_n = int(np.argmax(r_all[t]))
        for n in range(N):
            abs_xyz = deltas_to_absolute_xyz(ac[t, n, :, :3], p0)
            color = cmap(norm(r_all[t, n]))
            lw = 2.5 if n == best_n else 0.8
            alpha = 0.9 if n == best_n else 0.3
            ax.plot(abs_xyz[:, 0], abs_xyz[:, 1], abs_xyz[:, 2], color=color, linewidth=lw, alpha=alpha)
        ax.scatter([p0[0]], [p0[1]], [p0[2]], color="black", s=50, zorder=5, depthshade=False)
        ax.text(
            float(p0[0]), float(p0[1]), float(p0[2]),
            f"t{t}\n({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f})",
            fontsize=5, color="black", zorder=6,
        )

    if len(p0_list) > 1:
        pts = np.array(p0_list)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color="black", linewidth=1.0, linestyle="--", alpha=0.6, zorder=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, fontsize=9)
    apply_3d_view(ax, view_elev, view_azim)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.12, label="visual reward (global)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_visual_composite_figure(
    action_chunk: np.ndarray,
    rewards_vec: np.ndarray,
    p0: np.ndarray,
    rgb0: np.ndarray | None,
    rgb1: np.ndarray | None,
    rgb2: np.ndarray | None,
    title: str,
    view_elev: float,
    view_azim: float,
    camera_stored_as_bgr: bool,
) -> plt.Figure:
    r0 = to_rgb_camera(rgb0, stored_as_bgr=camera_stored_as_bgr)
    r1 = to_rgb_camera(rgb1, stored_as_bgr=camera_stored_as_bgr)
    r2 = to_rgb_camera(rgb2)
    tw, th = _panel_target_hw([r0, r1, r2], target_h=200)
    r0 = resize_rgb_to_wh(r0, tw, th)
    r1 = resize_rgb_to_wh(r1, tw, th)
    r2 = resize_rgb_to_wh(r2, tw, th)

    fig = plt.figure(figsize=(14.5, 4.2))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1.2, 1, 1, 1], wspace=0.22)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    sm = draw_chunk_3d_on_axes(ax3d, action_chunk, rewards_vec, p0, None, view_elev, view_azim)
    ax3d.set_title("EE traj (visual reward)", fontsize=10)
    fig.colorbar(sm, ax=ax3d, shrink=0.82, pad=0.02, label="visual reward")

    for col, (name, arr) in enumerate(
        (("camera_0", r0), ("camera_1", r1), ("camera_2 (tactile)", r2)), start=1
    ):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(arr)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Tactile guidance
# ─────────────────────────────────────────────────────────────────────────────

def plot_guidance_rewards(
    guidance_step_rewards: np.ndarray,
    out_path: Path,
    title: str | None = None,
) -> None:
    """Overview of tactile reward across all queries and guidance steps."""
    T, R = guidance_step_rewards.shape
    steps = np.arange(R)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap_q = plt.colormaps["viridis"] if hasattr(plt, "colormaps") else plt.get_cmap("viridis")
    for t in range(T):
        color = cmap_q(t / max(T - 1, 1))
        axes[0].plot(steps, guidance_step_rewards[t], color=color, linewidth=1.5, alpha=0.85, label=f"q{t}")
    # Mean ± std band
    mean_r = guidance_step_rewards.mean(axis=0)
    std_r  = guidance_step_rewards.std(axis=0, ddof=0)
    axes[0].plot(steps, mean_r, color="black", linewidth=2.5, linestyle="--", label="mean", zorder=5)
    axes[0].fill_between(steps, mean_r - std_r, mean_r + std_r, alpha=0.15, color="black")
    axes[0].set_xlabel("Guidance step")
    axes[0].set_ylabel("Tactile CLIP reward")
    axes[0].set_title("Tactile reward per guidance step (each line = one query)")
    axes[0].grid(True, alpha=0.3)
    if T <= 12:
        axes[0].legend(fontsize=7, ncol=2, loc="best")

    im = axes[1].imshow(
        guidance_step_rewards,
        aspect="auto", origin="upper", cmap="viridis",
        interpolation="nearest",
        extent=[-0.5, R - 0.5, T - 0.5, -0.5],
    )
    axes[1].set_xlabel("Guidance step")
    axes[1].set_ylabel("Query index")
    axes[1].set_title("Tactile reward heatmap (queries × guidance steps)")
    plt.colorbar(im, ax=axes[1], label="tactile reward")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_guidance_summary(
    guidance_step_rewards: np.ndarray,
    chunk_before: np.ndarray,
    chunk_after: np.ndarray,
    out_path: Path,
    title: str | None = None,
) -> None:
    """Per-query: reward delta, action change magnitude, and scatter."""
    T, R = guidance_step_rewards.shape
    queries = np.arange(T)

    reward_init  = guidance_step_rewards[:, 0]
    reward_final = guidance_step_rewards[:, -1]
    reward_delta = reward_final - reward_init

    ac_diff = np.asarray(chunk_after, dtype=np.float64) - np.asarray(chunk_before, dtype=np.float64)
    ac_change_norm = np.linalg.norm(ac_diff, axis=-1).mean(axis=-1)  # (T,)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(queries, reward_init,  "o-", color="tab:red",   label="initial reward", linewidth=2)
    axes[0, 0].plot(queries, reward_final, "o-", color="tab:green", label="final reward",   linewidth=2)
    axes[0, 0].set_xlabel("Query index")
    axes[0, 0].set_ylabel("Tactile CLIP reward")
    axes[0, 0].set_title("Tactile reward: initial vs final per query")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    bar_colors = ["tab:green" if d >= 0 else "tab:red" for d in reward_delta]
    axes[0, 1].bar(queries, reward_delta, color=bar_colors, alpha=0.8)
    axes[0, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 1].set_xlabel("Query index")
    axes[0, 1].set_ylabel("Δ reward (final − initial)")
    axes[0, 1].set_title("Reward improvement from guidance  (green = improved)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(queries, ac_change_norm, color="steelblue", alpha=0.8)
    axes[1, 0].set_xlabel("Query index")
    axes[1, 0].set_ylabel("Mean L2 change per action step")
    axes[1, 0].set_title("Action magnitude changed by guidance (per query)")
    axes[1, 0].grid(True, alpha=0.3)

    sc = axes[1, 1].scatter(ac_change_norm, reward_delta,
                            c=queries, cmap="viridis", s=80, zorder=3)
    axes[1, 1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for t in range(T):
        axes[1, 1].annotate(
            f"q{t}", (ac_change_norm[t], reward_delta[t]),
            fontsize=7, textcoords="offset points", xytext=(4, 4),
        )
    plt.colorbar(sc, ax=axes[1, 1], label="query index")
    axes[1, 1].set_xlabel("Action change (mean L2)")
    axes[1, 1].set_ylabel("Δ reward")
    axes[1, 1].set_title("Reward improvement vs action change")
    axes[1, 1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_guidance_before_after_3d(
    chunk_before: np.ndarray,
    chunk_after: np.ndarray,
    p0: np.ndarray,
    guidance_step_xt: np.ndarray,
    guidance_step_rewards: np.ndarray,
    query_idx: int,
    out_path: Path,
    view_elev: float = 0.0,
    view_azim: float = 90.0,
) -> None:
    """
    Per-query detail: 3D before/after + guidance-step trajectories (faded) +
    action-dimension overlay + reward progression curve.
    """
    H = chunk_before.shape[0]
    R = len(guidance_step_rewards)
    steps = np.arange(R)

    fig = plt.figure(figsize=(16, 5))
    gs_fig = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.5, 1.5, 1.0], wspace=0.30)

    # Left: 3D trajectories
    ax3d = fig.add_subplot(gs_fig[0, 0], projection="3d")
    p0_arr = np.asarray(p0, dtype=np.float64).reshape(3)

    # Intermediate guidance steps (faded, Blues color gradient)
    if guidance_step_xt is not None and R > 0:
        cmap_g = plt.colormaps["Blues"] if hasattr(plt, "colormaps") else plt.get_cmap("Blues")
        xt = np.asarray(guidance_step_xt, dtype=np.float64)
        for r in range(R):
            xyz_r = deltas_to_absolute_xyz(xt[r, :, :3], p0_arr)
            alpha_r = 0.25 + 0.45 * (r / max(R - 1, 1))
            color_r = cmap_g(0.35 + 0.50 * (r / max(R - 1, 1)))
            ax3d.plot(xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2],
                      color=color_r, linewidth=1.0, alpha=alpha_r, linestyle="--")

    before_xyz = deltas_to_absolute_xyz(chunk_before[:, :3], p0_arr)
    after_xyz  = deltas_to_absolute_xyz(chunk_after[:, :3],  p0_arr)
    ax3d.plot(before_xyz[:, 0], before_xyz[:, 1], before_xyz[:, 2],
              color="tab:red",   linewidth=2.5, label="before guidance", alpha=0.95)
    ax3d.plot(after_xyz[:, 0],  after_xyz[:, 1],  after_xyz[:, 2],
              color="tab:green", linewidth=2.5, label="after guidance",  alpha=0.95)
    ax3d.scatter([p0_arr[0]], [p0_arr[1]], [p0_arr[2]], color="black", s=60, zorder=5, depthshade=False)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title(f"Query {query_idx}: EE trajectories", fontsize=9)
    ax3d.legend(fontsize=8)
    apply_3d_view(ax3d, view_elev, view_azim)

    # Middle: action-dimension overlay (before dashed, after solid)
    ax_ac = fig.add_subplot(gs_fig[0, 1])
    t_steps = np.arange(H)
    dim_colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:gray"]
    for d in range(7):
        lbl = ACTION_DIM_LABELS[d] if d < len(ACTION_DIM_LABELS) else f"d{d}"
        ax_ac.plot(t_steps, chunk_before[:, d], color=dim_colors[d],
                   linestyle="--", linewidth=1.5, alpha=0.65, label=f"{lbl}–")
        ax_ac.plot(t_steps, chunk_after[:, d],  color=dim_colors[d],
                   linestyle="-",  linewidth=2.0, alpha=0.90, label=f"{lbl}+")
    ax_ac.set_xlabel("Action step in chunk")
    ax_ac.set_ylabel("Action value")
    ax_ac.set_title("Action dims: before (--) vs after (—)")
    ax_ac.grid(True, alpha=0.3)
    ax_ac.legend(fontsize=6, ncol=2, loc="best")

    # Right: reward progression
    ax_r = fig.add_subplot(gs_fig[0, 2])
    ax_r.plot(steps, guidance_step_rewards, color="steelblue", linewidth=2.5, marker="o", markersize=6, zorder=2)
    ax_r.scatter([0],     [guidance_step_rewards[0]],  color="tab:red",   s=100, zorder=5, edgecolors="white", linewidths=1.2, label="initial")
    ax_r.scatter([R - 1], [guidance_step_rewards[-1]], color="tab:green", s=100, zorder=5, edgecolors="white", linewidths=1.2, label="final")
    ax_r.set_xlabel("Guidance step")
    ax_r.set_ylabel("Tactile CLIP reward")
    ax_r.set_title("Reward progression")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=9)

    fig.suptitle(f"Query {query_idx} — Tactile guidance analysis", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_guidance_xt_heatmap(
    guidance_step_xt: np.ndarray,
    guidance_step_rewards: np.ndarray,
    query_idx: int,
    out_path: Path,
) -> None:
    """Action chunk mean (over horizon) evolution as a heatmap across guidance steps."""
    R, H, D = guidance_step_xt.shape
    steps = np.arange(R)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.2, 1.0])

    xt_mean = guidance_step_xt.mean(axis=1)  # (R, D)
    im = axes[0].imshow(
        xt_mean.T,
        aspect="auto", origin="lower", cmap="RdBu",
        vmin=-1.0, vmax=1.0,
        extent=[-0.5, R - 0.5, -0.5, D - 0.5],
    )
    axes[0].set_xlabel("Guidance step")
    axes[0].set_ylabel("Action dim")
    axes[0].set_title(f"Query {query_idx}: mean normalized action (over horizon) vs guidance step")
    axes[0].set_yticks(range(D))
    axes[0].set_yticklabels(ACTION_DIM_LABELS[:D], fontsize=9)
    plt.colorbar(im, ax=axes[0], label="normalized action (policy space)")

    axes[1].plot(steps, guidance_step_rewards, color="steelblue", linewidth=2.5, marker="o", markersize=6)
    axes[1].fill_between(steps, guidance_step_rewards, alpha=0.2, color="steelblue")
    axes[1].set_xlabel("Guidance step")
    axes[1].set_ylabel("Tactile CLIP reward")
    axes[1].set_title("Tactile reward vs guidance step")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Query {query_idx} — guidance step evolution", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_guidance_frame_figure(
    t_cur: int,
    guidance_step_rewards: np.ndarray,
    chunk_before_t: np.ndarray,
    chunk_after_t: np.ndarray,
    guidance_step_xt_t: np.ndarray | None,
    p0: np.ndarray,
    q_idx_t: int,
    cam0: np.ndarray | None,
    cam1: np.ndarray | None,
    cam2: np.ndarray | None,
    n_frames: int,
    stem_title: str,
    camera_stored_as_bgr: bool,
    view_elev: float,
    view_azim: float,
) -> plt.Figure:
    """One video frame: 3D before/after + reward curve + xt heatmap + cameras."""
    R = len(guidance_step_rewards)
    D = 7

    fig = plt.figure(figsize=(16.0, 8.0))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.35, 1.0], hspace=0.38)
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.28)
    bot_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.10)

    # ── Row 0: 3D  |  reward curve  |  xt heatmap ────────────────────────
    ax3d = fig.add_subplot(top_gs[0, 0], projection="3d")
    p0_arr = np.asarray(p0, dtype=np.float64).reshape(3)
    before_xyz = deltas_to_absolute_xyz(chunk_before_t[:, :3], p0_arr)
    after_xyz  = deltas_to_absolute_xyz(chunk_after_t[:, :3],  p0_arr)
    ax3d.plot(before_xyz[:, 0], before_xyz[:, 1], before_xyz[:, 2],
              color="tab:red",   linewidth=2.0, label="before", alpha=0.9)
    ax3d.plot(after_xyz[:, 0],  after_xyz[:, 1],  after_xyz[:, 2],
              color="tab:green", linewidth=2.0, label="after",  alpha=0.9)
    ax3d.scatter([p0_arr[0]], [p0_arr[1]], [p0_arr[2]], color="black", s=50, zorder=5, depthshade=False)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title(f"Query {t_cur}: before vs after", fontsize=9)
    ax3d.legend(fontsize=7)
    apply_3d_view(ax3d, view_elev, view_azim)

    ax_r = fig.add_subplot(top_gs[0, 1])
    steps = np.arange(R)
    ax_r.plot(steps, guidance_step_rewards, color="steelblue", linewidth=2.5, marker="o", markersize=5, zorder=2)
    ax_r.scatter([0],     [guidance_step_rewards[0]],  color="tab:red",   s=80, zorder=5, edgecolors="white")
    ax_r.scatter([R - 1], [guidance_step_rewards[-1]], color="tab:green", s=80, zorder=5, edgecolors="white")
    ax_r.set_xlabel("Guidance step")
    ax_r.set_ylabel("Tactile reward")
    ax_r.set_title("Tactile reward progression")
    ax_r.grid(True, alpha=0.3)

    if guidance_step_xt_t is not None:
        xt_mean = guidance_step_xt_t.mean(axis=1)  # (R, D)
        ax_xt = fig.add_subplot(top_gs[0, 2])
        im = ax_xt.imshow(
            xt_mean.T, aspect="auto", origin="lower", cmap="RdBu",
            vmin=-1.0, vmax=1.0,
            extent=[-0.5, R - 0.5, -0.5, min(D, xt_mean.shape[1]) - 0.5],
        )
        ax_xt.set_xlabel("Guidance step")
        ax_xt.set_ylabel("Action dim")
        ax_xt.set_title("Mean action evolution", fontsize=9)
        ax_xt.set_yticks(range(min(D, xt_mean.shape[1])))
        ax_xt.set_yticklabels(ACTION_DIM_LABELS[:min(D, xt_mean.shape[1])], fontsize=7)
        plt.colorbar(im, ax=ax_xt, fraction=0.046, pad=0.04, label="norm. action")

    # ── Row 1: cameras ────────────────────────────────────────────────────
    fi = int(np.clip(q_idx_t, 0, n_frames - 1)) if n_frames > 0 else 0
    cam_imgs = [
        to_rgb_camera(cam0[fi] if cam0 is not None and n_frames > 0 else None, stored_as_bgr=camera_stored_as_bgr),
        to_rgb_camera(cam1[fi] if cam1 is not None and n_frames > 0 else None, stored_as_bgr=camera_stored_as_bgr),
        to_rgb_camera(cam2[fi] if cam2 is not None and n_frames > 0 else None),
    ]
    cam_names = ["camera_0", "camera_1", "camera_2 (tactile)"]
    tw, th = _panel_target_hw(cam_imgs, target_h=190)
    for col, (name, img) in enumerate(zip(cam_names, cam_imgs)):
        ax_c = fig.add_subplot(bot_gs[0, col])
        ax_c.imshow(resize_rgb_to_wh(img, tw, th))
        ax_c.set_title(f"{name}  (frame {fi})", fontsize=9)
        ax_c.axis("off")

    fig.suptitle(f"{stem_title} — Query {t_cur}  ({R} guidance steps)", fontsize=11)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot two-step visual-sampling + tactile-guidance logs"
    )
    parser.add_argument("--hdf5_path", type=Path, required=True, help="Path to traj_XXXX.hdf5")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: <hdf5_dir>/plots_two_step_guidance/<stem>)",
    )
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit number of decisions to plot")
    parser.add_argument("--fps", type=float, default=4.0, help="FPS for output MP4s")
    parser.add_argument("--no-video", action="store_true", help="Skip all MP4 outputs")
    parser.add_argument("--no-composite", action="store_true", help="Skip composite PNG/video for Part 1")
    parser.add_argument("--no-guidance-plots", action="store_true", help="Skip Part 2 (tactile guidance)")
    parser.add_argument("--view-elev", type=float, default=0.0, help="3D view_init elevation")
    parser.add_argument("--view-azim", type=float, default=0.0, help="3D view_init azimuth")
    parser.add_argument("--camera-as-bgr", action="store_true", help="Treat camera_0/1 frames as BGR")
    parser.add_argument("--composite-dpi", type=int, default=100, help="DPI for composite rasterization")
    args = parser.parse_args()

    path = args.hdf5_path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    out_dir = args.out_dir or (path.parent / "plots_two_step_guidance" / path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    data = load_hdf5(path)
    ac_log         = np.asarray(data["action_chunks_log"])           # (T, N, H, 7)
    vision_rewards = np.asarray(data["vision_rewards_log"])          # (T, N)
    decision_ee    = np.asarray(data["decision_ee_xyz"]) if "decision_ee_xyz" in data else None
    q_idx_raw      = data.get("policy_query_frame_idx")

    has_guidance = all(
        k in data for k in
        ("guidance_step_rewards", "guidance_step_xt", "chunk_before_guidance", "chunk_after_guidance")
    )

    T_ac = int(ac_log.shape[0])
    T_vr = int(vision_rewards.shape[0])
    if T_ac != T_vr:
        # Suffix-align: the last T_vr rows of action_chunks match the vision_rewards rows.
        # This happens when the first policy query fired before the robometer session was ready.
        print(
            f"Warning: action_chunks_log has {T_ac} rows but vision_rewards_log has {T_vr}; "
            f"using the last {T_vr} rows of action_chunks (and q_idx / decision_ee) to match rewards."
        )
        ac_log = ac_log[-T_vr:]
        if decision_ee is not None:
            decision_ee = decision_ee[-T_vr:] if decision_ee.shape[0] == T_ac else decision_ee
        if q_idx_raw is not None:
            q_arr = np.asarray(q_idx_raw, dtype=np.int64).reshape(-1)
            if q_arr.shape[0] == T_ac:
                q_idx_raw = q_arr[-T_vr:]
    T = T_vr

    cam0 = data.get("camera_0")
    cam1 = data.get("camera_1")
    cam2 = data.get("camera_2")
    n_frames = int(cam0.shape[0]) if cam0 is not None else 0

    q_idx: np.ndarray | None = None
    if q_idx_raw is not None:
        q_idx = np.asarray(q_idx_raw, dtype=np.int64).reshape(-1)
        if q_idx.shape[0] != T:
            print(f"Warning: policy_query_frame_idx length {q_idx.shape[0]} != {T}; re-inferring.")
            q_idx = None
    if q_idx is None and n_frames > 0:
        q_idx = infer_query_frame_indices(T, n_frames)
        print(f"Inferred {T} query frame indices over {n_frames} camera frames.")

    n_plot = T if args.max_chunks is None else min(T, args.max_chunks)

    # ════════════════════════════════════════════════════════════════════════
    # Part 1 — Visual sampling
    # ════════════════════════════════════════════════════════════════════════
    print("\n=== Part 1: Visual sampling selection ===")

    plot_visual_rewards(
        vision_rewards[:n_plot],
        out_dir / "visual_rewards_over_decisions.png",
        title=path.name,
    )
    print("Wrote visual_rewards_over_decisions.png")

    chunk_dir = out_dir / "chunks_3d"
    chunk_dir.mkdir(exist_ok=True)
    for t in range(n_plot):
        p0 = (
            np.asarray(decision_ee[t], dtype=np.float64).reshape(3)
            if decision_ee is not None else np.zeros(3, dtype=np.float64)
        )
        r_vec = vision_rewards[t]
        plot_chunk_3d_visual(
            ac_log[t], r_vec, p0,
            chunk_dir / f"chunk_{t:04d}.png",
            title=f"{path.name} — decision {t}",
            reward_norm=reward_norm_per_chunk(r_vec),
            view_elev=args.view_elev, view_azim=args.view_azim,
        )
    print(f"Wrote {n_plot} 3D chunk plots → {chunk_dir}")

    plot_overview_3d_visual(
        ac_log[:n_plot], vision_rewards[:n_plot],
        decision_ee[:n_plot] if decision_ee is not None else None,
        out_dir / "overview_3d.png",
        title=f"{path.name} — overview ({n_plot} decisions)",
        view_elev=args.view_elev, view_azim=args.view_azim,
    )
    print("Wrote overview_3d.png")

    video_frames_vis: list[np.ndarray] = []
    if not args.no_composite and q_idx is not None and n_frames > 0:
        comp_dir = out_dir / "chunks_composite"
        comp_dir.mkdir(exist_ok=True)
        for t in range(n_plot):
            p0 = (
                np.asarray(decision_ee[t]).reshape(3)
                if decision_ee is not None else np.zeros(3)
            )
            fi = int(np.clip(q_idx[t], 0, n_frames - 1))
            fig = build_visual_composite_figure(
                ac_log[t], vision_rewards[t], p0,
                cam0[fi] if cam0 is not None else None,
                cam1[fi] if cam1 is not None else None,
                cam2[fi] if cam2 is not None else None,
                title=f"{path.stem} — decision {t}",
                view_elev=args.view_elev, view_azim=args.view_azim,
                camera_stored_as_bgr=args.camera_as_bgr,
            )
            fig.savefig(comp_dir / f"chunk_{t:04d}.png",
                        dpi=args.composite_dpi, bbox_inches="tight", pad_inches=0.08)
            if not args.no_video:
                video_frames_vis.append(figure_to_bgr(fig, dpi=args.composite_dpi))
            else:
                plt.close(fig)
        print(f"Wrote {n_plot} composite plots → {comp_dir}")
    elif not args.no_composite:
        print("Skipping composite/video: no camera data or frame alignment available.")

    if not args.no_video and video_frames_vis:
        mp4_vis = out_dir / "trajectory_chunks.mp4"
        write_video_cv2(video_frames_vis, mp4_vis, args.fps)
        print(f"Wrote {mp4_vis}")

    # ════════════════════════════════════════════════════════════════════════
    # Part 2 — Tactile guidance
    # ════════════════════════════════════════════════════════════════════════
    if has_guidance and not args.no_guidance_plots:
        print("\n=== Part 2: Tactile guidance ===")

        guidance_rewards_log = np.asarray(data["guidance_step_rewards"])  # (T, R)
        guidance_xt_log      = np.asarray(data["guidance_step_xt"])       # (T, R, H, 7)
        chunk_before_log     = np.asarray(data["chunk_before_guidance"])  # (T, H, 7)
        chunk_after_log      = np.asarray(data["chunk_after_guidance"])   # (T, H, 7)

        T_g = min(n_plot, int(guidance_rewards_log.shape[0]))

        plot_guidance_rewards(
            guidance_rewards_log[:T_g],
            out_dir / "guidance_rewards.png",
            title=path.name,
        )
        print("Wrote guidance_rewards.png")

        plot_guidance_summary(
            guidance_rewards_log[:T_g],
            chunk_before_log[:T_g],
            chunk_after_log[:T_g],
            out_dir / "guidance_summary.png",
            title=path.name,
        )
        print("Wrote guidance_summary.png")

        before_after_dir = out_dir / "guidance_before_after_3d"
        before_after_dir.mkdir(exist_ok=True)
        xt_heatmap_dir = out_dir / "guidance_xt_heatmap"
        xt_heatmap_dir.mkdir(exist_ok=True)

        video_frames_guid: list[np.ndarray] = []
        for t in range(T_g):
            p0 = (
                np.asarray(decision_ee[t]).reshape(3)
                if decision_ee is not None else np.zeros(3)
            )
            plot_guidance_before_after_3d(
                chunk_before_log[t], chunk_after_log[t], p0,
                guidance_xt_log[t], guidance_rewards_log[t],
                query_idx=t,
                out_path=before_after_dir / f"query_{t:04d}.png",
                view_elev=args.view_elev, view_azim=args.view_azim,
            )
            plot_guidance_xt_heatmap(
                guidance_xt_log[t], guidance_rewards_log[t],
                query_idx=t,
                out_path=xt_heatmap_dir / f"query_{t:04d}.png",
            )

            if not args.no_video and q_idx is not None:
                q_fi = (
                    int(np.clip(q_idx[t], 0, n_frames - 1))
                    if n_frames > 0 else 0
                )
                fig_g = build_guidance_frame_figure(
                    t_cur=t,
                    guidance_step_rewards=guidance_rewards_log[t],
                    chunk_before_t=chunk_before_log[t],
                    chunk_after_t=chunk_after_log[t],
                    guidance_step_xt_t=guidance_xt_log[t],
                    p0=p0,
                    q_idx_t=q_fi,
                    cam0=cam0, cam1=cam1, cam2=cam2,
                    n_frames=n_frames,
                    stem_title=path.stem,
                    camera_stored_as_bgr=args.camera_as_bgr,
                    view_elev=args.view_elev,
                    view_azim=args.view_azim,
                )
                video_frames_guid.append(figure_to_bgr(fig_g, dpi=args.composite_dpi))

        print(f"Wrote {T_g} before/after 3D plots → {before_after_dir}")
        print(f"Wrote {T_g} xt heatmap plots → {xt_heatmap_dir}")

        if not args.no_video and video_frames_guid:
            gvid = out_dir / "guidance_video.mp4"
            write_video_cv2(video_frames_guid, gvid, args.fps)
            print(f"Wrote {gvid}")

    elif not has_guidance:
        print(
            "\nNote: guidance keys missing in HDF5 "
            "(guidance_step_rewards / guidance_step_xt / chunk_before/after_guidance). "
            "Part 2 skipped."
        )

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
