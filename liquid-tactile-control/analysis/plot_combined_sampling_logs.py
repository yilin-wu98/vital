#!/usr/bin/env python3
"""
Plot combined tactile + vision sampling logs from run_dp_tactile_vision_combined.py.

HDF5 ``data/`` group expected:
  - action_chunks_log      (T, N, H, 7)
  - tactile_rewards_log    (T, N)
  - vision_rewards_log     (T, N)
  - decision_ee_xyz        (T, 3) optional
  - policy_query_frame_idx (T,) optional
  - camera_0, camera_1, camera_2 — image time series
  attrs:
  - selection_source_log   JSON list of "vision"|"tactile"|"random" (length T)

Falls back to ``rewards_log`` as vision_rewards_log if the combined keys are absent.

Outputs under ``--out-dir``:
  - combined_rewards.mp4  — per-decision video frame:
      row 0: 3D EE fan (tactile reward colors) | 3D EE fan (vision reward colors) |
              max-reward-over-samples line plot (both signals, running up to t)
      row 1: tactile std curve + threshold | vision std curve + threshold |
              selection source log
      row 2: camera_0 | camera_1 | camera_2 at query frame
  - overview_3d_tactile.png / overview_3d_vision.png
  - rewards_over_decisions.png  (heatmap + lines for both signals)
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Optional

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

_SOURCE_COLORS = {"vision": "steelblue", "tactile": "seagreen", "random": "silver"}
_SOURCE_MARKER = {"vision": "o", "tactile": "s", "random": "^"}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_hdf5(path: Path) -> tuple[dict, dict]:
    """Return (arrays_dict, attrs_dict).  Rewards are broadcast to 2-D."""
    with h5py.File(path, "r") as f:
        g = f["data"]
        arrays: dict = {}
        for k in g.keys():
            arrays[k] = np.asarray(g[k][()])
        attrs: dict = dict(g.attrs)

    if "action_chunks_log" not in arrays:
        raise KeyError(f"{path} missing data/action_chunks_log")

    # Normalise reward arrays to (T, N)
    for key in ("tactile_rewards_log", "vision_rewards_log", "rewards_log"):
        if key in arrays:
            r = arrays[key]
            arrays[key] = r.mean(axis=-1) if r.ndim == 3 else r

    # Fallback: single rewards_log → use as vision
    if "vision_rewards_log" not in arrays and "rewards_log" in arrays:
        arrays["vision_rewards_log"] = arrays["rewards_log"]
    if "tactile_rewards_log" not in arrays:
        if "vision_rewards_log" in arrays:
            T, N = arrays["vision_rewards_log"].shape
            arrays["tactile_rewards_log"] = np.zeros((T, N), dtype=np.float32)
        else:
            raise KeyError(f"{path} missing both tactile_rewards_log and rewards_log")

    return arrays, attrs


def infer_query_frame_indices(n_queries: int, n_cam_frames: int) -> np.ndarray:
    if n_cam_frames <= 0:
        return np.zeros(n_queries, dtype=np.int64)
    if n_queries == 1:
        return np.array([n_cam_frames - 1], dtype=np.int64)
    return np.round(np.linspace(0, n_cam_frames - 1, n_queries)).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_rgb(img: np.ndarray | None, bgr: bool = False) -> np.ndarray:
    if img is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] == 4:
        x = x[..., :3]
    if bgr and x.shape[-1] == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x


def resize_to_wh(im: np.ndarray, w: int, h: int) -> np.ndarray:
    x = to_rgb(im)
    if x.shape[0] == h and x.shape[1] == w:
        return x
    return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)


def _panel_hw(images: list[np.ndarray | None], target_h: int = 200) -> tuple[int, int]:
    best_w = target_h
    for im in images:
        if im is None or (hasattr(im, "size") and im.size == 0):
            continue
        h0, w0 = int(im.shape[0]), int(im.shape[1])
        if h0 > 0:
            best_w = max(best_w, int(round(target_h * w0 / h0)))
    return best_w, target_h


# ─────────────────────────────────────────────────────────────────────────────
# 3-D plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def deltas_to_absolute_xyz(deltas: np.ndarray, p0: np.ndarray) -> np.ndarray:
    d = np.asarray(deltas, dtype=np.float64)
    origin = np.asarray(p0, dtype=np.float64).reshape(1, 3)
    return np.vstack([origin, origin + np.cumsum(d[:, :3] * ACTION_SCALE, axis=0)])


def draw_chunk_3d(ax, action_chunk: np.ndarray, rewards: np.ndarray, p0: np.ndarray,
                  rmin: float, rmax: float, view_elev: float, view_azim: float,
                  best_idx: int | None = None, label_prefix: str = "") -> cm.ScalarMappable:
    N, H = action_chunk.shape[:2]
    r = np.asarray(rewards, dtype=np.float64).reshape(N)
    norm = plt.Normalize(rmin, rmax)
    cmap = plt.colormaps.get_cmap("coolwarm")
    p0_arr = np.asarray(p0, dtype=np.float64).reshape(3)

    ax.scatter([p0_arr[0]], [p0_arr[1]], [p0_arr[2]], color="black", s=40, zorder=5, depthshade=False)

    for n in range(N):
        xyz = deltas_to_absolute_xyz(action_chunk[n, :, :3], p0_arr)
        color = cmap(norm(r[n]))
        is_best = (n == best_idx)
        lw = 3.0 if is_best else 1.0
        alpha = 1.0 if is_best else 0.45
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=lw, alpha=alpha,
                zorder=4 if is_best else 2)
        if is_best:
            ax.text(float(xyz[-1, 0]), float(xyz[-1, 1]), float(xyz[-1, 2]),
                    f"★{r[n]:.3f}", fontsize=11, color=color, zorder=6)
        else:
            ax.text(float(xyz[-1, 0]), float(xyz[-1, 1]), float(xyz[-1, 2]),
                    f"{r[n]:.3f}", fontsize=10, color=color)

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("z", fontsize=10)
    ax.tick_params(labelsize=9)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    ax.view_init(elev=view_elev, azim=view_azim)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


# ─────────────────────────────────────────────────────────────────────────────
# Per-decision composite frame
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_frame(
    t_cur: int,
    n_plot: int,
    ac_log: np.ndarray,          # (T, N, H, 7)
    tactile_r: np.ndarray,       # (T, N)
    vision_r: np.ndarray,        # (T, N)
    selection_sources: list[str] | None,
    q_idx: np.ndarray | None,    # (T,)
    decision_ee: np.ndarray | None,  # (T, 3)
    cam0: np.ndarray | None,
    cam1: np.ndarray | None,
    cam2: np.ndarray | None,
    n_frames: int,
    tactile_std_threshold: float,
    vision_std_threshold: float,
    stem_title: str,
    view_elev: float,
    view_azim: float,
    camera_stored_as_bgr: bool,
) -> plt.Figure:
    t = int(np.clip(t_cur, 0, n_plot - 1))
    T, N = tactile_r.shape

    # ── Pre-compute running stats ──────────────────────────────────────────
    tac_std  = np.std(tactile_r[:n_plot], axis=1, ddof=0)
    vis_std  = np.std(vision_r[:n_plot],  axis=1, ddof=0)
    tac_best = np.max(tactile_r[:n_plot], axis=1)  # best-sample reward per decision
    vis_best = np.max(vision_r[:n_plot],  axis=1)

    # Reward norm: per-step (only over the N samples at decision t)
    t_rmin, t_rmax = float(tactile_r[t].min()), float(tactile_r[t].max())
    v_rmin, v_rmax = float(vision_r[t].min()),  float(vision_r[t].max())
    if t_rmax <= t_rmin: t_rmax = t_rmin + 1e-6
    if v_rmax <= v_rmin: v_rmax = v_rmin + 1e-6

    best_tac_idx = int(np.argmax(tactile_r[t]))
    best_vis_idx = int(np.argmax(vision_r[t]))
    # Determine which sample was actually selected
    if selection_sources is not None and t < len(selection_sources):
        src = selection_sources[t]
        selected_idx = best_vis_idx if src == "vision" else (
            best_tac_idx if src == "tactile" else int(np.random.randint(N))
        )
    else:
        src = "unknown"
        selected_idx = best_vis_idx

    # EE position at decision t
    if decision_ee is not None:
        p0 = np.asarray(decision_ee[t], dtype=np.float64).reshape(3)
    else:
        p0 = np.zeros(3, dtype=np.float64)

    # Camera frame index
    if q_idx is not None and n_frames > 0:
        fi = int(np.clip(int(q_idx[t]), 0, n_frames - 1))
    elif n_frames > 0:
        fi = int(infer_query_frame_indices(n_plot, n_frames)[t])
    else:
        fi = 0

    # Camera images
    c0 = cam0[fi] if cam0 is not None and n_frames > 0 else None
    c1 = cam1[fi] if cam1 is not None and n_frames > 0 else None
    c2 = cam2[fi] if cam2 is not None and n_frames > 0 else None
    r0 = to_rgb(c0, bgr=camera_stored_as_bgr)
    r1 = to_rgb(c1, bgr=camera_stored_as_bgr)
    r2 = to_rgb(c2, bgr=camera_stored_as_bgr)
    tw, th = _panel_hw([r0, r1, r2], target_h=200)
    r0 = resize_to_wh(r0, tw, th)
    r1 = resize_to_wh(r1, tw, th)
    r2 = resize_to_wh(r2, tw, th)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 12))
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        height_ratios=[2.2, 1.3, 0.95],
        hspace=0.42, wspace=0.30,
    )

    # ── Row 0: 3D fan plots + selected reward over time ────────────────────
    ax_tac3d = fig.add_subplot(gs[0, 0], projection="3d")
    sm_t = draw_chunk_3d(
        ax_tac3d, ac_log[t], tactile_r[t], p0,
        t_rmin, t_rmax, view_elev, view_azim,
        best_idx=best_tac_idx, label_prefix="tac",
    )
    ax_tac3d.set_title(f"Tactile reward (decision {t})\nbest={tactile_r[t, best_tac_idx]:.4f}",
                       fontsize=9)
    fig.colorbar(sm_t, ax=ax_tac3d, shrink=0.55, pad=0.05, label="tactile r")

    ax_vis3d = fig.add_subplot(gs[0, 1], projection="3d")
    sm_v = draw_chunk_3d(
        ax_vis3d, ac_log[t], vision_r[t], p0,
        v_rmin, v_rmax, view_elev, view_azim,
        best_idx=best_vis_idx, label_prefix="vis",
    )
    ax_vis3d.set_title(f"Vision reward (decision {t})\nbest={vision_r[t, best_vis_idx]:.4f}",
                       fontsize=9)
    fig.colorbar(sm_v, ax=ax_vis3d, shrink=0.55, pad=0.05, label="vision r")

    # Selected reward over time (running line plot)
    ax_sel = fig.add_subplot(gs[0, 2])
    xs = np.arange(t + 1, dtype=float)
    ax_sel.plot(xs, tac_best[:t + 1], color="seagreen", linewidth=1.8,
                marker="o", markersize=4, label="tactile best", alpha=0.9)
    ax_sel.plot(xs, vis_best[:t + 1], color="steelblue", linewidth=1.8,
                marker="o", markersize=4, label="vision best", alpha=0.9)
    ax_sel.scatter([float(t)], [float(tac_best[t])], color="seagreen", s=80,
                   edgecolors="white", linewidths=1.5, zorder=5)
    ax_sel.scatter([float(t)], [float(vis_best[t])], color="steelblue", s=80,
                   edgecolors="white", linewidths=1.5, zorder=5)
    all_best = np.concatenate([tac_best[:n_plot], vis_best[:n_plot]])
    ylo, yhi = float(np.nanmin(all_best)), float(np.nanmax(all_best))
    if yhi <= ylo: yhi = ylo + 1e-6
    pad = 0.05 * (yhi - ylo)
    ax_sel.set_ylim(ylo - pad, yhi + pad)
    ax_sel.set_xlim(-0.5, n_plot - 0.5)
    ax_sel.set_xlabel("Decision", fontsize=9)
    ax_sel.set_ylabel("Max reward over samples", fontsize=9)
    ax_sel.set_title("Best-sample reward over time", fontsize=10)
    ax_sel.legend(fontsize=8, loc="best")
    ax_sel.grid(True, alpha=0.3)

    # Shade region of current decision
    ax_sel.axvline(x=float(t), color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    # ── Row 1: std curves + source log ────────────────────────────────────
    ax_tac_std = fig.add_subplot(gs[1, 0])
    xs_std = np.arange(t + 1, dtype=float)
    ax_tac_std.plot(xs_std, tac_std[:t + 1], color="seagreen", linewidth=1.8,
                    marker="o", markersize=4)
    ax_tac_std.axhline(y=tactile_std_threshold, color="tomato", linewidth=1.5,
                       linestyle="--", label=f"threshold={tactile_std_threshold:.4f}")
    ax_tac_std.scatter([float(t)], [float(tac_std[t])], color="seagreen", s=80,
                       edgecolors="white", linewidths=1.5, zorder=5)
    tac_std_all = tac_std[:n_plot]
    ylo_ts = min(float(np.nanmin(tac_std_all)), tactile_std_threshold)
    yhi_ts = max(float(np.nanmax(tac_std_all)), tactile_std_threshold)
    pad_ts = 0.08 * (yhi_ts - ylo_ts + 1e-9)
    ax_tac_std.set_ylim(ylo_ts - pad_ts, yhi_ts + pad_ts)
    ax_tac_std.set_xlim(-0.5, n_plot - 0.5)
    ax_tac_std.set_xlabel("Decision", fontsize=9)
    ax_tac_std.set_ylabel("Std", fontsize=9)
    ax_tac_std.set_title(f"Tactile reward std  (cur={tac_std[t]:.4f})", fontsize=10)
    ax_tac_std.legend(fontsize=8)
    ax_tac_std.grid(True, alpha=0.3)

    ax_vis_std = fig.add_subplot(gs[1, 1])
    ax_vis_std.plot(xs_std, vis_std[:t + 1], color="steelblue", linewidth=1.8,
                    marker="o", markersize=4)
    ax_vis_std.axhline(y=vision_std_threshold, color="tomato", linewidth=1.5,
                       linestyle="--", label=f"threshold={vision_std_threshold:.4f}")
    ax_vis_std.scatter([float(t)], [float(vis_std[t])], color="steelblue", s=80,
                       edgecolors="white", linewidths=1.5, zorder=5)
    vis_std_all = vis_std[:n_plot]
    ylo_vs = min(float(np.nanmin(vis_std_all)), vision_std_threshold)
    yhi_vs = max(float(np.nanmax(vis_std_all)), vision_std_threshold)
    pad_vs = 0.08 * (yhi_vs - ylo_vs + 1e-9)
    ax_vis_std.set_ylim(ylo_vs - pad_vs, yhi_vs + pad_vs)
    ax_vis_std.set_xlim(-0.5, n_plot - 0.5)
    ax_vis_std.set_xlabel("Decision", fontsize=9)
    ax_vis_std.set_ylabel("Std", fontsize=9)
    ax_vis_std.set_title(f"Vision reward std  (cur={vis_std[t]:.4f})", fontsize=10)
    ax_vis_std.legend(fontsize=8)
    ax_vis_std.grid(True, alpha=0.3)

    # Selection source log
    ax_src = fig.add_subplot(gs[1, 2])
    if selection_sources is not None:
        src_colors = [_SOURCE_COLORS.get(s, "silver") for s in selection_sources[:t + 1]]
        src_vals   = [{"vision": 2, "tactile": 1, "random": 0}.get(s, -1)
                      for s in selection_sources[:t + 1]]
        ax_src.scatter(np.arange(t + 1), src_vals, c=src_colors, s=55,
                       zorder=5, edgecolors="white", linewidths=0.8)
        # Highlight current
        ax_src.scatter([float(t)], [float(src_vals[t])],
                       c=[src_colors[t]], s=140, zorder=6,
                       edgecolors="black", linewidths=1.5)
        ax_src.set_yticks([0, 1, 2])
        ax_src.set_yticklabels(["random", "tactile", "vision"], fontsize=8)
        ax_src.set_xlim(-0.5, n_plot - 0.5)
        ax_src.set_xlabel("Decision", fontsize=9)
        ax_src.set_title(f"Selection source  (cur={src})", fontsize=10)
        # Legend patches
        from matplotlib.lines import Line2D
        legend_els = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_SOURCE_COLORS[k],
                   markersize=8, label=k)
            for k in ("vision", "tactile", "random")
        ]
        ax_src.legend(handles=legend_els, fontsize=8, loc="best")
    else:
        ax_src.text(0.5, 0.5, "no source log", ha="center", va="center",
                    transform=ax_src.transAxes, fontsize=10, color="gray")
        ax_src.set_title("Selection source", fontsize=10)
    ax_src.grid(True, alpha=0.3)

    # ── Row 2: camera images ───────────────────────────────────────────────
    for col, (name, arr) in enumerate((("camera_0", r0), ("camera_1", r1), ("camera_2", r2))):
        ax_c = fig.add_subplot(gs[2, col])
        ax_c.imshow(arr)
        ax_c.set_title(f"{name}  (frame {fi})", fontsize=9)
        ax_c.axis("off")

    fig.suptitle(
        f"{stem_title}  —  decision {t + 1}/{n_plot}  |  selected={src}",
        fontsize=11, y=1.005,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Static plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_rewards_figure(tactile_r: np.ndarray, vision_r: np.ndarray,
                        out_path: Path, title: str) -> None:
    T, N = tactile_r.shape
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for row_i, (label, r2d, color) in enumerate([
        ("Tactile", tactile_r, "seagreen"),
        ("Vision",  vision_r,  "steelblue"),
    ]):
        im = axes[row_i, 0].imshow(
            r2d.T, aspect="auto", origin="lower", cmap="viridis",
            interpolation="nearest", extent=[-0.5, T - 0.5, -0.5, N - 0.5],
        )
        axes[row_i, 0].set_xlabel("Decision")
        axes[row_i, 0].set_ylabel("Sample")
        axes[row_i, 0].set_title(f"{label} reward heatmap")
        plt.colorbar(im, ax=axes[row_i, 0])

        for n in range(N):
            axes[row_i, 1].plot(np.arange(T), r2d[:, n], marker="o", markersize=2,
                                alpha=0.7, color=color)
        axes[row_i, 1].set_xlabel("Decision")
        axes[row_i, 1].set_ylabel("Reward")
        axes[row_i, 1].set_title(f"{label} reward per sample")
        axes[row_i, 1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_overview_3d(ac_log: np.ndarray, rewards_2d: np.ndarray,
                     decision_ee: np.ndarray | None, out_path: Path,
                     title: str, view_elev: float, view_azim: float,
                     reward_label: str = "reward") -> None:
    T, N = rewards_2d.shape[:2]
    r_all = np.asarray(rewards_2d, dtype=np.float64)
    rmin, rmax = float(r_all.min()), float(r_all.max())
    if rmax == rmin: rmax = rmin + 1.0
    norm = plt.Normalize(rmin, rmax)
    cmap = plt.colormaps.get_cmap("coolwarm")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    p0_list: list[np.ndarray] = []

    for t in range(T):
        p0 = (np.asarray(decision_ee[t], dtype=np.float64).reshape(3)
              if decision_ee is not None
              else np.zeros(3, dtype=np.float64))
        p0_list.append(p0)
        best_n = int(np.argmax(r_all[t]))
        for n in range(N):
            xyz = deltas_to_absolute_xyz(ac_log[t, n, :, :3], p0)
            color = cmap(norm(r_all[t, n]))
            lw = 2.2 if n == best_n else 0.6
            alpha = 0.9 if n == best_n else 0.25
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=lw, alpha=alpha)
        ax.scatter([p0[0]], [p0[1]], [p0[2]], color="black", s=40, zorder=5, depthshade=False)

    if len(p0_list) > 1:
        pts = np.array(p0_list)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color="black", linewidth=1.0, linestyle="--", alpha=0.5, zorder=4)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=view_elev, azim=view_azim)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.12, label=reward_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Video helpers
# ─────────────────────────────────────────────────────────────────────────────

def figure_to_bgr(fig: plt.Figure, dpi: int = 110) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return img


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot combined tactile+vision sampling logs from run_dp_tactile_vision_combined.py"
    )
    parser.add_argument("--hdf5_path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: <hdf5_dir>/plots_combined/<stem>)")
    parser.add_argument("--fps", type=float, default=4.0, help="Video FPS")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--view-elev", type=float, default=0.0)
    parser.add_argument("--view-azim", type=float, default=0.0)
    parser.add_argument("--camera-as-bgr", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--composite-dpi", type=int, default=100)
    parser.add_argument(
        "--tactile-std-threshold", type=float, default=0.001,
        help="Tactile std threshold line in the std curve panel",
    )
    parser.add_argument(
        "--vision-std-threshold", type=float, default=0.006,
        help="Vision std threshold line in the std curve panel",
    )
    args = parser.parse_args()

    path = args.hdf5_path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    out_dir = args.out_dir or (path.parent / "plots_combined" / path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays, attrs = load_hdf5(path)
    ac_log         = np.asarray(arrays["action_chunks_log"],   dtype=np.float64)
    tactile_r      = np.asarray(arrays["tactile_rewards_log"], dtype=np.float64)
    vision_r       = np.asarray(arrays["vision_rewards_log"],  dtype=np.float64)
    decision_ee    = np.asarray(arrays["decision_ee_xyz"])     if "decision_ee_xyz"    in arrays else None
    q_idx_raw      = np.asarray(arrays["policy_query_frame_idx"], dtype=np.int64).reshape(-1) \
                     if "policy_query_frame_idx" in arrays else None

    selection_sources: list[str] | None = None
    if "selection_source_log" in attrs:
        try:
            selection_sources = json.loads(attrs["selection_source_log"])
        except Exception:
            pass

    cam0 = arrays.get("camera_0")
    cam1 = arrays.get("camera_1")
    cam2 = arrays.get("camera_2")
    n_frames = int(cam0.shape[0]) if cam0 is not None and cam0.ndim >= 1 else 0

    # Align lengths
    T_ac, T_tac, T_vis = int(ac_log.shape[0]), int(tactile_r.shape[0]), int(vision_r.shape[0])
    T = min(T_ac, T_tac, T_vis)
    ac_log    = ac_log[:T]
    tactile_r = tactile_r[:T]
    vision_r  = vision_r[:T]
    if decision_ee is not None:
        decision_ee = decision_ee[:T]
    if q_idx_raw is not None and q_idx_raw.shape[0] >= T:
        q_idx = q_idx_raw[:T]
    elif n_frames > 0:
        q_idx = infer_query_frame_indices(T, n_frames)
        print("Note: policy_query_frame_idx missing/short; using evenly-spaced indices.")
    else:
        q_idx = None
    if selection_sources is not None:
        selection_sources = selection_sources[:T]

    n_plot = T if args.max_chunks is None else min(T, int(args.max_chunks))
    print(f"Plotting {n_plot} decisions from {path.name}")

    # ── Static plots ──────────────────────────────────────────────────────
    plot_rewards_figure(
        tactile_r[:n_plot], vision_r[:n_plot],
        out_dir / "rewards_over_decisions.png",
        title=path.name,
    )
    print(f"Wrote rewards_over_decisions.png")

    plot_overview_3d(
        ac_log[:n_plot], tactile_r[:n_plot],
        decision_ee[:n_plot] if decision_ee is not None else None,
        out_dir / "overview_3d_tactile.png",
        title=f"{path.name} — tactile reward overview",
        view_elev=args.view_elev, view_azim=args.view_azim,
        reward_label="tactile reward",
    )
    print(f"Wrote overview_3d_tactile.png")

    plot_overview_3d(
        ac_log[:n_plot], vision_r[:n_plot],
        decision_ee[:n_plot] if decision_ee is not None else None,
        out_dir / "overview_3d_vision.png",
        title=f"{path.name} — vision reward overview",
        view_elev=args.view_elev, view_azim=args.view_azim,
        reward_label="vision reward",
    )
    print(f"Wrote overview_3d_vision.png")

    # ── Combined video ────────────────────────────────────────────────────
    if not args.no_video:
        frames: list[np.ndarray] = []
        for t in range(n_plot):
            if (t + 1) % 5 == 0 or t == 0:
                print(f"  Rendering frame {t + 1}/{n_plot} …")
            fig = build_combined_frame(
                t_cur=t,
                n_plot=n_plot,
                ac_log=ac_log,
                tactile_r=tactile_r,
                vision_r=vision_r,
                selection_sources=selection_sources,
                q_idx=q_idx,
                decision_ee=decision_ee,
                cam0=cam0, cam1=cam1, cam2=cam2,
                n_frames=n_frames,
                tactile_std_threshold=args.tactile_std_threshold,
                vision_std_threshold=args.vision_std_threshold,
                stem_title=path.stem,
                view_elev=args.view_elev,
                view_azim=args.view_azim,
                camera_stored_as_bgr=args.camera_as_bgr,
            )
            frames.append(figure_to_bgr(fig, dpi=args.composite_dpi))

        mp4_path = out_dir / "combined_rewards.mp4"
        write_video(frames, mp4_path, args.fps)
        print(f"Wrote {mp4_path}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
