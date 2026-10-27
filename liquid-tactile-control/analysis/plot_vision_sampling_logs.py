#!/usr/bin/env python3
"""
Plot vision-focused sampling logs (e.g. run_dp_tactile_sampling_vision_robometer.py,
run_dp_model_tactile_sampling_vision.py).

HDF5 ``data/`` group expected:
  - action_chunks_log  (T, N, H, 7)
  - rewards_log        (T, N) or (T, N, W) — W>1 is averaged for 3D coloring
  - decision_ee_xyz    (T, 3) optional; else --base-xyz or zeros for delta integration
  - policy_query_frame_idx (T,) optional; else inferred from camera frame count
  - camera_0, camera_1, camera_2 — image time series for composite panels

If ``len(rewards_log) < len(action_chunks_log)`` (e.g. first queries had no WM reward),
the last R rows of actions / q_idx / decision_ee are aligned with rewards (suffix trim).

Outputs under ``--out-dir`` (default: ``<hdf5_parent>/plots_vision_sampling/<stem>/``):
  - rewards_over_decisions.png
  - rewards_samples_vs_wm_steps.png (only if rewards_log is 3D)
  - chunks_3d/chunk_XXXX.png — per-decision 3D EE fan, color = reward
  - chunks_composite/chunk_XXXX.png — same 3D + camera_0/1/2 at query frame
  - overview_3d.png — all decisions on one 3D axes
  - trajectory_chunks.mp4 (unless --no-video)
  - reward_variance_cameras.mp4 — reward variance vs decision (curve grows each frame) +
    camera_0/1/2 at the current query frame (unless --no-video or --no-reward-variance-video)
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


def load_hdf5(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        g = f["data"]
        out = {}
        for k in g.keys():
            out[k] = np.asarray(g[k][()])
        if "action_chunks_log" not in out:
            raise KeyError(f"{path} missing data/action_chunks_log")
        if "rewards_log" not in out:
            raise KeyError(f"{path} missing data/rewards_log")
        return out


def rewards_to_2d(rewards: np.ndarray) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim == 2:
        return r
    if r.ndim == 3:
        return r.mean(axis=-1)
    raise ValueError(f"rewards_log shape {r.shape} not supported (expect 2D or 3D)")


def infer_query_frame_indices(n_queries: int, n_cam_frames: int) -> np.ndarray:
    if n_cam_frames <= 0:
        return np.zeros(n_queries, dtype=np.int64)
    if n_queries == 1:
        return np.array([n_cam_frames - 1], dtype=np.int64)
    return np.round(np.linspace(0, n_cam_frames - 1, n_queries)).astype(np.int64)


def subset_logs_by_min_query_frame(
    min_frame: int,
    ac_log: np.ndarray,
    rewards_log: np.ndarray,
    q_idx: np.ndarray,
    decision_ee: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    T = int(ac_log.shape[0])
    if q_idx.shape[0] != T:
        raise ValueError("q_idx length must match action_chunks_log")
    if decision_ee is not None and decision_ee.shape[0] != T:
        raise ValueError("decision_ee_xyz length must match action_chunks_log for filtering")

    keep = q_idx >= min_frame
    if not np.any(keep):
        raise ValueError(
            f"No rows left after requiring policy_query_frame_idx >= {min_frame}"
        )
    ac_log = np.asarray(ac_log)[keep]
    rewards_log = np.asarray(rewards_log)[keep]
    q_idx = q_idx[keep]
    if decision_ee is not None:
        decision_ee = np.asarray(decision_ee)[keep]
    return ac_log, rewards_log, q_idx, decision_ee


def align_action_chunks_to_rewards(
    ac_log: np.ndarray,
    rewards_log: np.ndarray,
    q_idx: np.ndarray | None,
    decision_ee: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Suffix-align when more policy queries were logged than reward rows (common with robometer)."""
    Ta, Tr = int(ac_log.shape[0]), int(rewards_log.shape[0])
    if Ta == Tr:
        return ac_log, rewards_log, q_idx, decision_ee
    if Ta > Tr:
        print(
            f"Warning: action_chunks_log has {Ta} rows but rewards_log has {Tr}; "
            f"using the last {Tr} rows of actions (and q_idx / decision_ee when lengths match Ta) "
            "to match rewards."
        )
        ac_log = np.asarray(ac_log)[-Tr:]
        rewards_log = np.asarray(rewards_log)
        if q_idx is not None:
            q_arr = np.asarray(q_idx, dtype=np.int64).reshape(-1)
            if q_arr.shape[0] == Ta:
                q_idx = q_arr[-Tr:]
            elif q_arr.shape[0] == Tr:
                q_idx = q_arr
            else:
                print(
                    f"Warning: policy_query_frame_idx length {q_arr.shape[0]}; "
                    "will re-infer indices after trim."
                )
                q_idx = None
        if decision_ee is not None:
            d = np.asarray(decision_ee)
            if d.shape[0] == Ta:
                decision_ee = d[-Tr:]
            elif d.shape[0] == Tr:
                decision_ee = d
            else:
                raise ValueError(
                    f"decision_ee_xyz length {d.shape[0]} does not match action_chunks {Ta} or rewards {Tr}"
                )
        return ac_log, rewards_log, q_idx, decision_ee
    raise ValueError(
        f"rewards_log rows ({Tr}) > action_chunks_log rows ({Ta}); cannot align."
    )


def to_rgb_camera(img: np.ndarray | None, stored_as_bgr: bool = False) -> np.ndarray:
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


def apply_3d_view_faces_pos_y(ax, elev: float = 0.0, azim: float = 0.0):
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def deltas_to_absolute_xyz(deltas: np.ndarray, p0: np.ndarray) -> np.ndarray:
    d = np.asarray(deltas, dtype=np.float64)
    origin = np.asarray(p0, dtype=np.float64).reshape(1, 3)
    steps = np.cumsum(d[:, :3] * ACTION_SCALE, axis=0)
    return np.vstack([origin, origin + steps])


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
    if reward_norm is None:
        rmin, rmax = reward_norm_per_chunk(r)
    else:
        rmin, rmax = reward_norm
    norm = plt.Normalize(rmin, rmax)
    cmap = plt.colormaps["coolwarm"] if hasattr(plt, "colormaps") else plt.get_cmap("coolwarm")
    p0_arr = np.asarray(p0, dtype=np.float64).reshape(3)

    ax.scatter(
        [p0_arr[0]], [p0_arr[1]], [p0_arr[2]],
        color="black", s=60, zorder=5, depthshade=False,
    )
    ax.text(
        float(p0_arr[0]), float(p0_arr[1]), float(p0_arr[2]),
        f"EEF\n({p0_arr[0]:.3f},{p0_arr[1]:.3f},{p0_arr[2]:.3f})",
        fontsize=6, color="black", zorder=6,
    )

    for n in range(N):
        abs_xyz = deltas_to_absolute_xyz(ac[n, :, :3], p0_arr)
        color = cmap(norm(r[n]))
        ax.plot(
            abs_xyz[:, 0], abs_xyz[:, 1], abs_xyz[:, 2],
            color=color, linewidth=2.0, alpha=0.9,
        )
        ax.text(
            float(abs_xyz[-1, 0]), float(abs_xyz[-1, 1]), float(abs_xyz[-1, 2]),
            f"{r[n]:.3f}",
            fontsize=7,
            color=color,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    apply_3d_view_faces_pos_y(ax, elev=view_elev, azim=view_azim)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def plot_chunk_3d(
    action_chunk: np.ndarray,
    rewards_per_sample: np.ndarray,
    p0: np.ndarray,
    out_path: Path,
    title: str,
    reward_norm: tuple[float, float] | None = None,
    view_elev: float = 0.0,
    view_azim: float = 90.0,
):
    r = np.asarray(rewards_per_sample, dtype=np.float64).reshape(-1)
    rn = reward_norm if reward_norm is not None else reward_norm_per_chunk(r)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sm = draw_chunk_3d_on_axes(ax, action_chunk, rewards_per_sample, p0, rn, view_elev, view_azim)
    ax.set_title(title)
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.12, label="reward")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview_3d(
    ac_log: np.ndarray,
    rewards_2d: np.ndarray,
    decision_ee: np.ndarray | None,
    out_path: Path,
    title: str,
    view_elev: float = 0.0,
    view_azim: float = 0.0,
):
    ac = np.asarray(ac_log, dtype=np.float64)
    T, N, H = ac.shape[:3]
    r_all = np.asarray(rewards_2d, dtype=np.float64)
    rmin, rmax = float(r_all.min()), float(r_all.max())
    if rmax == rmin:
        rmax = rmin + 1.0
    norm = plt.Normalize(rmin, rmax)
    cmap = plt.colormaps["coolwarm"] if hasattr(plt, "colormaps") else plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    p0_list: list[np.ndarray] = []

    for t in range(T):
        if decision_ee is not None:
            p0 = np.asarray(decision_ee[t], dtype=np.float64).reshape(3)
        else:
            if t == 0:
                p0 = np.zeros(3, dtype=np.float64)
            else:
                prev_ac = ac[t - 1, int(np.argmax(r_all[t - 1])), :, :3]
                p0 = p0_list[-1] + prev_ac.sum(axis=0)
        p0_list.append(p0)

        best_n = int(np.argmax(r_all[t]))
        for n in range(N):
            abs_xyz = deltas_to_absolute_xyz(ac[t, n, :, :3], p0)
            color = cmap(norm(r_all[t, n]))
            lw = 2.5 if n == best_n else 0.8
            alpha = 0.9 if n == best_n else 0.3
            ax.plot(abs_xyz[:, 0], abs_xyz[:, 1], abs_xyz[:, 2],
                    color=color, linewidth=lw, alpha=alpha)

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
    apply_3d_view_faces_pos_y(ax, elev=view_elev, azim=view_azim)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.12, label="reward (global)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rewards_figure(
    rewards_2d: np.ndarray,
    out_path: Path,
    title: str | None = None,
):
    T, N = rewards_2d.shape
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.2, 1.0])
    im = axes[0].imshow(
        rewards_2d.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=[-0.5, T - 0.5, -0.5, N - 0.5],
    )
    axes[0].set_xlabel("Decision index (policy query)")
    axes[0].set_ylabel("Sample index")
    axes[0].set_title("Reward per sample (mean over WM steps when rewards_log is 3D)")
    plt.colorbar(im, ax=axes[0], label="reward")

    for n in range(N):
        axes[1].plot(np.arange(T), rewards_2d[:, n], marker="o", markersize=3, alpha=0.85, label=f"s{n}")
    axes[1].set_xlabel("Decision index")
    axes[1].set_ylabel("reward")
    axes[1].set_title("Reward vs decision index (one curve per sample)")
    axes[1].grid(True, alpha=0.3)
    if N <= 12:
        axes[1].legend(fontsize=7, ncol=2, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rewards_wm_steps(
    rewards_3d: np.ndarray,
    out_path: Path,
    title: str | None = None,
    decision_index: int = 0,
):
    r = np.asarray(rewards_3d, dtype=np.float64)
    if r.ndim != 3:
        return
    T, N, W = r.shape
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    di = min(max(decision_index, 0), T - 1)
    im0 = axes[0].imshow(
        r[di].T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        extent=[-0.5, N - 0.5, -0.5, W - 0.5],
    )
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("WM / horizon step")
    axes[0].set_title(f"Reward — decision {di}")
    plt.colorbar(im0, ax=axes[0], label="reward")

    r_mean = r.mean(axis=0)
    im1 = axes[1].imshow(
        r_mean.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        extent=[-0.5, N - 0.5, -0.5, W - 0.5],
    )
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("WM / horizon step")
    axes[1].set_title("Reward — mean over all decisions")
    plt.colorbar(im1, ax=axes[1], label="reward")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _panel_target_hw(rgb_list: list[np.ndarray], target_h: int = 200) -> tuple[int, int]:
    """Pick (W,H) so all panels share height ``target_h`` from the widest aspect ratio."""
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


def build_reward_variance_progress_figure(
    t_cur: int,
    n_plot: int,
    var_per_decision: np.ndarray,
    q_idx: np.ndarray | None,
    cam0: np.ndarray | None,
    cam1: np.ndarray | None,
    cam2: np.ndarray | None,
    n_frames: int,
    stem_title: str,
    camera_stored_as_bgr: bool,
) -> plt.Figure:
    """
    One frame of the variance video: cumulative Var(reward over samples) vs decision
    up to ``t_cur``, plus wrist / front / tactile images at the query frame for ``t_cur``.
    """
    t_cur = int(np.clip(t_cur, 0, n_plot - 1))
    var_per_decision = np.asarray(var_per_decision, dtype=np.float64).reshape(-1)

    fig = plt.figure(figsize=(12.5, 6.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.25], hspace=0.32, wspace=0.12)

    ax_curve = fig.add_subplot(gs[0, :])
    xs = np.arange(t_cur + 1, dtype=np.float64)
    ys = var_per_decision[: t_cur + 1]
    ax_curve.plot(xs, ys, color="steelblue", linewidth=2.2, marker="o", markersize=5, zorder=2)
    ax_curve.scatter(
        [float(t_cur)],
        [float(var_per_decision[t_cur])],
        color="orangered",
        s=96,
        zorder=5,
        edgecolors="white",
        linewidths=1.2,
        label="current decision",
    )
    v_full = var_per_decision[:n_plot]
    ymin, ymax = float(np.min(v_full)), float(np.max(v_full))
    if not np.isfinite(ymin):
        ymin = 0.0
    if not np.isfinite(ymax):
        ymax = 1.0
    if ymax <= ymin:
        ymax = ymin + 1e-9
    pad = 0.06 * (ymax - ymin + 1e-12)
    ax_curve.set_ylim(ymin - pad, ymax + pad)
    ax_curve.set_xlim(-0.5, max(n_plot - 1, 0) + 0.5)
    ax_curve.set_xlabel("Decision index")
    ax_curve.set_ylabel("Var(reward over samples)")
    ax_curve.set_title(
        f"Reward variance vs decision — frame {t_cur + 1} / {n_plot}  ({stem_title})",
        fontsize=11,
    )
    ax_curve.grid(True, alpha=0.35)
    ax_curve.legend(loc="upper right", fontsize=9)

    if q_idx is not None and n_frames > 0:
        fi = int(np.clip(int(q_idx[t_cur]), 0, n_frames - 1))
    elif n_frames > 0:
        fi = int(infer_query_frame_indices(n_plot, n_frames)[t_cur])
    else:
        fi = 0

    rgb0 = cam0[fi] if cam0 is not None and n_frames > 0 else None
    rgb1 = cam1[fi] if cam1 is not None and n_frames > 0 else None
    rgb2 = cam2[fi] if cam2 is not None and n_frames > 0 else None

    r0 = to_rgb_camera(rgb0, stored_as_bgr=camera_stored_as_bgr)
    r1 = to_rgb_camera(rgb1, stored_as_bgr=camera_stored_as_bgr)
    r2 = to_rgb_camera(rgb2, stored_as_bgr=camera_stored_as_bgr)
    tw, th = _panel_target_hw([r0, r1, r2], target_h=220)
    r0 = resize_rgb_to_wh(r0, tw, th)
    r1 = resize_rgb_to_wh(r1, tw, th)
    r2 = resize_rgb_to_wh(r2, tw, th)

    for col, (name, arr) in enumerate(
        (("camera_0", r0), ("camera_1", r1), ("camera_2", r2)),
    ):
        ax_c = fig.add_subplot(gs[1, col])
        ax_c.imshow(arr)
        ax_c.set_title(f"{name}  (frame {fi})", fontsize=10)
        ax_c.axis("off")

    fig.tight_layout()
    return fig


def build_vision_composite_figure(
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
    r2 = to_rgb_camera(rgb2, stored_as_bgr=camera_stored_as_bgr)
    tw, th = _panel_target_hw([r0, r1, r2], target_h=200)
    r0 = resize_rgb_to_wh(r0, tw, th)
    r1 = resize_rgb_to_wh(r1, tw, th)
    r2 = resize_rgb_to_wh(r2, tw, th)

    fig = plt.figure(figsize=(14.5, 4.2))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1.2, 1, 1, 1], wspace=0.22)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    sm = draw_chunk_3d_on_axes(
        ax3d, action_chunk, rewards_vec, p0, None, view_elev, view_azim,
    )
    ax3d.set_title("EE traj (reward)", fontsize=10)
    fig.colorbar(sm, ax=ax3d, shrink=0.82, pad=0.02, label="reward")

    for col, (name, arr) in enumerate(
        (("camera_0", r0), ("camera_1", r1), ("camera_2", r2)), start=1
    ):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(arr)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


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


def main():
    parser = argparse.ArgumentParser(description="Plot vision sampling logs (3D + cameras)")
    parser.add_argument("--hdf5_path", type=Path, required=True, help="Path to traj_XXXX.hdf5")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <hdf5_dir>/plots_vision_sampling/<stem>)",
    )
    parser.add_argument(
        "--base-xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="EE origin when decision_ee_xyz is missing",
    )
    parser.add_argument("--max-chunks", type=int, default=None, help="Max decisions to render")
    parser.add_argument(
        "--wm-detail-decision",
        type=int,
        default=0,
        help="Decision index for rewards_samples_vs_wm_steps.png left panel",
    )
    parser.add_argument("--fps", type=float, default=6.0, help="FPS for trajectory_chunks.mp4")
    parser.add_argument("--no-video", action="store_true", help="Skip all MP4 outputs")
    parser.add_argument(
        "--no-reward-variance-video",
        action="store_true",
        help="Skip reward_variance_cameras.mp4 (still writes trajectory_chunks if composites run)",
    )
    parser.add_argument("--no-composite", action="store_true", help="Skip composite PNGs and video")
    parser.add_argument("--only-3d", action="store_true", help="Only chunks_3d/ + overview (no composites / reward heatmaps)")
    parser.add_argument("--composite-dpi", type=int, default=100, help="DPI for composite rasterization")
    parser.add_argument("--view-elev", type=float, default=0.0, help="3D view_init elev")
    parser.add_argument("--view-azim", type=float, default=0.0, help="3D view_init azimuth")
    parser.add_argument(
        "--camera-as-bgr",
        action="store_true",
        help="Treat camera_0/1 as BGR and convert to RGB",
    )
    parser.add_argument(
        "--min-query-frame",
        type=int,
        default=4,
        help="Drop decisions with policy_query_frame_idx < this (0 = keep all)",
    )
    args = parser.parse_args()

    path = args.hdf5_path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = path.parent / "plots_vision_sampling" / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_hdf5(path)
    ac_log = np.asarray(data["action_chunks_log"])
    rewards_log = np.asarray(data["rewards_log"])
    decision_ee = data.get("decision_ee_xyz")
    if decision_ee is not None:
        decision_ee = np.asarray(decision_ee)

    ac_log, rewards_log, q_idx, decision_ee = align_action_chunks_to_rewards(
        ac_log, rewards_log,
        data.get("policy_query_frame_idx"),
        decision_ee,
    )
    if q_idx is not None:
        q_idx = np.asarray(q_idx, dtype=np.int64).reshape(-1)

    cam0 = data.get("camera_0")
    cam1 = data.get("camera_1")
    cam2 = data.get("camera_2")
    n_frames = int(cam0.shape[0]) if cam0 is not None and cam0.ndim >= 1 else 0

    T = int(ac_log.shape[0])
    if q_idx is None and n_frames > 0:
        q_idx = infer_query_frame_indices(T, n_frames)
        print(
            "Note: policy_query_frame_idx missing or dropped; "
            f"using evenly spaced indices over {n_frames} frames."
        )
    if q_idx is not None and int(q_idx.shape[0]) != T:
        print(f"Warning: q_idx length {q_idx.shape[0]} != {T}; re-inferring.")
        q_idx = infer_query_frame_indices(T, n_frames) if n_frames > 0 else np.zeros(T, dtype=np.int64)

    if (
        q_idx is not None
        and int(q_idx.shape[0]) == T
        and args.min_query_frame > 0
    ):
        try:
            ac_log, rewards_log, q_idx, decision_ee = subset_logs_by_min_query_frame(
                args.min_query_frame, ac_log, rewards_log, q_idx, decision_ee,
            )
        except ValueError as e:
            print(f"Warning: --min-query-frame not applied: {e}")
        else:
            print(
                f"Applied --min-query-frame {args.min_query_frame}: "
                f"{T - int(ac_log.shape[0])} rows dropped, {int(ac_log.shape[0])} kept."
            )
            T = int(ac_log.shape[0])

    rewards_2d = rewards_to_2d(rewards_log)
    if ac_log.shape[0] != rewards_2d.shape[0]:
        raise ValueError(
            f"Length mismatch after align: action_chunks {ac_log.shape[0]} vs rewards {rewards_2d.shape[0]}"
        )

    if decision_ee is None and args.base_xyz is None:
        print(
            "Warning: no decision_ee_xyz and no --base-xyz; "
            "using (0,0,0) as origin for 3D delta paths."
        )

    if not args.only_3d:
        plot_rewards_figure(rewards_2d, out_dir / "rewards_over_decisions.png", title=path.name)
        print(f"Wrote {out_dir / 'rewards_over_decisions.png'}")

    if not args.only_3d and rewards_log.ndim == 3:
        plot_rewards_wm_steps(
            rewards_log,
            out_dir / "rewards_samples_vs_wm_steps.png",
            title=path.name,
            decision_index=args.wm_detail_decision,
        )
        print(f"Wrote {out_dir / 'rewards_samples_vs_wm_steps.png'}")

    n_plot = T if args.max_chunks is None else min(T, args.max_chunks)
    chunk_dir = out_dir / "chunks_3d"
    chunk_dir.mkdir(exist_ok=True)
    comp_dir = out_dir / "chunks_composite"
    if not args.only_3d and not args.no_composite:
        comp_dir.mkdir(exist_ok=True)

    video_frames: list[np.ndarray] = []
    composite_warned = False

    for t in range(n_plot):
        if decision_ee is not None:
            p0 = decision_ee[t]
        elif args.base_xyz is not None:
            p0 = np.array(args.base_xyz, dtype=np.float64)
        else:
            p0 = np.zeros(3, dtype=np.float64)

        r_vec = rewards_2d[t]
        plot_chunk_3d(
            ac_log[t],
            r_vec,
            p0,
            chunk_dir / f"chunk_{t:04d}.png",
            title=f"{path.name} — decision {t}",
            reward_norm=reward_norm_per_chunk(r_vec),
            view_elev=args.view_elev,
            view_azim=args.view_azim,
        )

        if not args.only_3d and not args.no_composite and q_idx is not None and n_frames > 0:
            fi = int(np.clip(q_idx[t], 0, n_frames - 1))
            rgb0 = cam0[fi] if cam0 is not None else None
            rgb1 = cam1[fi] if cam1 is not None else None
            rgb2 = cam2[fi] if cam2 is not None else None
            fig = build_vision_composite_figure(
                ac_log[t],
                r_vec,
                p0,
                rgb0,
                rgb1,
                rgb2,
                title=f"{path.stem} — decision {t}",
                view_elev=args.view_elev,
                view_azim=args.view_azim,
                camera_stored_as_bgr=args.camera_as_bgr,
            )
            comp_path = comp_dir / f"chunk_{t:04d}.png"
            fig.savefig(comp_path, dpi=args.composite_dpi, bbox_inches="tight", pad_inches=0.08)
            if not args.no_video:
                video_frames.append(figure_to_bgr(fig, dpi=args.composite_dpi))
            else:
                plt.close(fig)
        elif not args.only_3d and not args.no_composite and not composite_warned:
            print("Skipping composite/video: missing cameras or frame alignment.")
            composite_warned = True

    print(f"Wrote {n_plot} figures under {chunk_dir}")

    plot_overview_3d(
        ac_log[:n_plot],
        rewards_2d[:n_plot],
        decision_ee[:n_plot] if decision_ee is not None else None,
        out_dir / "overview_3d.png",
        title=f"{path.name} — overview ({n_plot} decisions)",
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )
    print(f"Wrote {out_dir / 'overview_3d.png'}")

    if not args.only_3d and not args.no_composite and not args.no_video and video_frames:
        mp4_path = out_dir / "trajectory_chunks.mp4"
        write_video_cv2(video_frames, mp4_path, args.fps)
        print(f"Wrote {mp4_path}")

    # Progressive reward variance + cameras (independent of --no-composite)
    if not args.no_video and not args.no_reward_variance_video and n_plot > 0:
        ## use std instead of var
        var_per_decision = np.std(rewards_2d[:n_plot], axis=1, ddof=0)
        qv = q_idx
        if qv is None and n_frames > 0:
            qv = infer_query_frame_indices(n_plot, n_frames)
        var_frames: list[np.ndarray] = []
        for t in range(n_plot):
            fig_v = build_reward_variance_progress_figure(
                t_cur=t,
                n_plot=n_plot,
                var_per_decision=var_per_decision,
                q_idx=qv,
                cam0=cam0,
                cam1=cam1,
                cam2=cam2,
                n_frames=n_frames,
                stem_title=path.stem,
                camera_stored_as_bgr=args.camera_as_bgr,
            )
            var_frames.append(figure_to_bgr(fig_v, dpi=args.composite_dpi))
        var_mp4 = out_dir / "reward_variance_cameras.mp4"
        write_video_cv2(var_frames, var_mp4, args.fps)
        print(f"Wrote {var_mp4}")


if __name__ == "__main__":
    main()
