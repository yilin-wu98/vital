# Publication cleanup (2026-10-27): adapted from AnyTouch2/scripts/infer_touch_text_similarity.py.
from __future__ import annotations

"""
Compute cosine similarity between tactile video embeddings (AnyTouch) and
text embeddings (CLIP text encoder) from the same aligned 512-dim space.

Usage:
    python scripts/infer_touch_text_similarity.py \
        --video_path path/to/video.mp4 \
        --bg_image path/to/background.png \
        --text "pressing hard on a rough surface" \
        --checkpoint checkpoints/checkpoint-4frames.pth \
        --sensor digit

    # Multiple sentences (comma-separated or repeated --text):
    python scripts/infer_touch_text_similarity.py \
        --video_path path/to/video.mp4 \
        --text "pressing hard" --text "sliding gently" --text "no contact" \
        --checkpoint checkpoints/checkpoint-4frames.pth \
        --sensor digit
"""

import os
import sys
import argparse
import glob
import copy

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, CLIPConfig, CLIPModel, CLIPTokenizer


from anytouch_reward.model import TactileVideoMAE
from anytouch_reward.reward import pool_tokens

from anytouch_reward.loading import resolve_clip_dir
CLIP_LOCAL_DIR = str(resolve_clip_dir())

SENSOR_NAME_TO_ID = {
    "gelsight": 0,
    "digit": 1,
    "gelslim": 2,
    "gelsight_mini": 3,
    "duragel": 4,
    "dm": 5,
    "universal": -1,
}

OFFSET = 130.0 / 255.0
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def draw_arrows(frame, init_markers, motion, arrow_scale=1.0,
                color=(0, 255, 255), thickness=2, tip_length=0.3,
                magnitude_coloring=False):
    """Draw displacement arrows on a frame (BGR colour conventions)."""
    img = frame.copy()
    magnitudes = np.linalg.norm(motion, axis=1)
    max_mag = magnitudes.max() if magnitudes.max() > 0 else 1.0

    for i in range(len(init_markers)):
        dx, dy = motion[i]
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            continue
        sx, sy = int(init_markers[i, 0]), int(init_markers[i, 1])
        ex = int(np.clip(sx + dx * arrow_scale, 0, img.shape[1] - 1))
        ey = int(np.clip(sy + dy * arrow_scale, 0, img.shape[0] - 1))
        if magnitude_coloring:
            ratio = magnitudes[i] / max_mag
            c = (int(255 * (1 - ratio)), 0, int(255 * ratio))
        else:
            c = color
        cv2.arrowedLine(img, (sx, sy), (ex, ey), c, thickness, tipLength=tip_length)
    return img


def compute_tactile_motion_frames(
    tactile_frames_rgb: list[np.ndarray],
    arrow_scale: float = 3.0,
    magnitude_coloring: bool = True,
) -> tuple[list[np.ndarray] | None, list[float] | None]:
    """Run marker tracking on tactile frames. Returns (arrow_frames, motion_magnitudes).
    motion_magnitudes: mean displacement magnitude per frame (pixels)."""
    if not tactile_frames_rgb:
        return None, None
    try:
        from gsmini.markertracker import MarkerTracker
    except ImportError:
        print("  Optional gsmini package absent; skipping marker-motion overlay")
        return None, None

    first_bgr = cv2.cvtColor(tactile_frames_rgb[0], cv2.COLOR_RGB2BGR)

    mtracker = MarkerTracker(np.float32(first_bgr) / 255.0)
    Ox = mtracker.initial_marker_center[:, 1]
    Oy = mtracker.initial_marker_center[:, 0]
    init_markers = np.array((Ox, Oy), np.float32).T.reshape((-1, 2))
    nct = len(mtracker.initial_marker_center)

    p0 = np.array((Ox, Oy), np.float32).T.reshape((-1, 1, 2))
    old_gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)

    arrow_frames = [tactile_frames_rgb[0].copy()]
    motion_magnitudes = [0.0]

    for idx in range(1, len(tactile_frames_rgb)):
        frame_bgr = cv2.cvtColor(tactile_frames_rgb[idx], cv2.COLOR_RGB2BGR)
        new_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **LK_PARAMS)

        if np.sum(st) >= nct:
            p0 = p1.reshape(-1, 1, 2)
            old_gray = new_gray

        markers = np.squeeze(p0)
        motion = markers - init_markers
        mags = np.linalg.norm(motion, axis=1)
        motion_magnitudes.append(float(np.mean(mags)))

        vis_bgr = draw_arrows(frame_bgr, init_markers, motion,
                              arrow_scale=arrow_scale,
                              magnitude_coloring=magnitude_coloring)
        arrow_frames.append(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))

    print(f"  Computed motion arrows for {len(arrow_frames)} tactile frames")
    return arrow_frames, motion_magnitudes


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute tactile-text similarity using AnyTouch + CLIP"
    )
    p.add_argument("--video_path", type=str, required=True,
                    help="Path to a video file, directory of frames, or glob pattern")
    p.add_argument("--bg_image", type=str, default=None,
                    help="Background (no-contact) image. If omitted, the first frame is used.")
    p.add_argument("--text", type=str, action="append", required=True,
                    help="Text sentence(s) to compare against. Repeat for multiple.")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to the AnyTouch backbone checkpoint (e.g. checkpoints/checkpoint-4frames.pth)")
    p.add_argument("--sensor", type=str, default="digit",
                    choices=list(SENSOR_NAME_TO_ID.keys()))
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--frame_stride", type=int, default=1,
                    help="Subsample every N-th raw frame before building windows")
    p.add_argument("--model_size", type=str, default="base")
    p.add_argument("--clip_dir", type=str, default=CLIP_LOCAL_DIR,
                    help="Local directory with CLIP config/tokenizer (default: CLIP-B-16)")
    p.add_argument("--output_dir", type=str, default="results/touch_text_similarity")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_plot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_csv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_video", action=argparse.BooleanOptionalAction, default=True,
                    help="Generate a result video: top = animated similarity curve, bottom = current frame")
    p.add_argument("--video_fps", type=float, default=15.0,
                    help="FPS for the output result video")
    p.add_argument("--aggregate", type=str, default="cls",
                    choices=["cls", "mean_patch", "max_patch", "min_patch"],
                    help="How to aggregate tactile tokens into a single embedding. "
                         "'cls' uses the CLS token, 'mean_patch' averages all patch tokens.")
    p.add_argument("--crop_tactile", action="store_true", default=False,
                    help="If set, treat each video frame as [front(256)|wrist(256)|tactile(340)] "
                         "concatenated horizontally, crop the tactile region, and resize to 320x240.")
    p.add_argument("--arrow_scale", type=float, default=3.0,
                    help="Arrow length multiplier for tactile motion overlay (default: 3)")
    p.add_argument("--magnitude_coloring", action="store_true", default=False,
                    help="Color motion arrows by displacement magnitude (blue→red)")
    p.add_argument("--motion_threshold", type=float, default=0.5,
                    help="Threshold (pixels) for motion magnitude crossing lines in video (default: 0.5)")
    return p.parse_args()


# ── Video / frame loading ──────────────────────────────────────────────────

# Layout: [front_view (256 w) | wrist_view (256 w) | tactile (340 w)] x 256 h
CONCAT_FRONT_W = 256
CONCAT_WRIST_W = 256
CONCAT_TACTILE_W = 340
CONCAT_H = 256
TACTILE_X_START = CONCAT_FRONT_W + CONCAT_WRIST_W  # 512
TACTILE_RESIZE = (320, 240)  # (width, height)


def extract_tactile_from_concat(frame: np.ndarray) -> np.ndarray:
    """Crop the rightmost tactile region from a horizontally-concatenated frame
    and resize to 320x240."""
    # if frame.shape[0] == 3* 256:
    tactile = frame[:, TACTILE_X_START:TACTILE_X_START + CONCAT_TACTILE_W]
    # else:
        # tactile = frame[:, 224*2:]
    tactile = cv2.resize(tactile, TACTILE_RESIZE, interpolation=cv2.INTER_AREA)
    # cv2.imshow("tactile", tactile)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return tactile


def load_frames_from_video(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_frames_from_directory(dir_path: str) -> list[np.ndarray]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dir_path, ext)))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No image files in {dir_path}")
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def split_concat_frames(
    raw_frames: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split concatenated frames into tactile crops and full display frames.
    Returns (tactile_frames, display_frames)."""
    tactile_frames = [extract_tactile_from_concat(f) for f in raw_frames]
    return tactile_frames, raw_frames


# ── Model loading ──────────────────────────────────────────────────────────

def _load_backbone_checkpoint(path: str) -> dict:
    """Load the backbone checkpoint and return the raw state dict."""
    return torch.load(path, map_location="cpu", weights_only=True)


def load_tactile_model(args, ckpt):
    from anytouch_reward.loading import tactile_model_from_state
    return tactile_model_from_state(
        ckpt, args.clip_dir, num_frames=args.num_frames, stride=args.stride,
        model_size=args.model_size,
    )


def load_clip_text_encoder(args, ckpt, device):
    from anytouch_reward.loading import text_model_from_state
    return text_model_from_state(ckpt, args.clip_dir, device)


# ── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_frames(
    frames: list[np.ndarray],
    bg_image: np.ndarray | None,
    num_frames: int,
    frame_stride: int,
) -> list[torch.Tensor]:
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    if bg_image is None:
        bg_image = frames[0]

    bg_tensor = to_tensor(Image.fromarray(bg_image).convert("RGB"))

    subsampled = frames[::frame_stride]

    windows = []
    for start in range(0, len(subsampled) - num_frames + 1):
        window_frames = subsampled[start : start + num_frames]
        tensors = [to_tensor(Image.fromarray(f).convert("RGB")) for f in window_frames]
        tensors.append(bg_tensor)
        stacked = torch.stack(tensors, dim=0)
        bg = stacked[-1:]
        imgs = stacked[:-1]
        imgs = imgs - bg + OFFSET
        imgs = torch.clamp(imgs, 0.0, 1.0)
        imgs = transform(imgs)
        windows.append(imgs)

    return windows


# ── Embedding extraction ──────────────────────────────────────────────────

@torch.no_grad()
def extract_tactile_embeddings(
    model: TactileVideoMAE,
    windows: list[torch.Tensor],
    sensor_id: int,
    device: torch.device,
    aggregate: str = "cls",
    batch_size: int = 32,
) -> np.ndarray:
    """Returns (N, 512) array of L2-normalized tactile embeddings."""
    model.eval()
    all_embeds = []

    for i in range(0, len(windows), batch_size):
        batch = torch.stack(windows[i : i + batch_size]).to(device)  # (B, T, C, H, W)
        sensors = torch.full((batch.shape[0],), sensor_id, dtype=torch.long, device=device)

        # get_cls=False returns all projected tokens (B, 398, 512)
        outputs = model(batch, sensor_type=sensors, probe=False, get_cls=False)

        emb = pool_tokens(outputs, aggregate)
        all_embeds.append(emb.cpu().float().numpy())

    return np.concatenate(all_embeds, axis=0)


@torch.no_grad()
def extract_text_embeddings(
    clip_model: CLIPModel,
    tokenizer: CLIPTokenizer,
    texts: list[str],
    device: torch.device,
) -> np.ndarray:
    """Returns (T, 512) array of L2-normalized text embeddings."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    text_emb = clip_model.get_text_features(**inputs)  # (T, 512)
    text_emb = F.normalize(text_emb, dim=-1)
    return text_emb.cpu().float().numpy()


# ── Visualization ──────────────────────────────────────────────────────────

def plot_similarity(
    sim_matrix: np.ndarray,
    texts: list[str],
    output_path: str,
    title: str = "Tactile-Text Cosine Similarity",
):
    """sim_matrix: (N_windows, N_texts)"""
    fig, ax = plt.subplots(figsize=(max(8, len(texts) * 2), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(texts)))

    for j, (text, color) in enumerate(zip(texts, colors)):
        short_label = text[:40] + ("..." if len(text) > 40 else "")
        ax.plot(sim_matrix[:, j], label=short_label, color=color, linewidth=1.5)

    ax.set_xlabel("Window index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=8, ncol=min(len(texts), 3), framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Similarity plot saved to {output_path}")


def plot_motion_magnitude(
    motion_magnitudes: list[float],
    output_path: str,
    title: str = "Marker Motion Magnitude",
):
    """Plot mean marker displacement magnitude over time (frame index)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(motion_magnitudes, color="#2e86ab", linewidth=1.5)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean displacement (pixels)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Motion magnitude plot saved to {output_path}")


def save_similarity_csv(
    sim_matrix: np.ndarray,
    texts: list[str],
    output_path: str,
):
    header = "window_idx," + ",".join(f'"{t}"' for t in texts)
    rows = [header]
    for i, row in enumerate(sim_matrix):
        vals = ",".join(f"{v:.6f}" for v in row)
        rows.append(f"{i},{vals}")
    with open(output_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Similarity CSV saved to {output_path}")


def _fig_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
    return img


def _find_motion_crossings(motion_magnitudes: list[float], threshold: float = 0.5) -> tuple[list[int], list[int]]:
    """Find frame indices where motion magnitude crosses threshold.
    Returns (rise_frames, drop_frames). Rise: below -> above. Drop: above -> below."""
    if not motion_magnitudes or len(motion_magnitudes) < 2:
        return [], []
    rise_frames = []
    drop_frames = []
    for i in range(1, len(motion_magnitudes)):
        prev, curr = motion_magnitudes[i - 1], motion_magnitudes[i]
        if prev < threshold <= curr:
            rise_frames.append(i)
        elif prev >= threshold > curr:
            drop_frames.append(i)
    return rise_frames, drop_frames


def generate_result_video(
    sim_matrix: np.ndarray,
    texts: list[str],
    raw_frames: list[np.ndarray],
    num_frames: int,
    frame_stride: int,
    output_path: str,
    fps: float = 15.0,
    title: str = "Tactile-Text Cosine Similarity",
    motion_arrow_frames: list[np.ndarray] | None = None,
    motion_magnitudes: list[float] | None = None,
    motion_threshold: float = 0.5,
):
    """
    Render a video: top = similarity curve animated up to current step,
    bottom = current raw frame (optionally with motion-arrow panel appended).
    Optional vertical lines at motion magnitude 0.5 crossings (rise/drop).
    """
    import matplotlib
    matplotlib.use("Agg")

    n_windows = sim_matrix.shape[0]
    n_texts = sim_matrix.shape[1]

    subsampled_indices = list(range(0, len(raw_frames), frame_stride))

    rise_x = []
    drop_x = []
    if motion_magnitudes is not None:
        rise_frames, drop_frames = _find_motion_crossings(motion_magnitudes, motion_threshold)
        for f in rise_frames:
            t = (f - (num_frames - 1) * frame_stride) / frame_stride
            if 0 <= t < n_windows:
                rise_x.append(t)
        for f in drop_frames:
            t = (f - (num_frames - 1) * frame_stride) / frame_stride
            if 0 <= t < n_windows:
                drop_x.append(t)

    sim_min = sim_matrix.min() - 0.02
    sim_max = sim_matrix.max() + 0.02

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_texts, 3)))

    sample_frame = raw_frames[0]
    fh, fw = sample_frame.shape[:2]

    if motion_arrow_frames is not None:
        ah, aw = motion_arrow_frames[0].shape[:2]
        arrow_display_w = int(aw * fh / ah)
        combined_fw = fw + arrow_display_w
    else:
        combined_fw = fw
        arrow_display_w = 0

    dpi = 120
    plot_w_px = max(combined_fw, 768)
    plot_w_inches = plot_w_px / dpi
    plot_h_inches = 5.0
    plot_h_px = int(plot_h_inches * dpi)

    frame_display_h = int(plot_w_px * fh / combined_fw)
    video_w = plot_w_px
    video_h = plot_h_px + frame_display_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))

    print(f"  Rendering result video ({n_windows} frames, {video_w}x{video_h}) ...")

    for t in range(n_windows):
        # --- top: similarity curve up to step t ---
        fig, ax = plt.subplots(figsize=(plot_w_inches, plot_h_inches), dpi=dpi)
        for j in range(n_texts):
            short_label = texts[j][:50] + ("..." if len(texts[j]) > 50 else "")
            ax.plot(
                range(t + 1),
                sim_matrix[: t + 1, j],
                color=colors[j],
                linewidth=1.8,
                label=short_label,
            )
            ax.scatter([t], [sim_matrix[t, j]], color=colors[j], s=30, zorder=5)

        ax.set_xlim(0, max(n_windows - 1, 1))
        ax.set_ylim(sim_min, sim_max)
        for x in rise_x:
            if x <= t:
                ax.axvline(x=x, color="green", linestyle="--", linewidth=1.2, alpha=0.8)
        for x in drop_x:
            if x <= t:
                ax.axvline(x=x, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Window index", fontsize=9)
        ax.set_ylabel("Cosine similarity", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  fontsize=7, ncol=min(n_texts, 3), framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0.1, 1, 1])

        plot_img = _fig_to_array(fig)
        plt.close(fig)

        plot_img = cv2.resize(plot_img, (plot_w_px, plot_h_px), interpolation=cv2.INTER_AREA)

        # --- bottom: current raw video frame (+ optional arrow panel) ---
        last_frame_in_window = min(
            subsampled_indices[t + num_frames - 1] if (t + num_frames - 1) < len(subsampled_indices)
            else len(raw_frames) - 1,
            len(raw_frames) - 1,
        )
        frame_rgb = raw_frames[last_frame_in_window]

        if motion_arrow_frames is not None:
            arrow_rgb = motion_arrow_frames[last_frame_in_window]
            arrow_resized = cv2.resize(arrow_rgb, (arrow_display_w, fh),
                                       interpolation=cv2.INTER_AREA)
            frame_rgb = np.hstack([frame_rgb, arrow_resized])

        frame_resized = cv2.resize(frame_rgb, (video_w, frame_display_h),
                                   interpolation=cv2.INTER_AREA)

        combined = np.vstack([plot_img, frame_resized])

        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"  Result video saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading backbone checkpoint...")
    ckpt = _load_backbone_checkpoint(args.checkpoint)

    tactile_model = load_tactile_model(args, ckpt)
    tactile_model = tactile_model.to(device).eval()

    clip_model, tokenizer = load_clip_text_encoder(args, ckpt, device)

    sensor_id = SENSOR_NAME_TO_ID[args.sensor]
    texts = args.text

    print(f"\nText queries ({len(texts)}):")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t}")

    # encode text once
    text_embeds = extract_text_embeddings(clip_model, tokenizer, texts, device)
    print(f"Text embeddings shape: {text_embeds.shape}")

    # resolve video sources
    if os.path.isdir(args.video_path):
        video_sources = [args.video_path]
        is_dir = True
    else:
        video_sources = sorted(glob.glob(args.video_path))
        is_dir = False

    if not video_sources:
        raise FileNotFoundError(f"No files matched: {args.video_path}")

    bg_image = None
    if args.bg_image is not None:
        bg_img_raw = np.array(Image.open(args.bg_image).convert("RGB"))
        if args.crop_tactile:
            bg_image = extract_tactile_from_concat(bg_img_raw)

        else:
            bg_image = bg_img_raw

    for src in video_sources:
        name = os.path.splitext(os.path.basename(src))[0]
        if is_dir:
            name = os.path.basename(src.rstrip("/"))
        print(f"\nProcessing: {src}")

        if os.path.isdir(src):
            raw_frames = load_frames_from_directory(src)
        else:
            raw_frames = load_frames_from_video(src)

        if not raw_frames:
            raise ValueError(f"No readable frames in {src}")
        print(f"  Loaded {len(raw_frames)} frames (shape: {raw_frames[0].shape})")

        if args.crop_tactile:
            tactile_frames, display_frames = split_concat_frames(raw_frames)
            print(f"  Cropped tactile region -> {tactile_frames[0].shape}")
            motion_arrow_frames, motion_magnitudes = compute_tactile_motion_frames(
                tactile_frames,
                arrow_scale=args.arrow_scale,
                magnitude_coloring=args.magnitude_coloring,
            )
            if motion_arrow_frames is not None:
                print(f"  Motion arrow frames shape: {motion_arrow_frames[0].shape}")
        else:
            tactile_frames = raw_frames
            display_frames = raw_frames
            motion_arrow_frames = None
            motion_magnitudes = None

        if len(tactile_frames) < args.num_frames:
            print(f"  Skipping: need >= {args.num_frames} frames, got {len(tactile_frames)}")
            continue

        windows = preprocess_frames(tactile_frames, bg_image, args.num_frames, args.frame_stride)
        if not windows:
            raise ValueError("Not enough frames after applying --frame_stride")
        print(f"  Built {len(windows)} sliding windows")

        touch_embeds = extract_tactile_embeddings(
            tactile_model, windows, sensor_id, device, args.aggregate
        )
        print(f"  Tactile embeddings shape: {touch_embeds.shape}")

        # cosine similarity: (N_windows, N_texts)
        sim_matrix = touch_embeds @ text_embeds.T
        print(f"  Similarity matrix shape: {sim_matrix.shape}")

        # per-window best matching text
        best_idx = sim_matrix.argmax(axis=1)
        mean_sim = sim_matrix.mean(axis=0)
        print("  Mean similarity per text:")
        for j, t in enumerate(texts):
            print(f"    [{j}] {t}: {mean_sim[j]:.4f}")

        if len(texts) > 1:
            overall_best = mean_sim.argmax()
            print(f"  Best overall match: [{overall_best}] \"{texts[overall_best]}\" "
                  f"(mean cosine = {mean_sim[overall_best]:.4f})")

        if args.save_plot:
            plot_similarity(
                sim_matrix, texts,
                os.path.join(args.output_dir, f"{name}_similarity.png"),
                title=f"Tactile-Text Similarity — {name}",
            )

        if motion_magnitudes is not None:
            plot_motion_magnitude(
                motion_magnitudes,
                os.path.join(args.output_dir, f"{name}_motion_magnitude.png"),
                title=f"Marker Motion Magnitude — {name}",
            )

        if args.save_csv:
            save_similarity_csv(
                sim_matrix, texts,
                os.path.join(args.output_dir, f"{name}_similarity.csv"),
            )

        if args.save_video:
            generate_result_video(
                sim_matrix, texts, display_frames,
                num_frames=args.num_frames,
                frame_stride=args.frame_stride,
                output_path=os.path.join(args.output_dir, f"{name}_result.mp4"),
                fps=args.video_fps,
                title=f"Tactile-Text Similarity — {name}",
                motion_arrow_frames=motion_arrow_frames,
                motion_magnitudes=motion_magnitudes,
                motion_threshold=args.motion_threshold,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
