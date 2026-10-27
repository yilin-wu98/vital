# Publication cleanup (2026-10-27): adapted from latent-tactile/dino_wm/eval_dino_wm_tactile_k_step.py.
"""
Evaluate the active liquid-task k-step concatenation world model.

Defaults come from configs/liquid.yaml. For each raw-time anchor, use recorded
embeddings, states, and actions to predict one transition eight frames ahead.
Predictions are not fed back as history in this evaluator.

- eval_hist_mode "lpb": history at raw indices
  t_last - (num_hist-1)*frameskip, ..., t_last (LPB grid), with collapsed action blocks.

- eval_hist_mode "consecutive": dense history at raw indices
  t_last - num_hist + 1, ..., t_last (matches temporal_mode consecutive training).

Video: top row = GT, second row = GT for early frames then model predictions from the
first valid target index onward; third row = |GT - pred|.
Optional fourth row (--decode_gt_embeddings): GT embeddings decoded through the same decoder,
useful for isolating decoder reconstruction error from world-model prediction error.

Embeddings: episode_*_traj_*_gt_embeds.npz and *_pred_embeds.npz contain camera-name
keys with arrays (T, N_patches, D). Unpredicted context slots are filled from GT:
with the active H=1, P=1, frameskip=8 recipe, only indices 8: are predictions.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dino_wm.tactile_dataset import CAMERA_CONFIG, load_full_episode
from dino_wm.tactile_k_step_dataset import TactileKStepDataset, _consecutive_anchor_bounds, _ensure_patch_count
from dino_wm.utils.save_video import save_rgb_mp4
from dino_wm.dino_models import NormStats
from dino_wm.dino_models_k_step import TactileKStepTransformer
from dino_wm.dino_decoder import VQVAE

CAMERA_EMB_DIMS = {"camera_0": 768, "camera_1": 768, "camera_2": 512}
PATCH_SIDE_DINOV3 = 14


def ensure_patch_grid(embd: torch.Tensor, target_side: int) -> torch.Tensor:
    """Reshape or interpolate patch embeddings to target_side x target_side grid."""
    B, T, N, D = embd.shape
    side = int(N ** 0.5)
    if side * side != N:
        target_n = target_side * target_side
        if N >= target_n:
            embd = embd[:, :, :target_n, :]
        else:
            pad = torch.zeros(B, T, target_n - N, D, device=embd.device, dtype=embd.dtype)
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


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove _orig_mod. prefix when checkpoint was saved with torch.compile."""
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k.replace(prefix, "", 1): v for k, v in state_dict.items()}


def load_decoders(decoder_dir: str, predict_cameras: list, device: str) -> dict:
    decoders = {}
    vision_path = os.path.join(decoder_dir, "decoder_camera_0_camera_1_last.pth")
    tactile_path = os.path.join(decoder_dir, "decoder_camera_2_last.pth")
    if os.path.isfile(vision_path):
        dec = VQVAE(emb_dim=768).to(device)
        dec.load_state_dict(torch.load(vision_path, map_location=device))
        dec.eval()
        for c in predict_cameras:
            if CAMERA_EMB_DIMS.get(c, 768) == 768:
                decoders[c] = dec
    if os.path.isfile(tactile_path) and "camera_2" in predict_cameras:
        dec = VQVAE(emb_dim=512).to(device)
        dec.load_state_dict(torch.load(tactile_path, map_location=device))
        dec.eval()
        decoders["camera_2"] = dec
    return decoders


def _pad_states_to_8(states: np.ndarray) -> np.ndarray:
    if states.shape[-1] > 8:
        return states[..., :8].copy()
    if states.shape[-1] < 8:
        pad = np.zeros((*states.shape[:-1], 8 - states.shape[-1]), dtype=np.float32)
        return np.concatenate([states, pad], axis=-1)
    return states.copy()


def create_episode_videos(
    transition,
    decoders: dict,
    eval_ds: TactileKStepDataset,
    predict_cameras: list,
    condition_cameras: list,
    args,
    device: str,
    norm_stats: NormStats,
) -> dict:
    """
    Full raw trajectory: each transition uses GT history (embeddings + states) and
    collapsed action blocks. History layout is set by args.eval_hist_mode (lpb vs consecutive).
    Writes decoded predictions at raw indices t_last + k*frameskip for k = 1..num_pred.
    """
    if not decoders:
        return {}
    videos = {}
    traj_ids = eval_ds.trajectory_ids[: args.num_episode_videos]

    num_hist = args.num_hist
    num_pred = args.num_pred
    frameskip = args.frameskip

    for ep_idx, traj_id in enumerate(tqdm(traj_ids, desc="Episode videos")):
        cams_need = list(set(predict_cameras) | set(condition_cameras))
        ep_data = load_full_episode(
            args.hdf5_path,
            traj_id,
            cams_need,
            not args.no_consolidated,
            resize_to_224=True,
        )
        if "states" not in ep_data or "actions" not in ep_data:
            continue

        T_full = ep_data[f"{predict_cameras[0]}_embd"].shape[0]
        T = min(T_full, args.max_episode_len)
        eval_hist_mode = getattr(args, "eval_hist_mode", "lpb")
        if eval_hist_mode == "consecutive":
            t_min_c, t_max_c = _consecutive_anchor_bounds(T, num_hist, num_pred, frameskip)
            if t_max_c < t_min_c:
                continue
        else:
            # LPB: need subsampled hist + at least one target block
            if T <= (num_hist - 1) * frameskip + frameskip:
                continue

        # GT images
        all_gt_imgs = []
        for cam in predict_cameras:
            if f"{cam}_image" in ep_data:
                gt_img = np.asarray(ep_data[f"{cam}_image"][:T], dtype=np.float32)
            else:
                gt_img = np.zeros((T, 224, 224, 3), dtype=np.float32)
            if gt_img.ndim == 3:
                gt_img = np.stack([gt_img] * 3, axis=-1)
            if gt_img.max() > 1.0:
                gt_img = gt_img / 255.0
            all_gt_imgs.append(np.clip(gt_img, 0, 1))

        states_np = _pad_states_to_8(np.asarray(ep_data["states"][:T], dtype=np.float32))
        states = torch.tensor(states_np, dtype=torch.float32, device=device).unsqueeze(0)
        if args.normalize_states:
            states = norm_stats.normalize_states(states)

        actions_full = torch.tensor(ep_data["actions"][:T], dtype=torch.float32, device=device).unsqueeze(0)
        if actions_full.shape[-1] > 7:
            actions_full = actions_full[..., :7]
        actions_full = norm_stats.normalize_acs(actions_full)

        all_embds = {}
        for c in condition_cameras:
            emb = _ensure_patch_count(np.asarray(ep_data[f"{c}_embd"][:T], dtype=np.float32))
            emb_t = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
            all_embds[c] = ensure_patch_grid(emb_t, PATCH_SIDE_DINOV3)

        save_gt = args.save_gt_embeddings
        save_pred = args.save_pred_embeddings
        gt_emb_np = {}
        pred_emb_np = {}
        pred_valid = None
        predict_grids = {}
        if save_gt or save_pred:
            for cam in predict_cameras:
                emb = _ensure_patch_count(np.asarray(ep_data[f"{cam}_embd"][:T], dtype=np.float32))
                emb_t = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
                predict_grids[cam] = ensure_patch_grid(emb_t, PATCH_SIDE_DINOV3)
            if save_gt:
                for cam, grid in predict_grids.items():
                    gt_emb_np[cam] = grid[0].detach().cpu().numpy().astype(np.float32)
            if save_pred:
                pred_valid = np.zeros(T, dtype=np.uint8)
                for cam, grid in predict_grids.items():
                    _, _, n_p, d_cam = grid.shape
                    pred_emb_np[cam] = np.full((T, n_p, d_cam), np.nan, dtype=np.float32)

        pred_imgs = [[all_gt_imgs[i][t].copy() for t in range(T)] for i in range(len(predict_cameras))]

        # Decode GT embeddings if requested
        decoded_gt_imgs = None
        if args.decode_gt_embeddings and decoders:
            decoded_gt_imgs = [[None] * T for _ in predict_cameras]
            with torch.inference_mode():
                for i, cam in enumerate(predict_cameras):
                    if cam not in decoders:
                        decoded_gt_imgs[i] = [all_gt_imgs[i][t].copy() for t in range(T)]
                        continue
                    emb_cam = _ensure_patch_count(np.asarray(ep_data[f"{cam}_embd"][:T], dtype=np.float32))
                    emb_t = torch.tensor(emb_cam, dtype=torch.float32, device=device).unsqueeze(0)
                    emb_t = ensure_patch_grid(emb_t, PATCH_SIDE_DINOV3)
                    for t in range(T):
                        gt_emb_frame = emb_t[:, t:t+1]
                        dec_img, _ = decoders[cam](gt_emb_frame)
                        dec_img = rearrange(dec_img, "(b t) c h w -> b t c h w", t=1)
                        dec_img = dec_img[0, 0].cpu().numpy().transpose(1, 2, 0)
                        decoded_gt_imgs[i][t] = np.clip(dec_img, 0, 1)

        with torch.inference_mode():
            # t_last = raw index of last history frame; targets at t_last + k*frameskip, k=1..num_pred
            if eval_hist_mode == "consecutive":
                t_first, t_last_max = _consecutive_anchor_bounds(T, num_hist, num_pred, frameskip)
                t_last_iter = range(t_first, t_last_max + 1)
            else:
                t_first = (num_hist - 1) * frameskip
                t_last_iter = range(t_first, T - frameskip)

            for t_last in t_last_iter:
                if eval_hist_mode == "consecutive":
                    hist_idx = list(range(t_last - num_hist + 1, t_last + 1))
                else:
                    hist_idx = [t_last - (num_hist - 1 - j) * frameskip for j in range(num_hist)]
                act_parts = []
                for j in range(num_hist):
                    aj = hist_idx[j]
                    sl = actions_full[:, aj : aj + frameskip, :]
                    act_parts.append(sl.reshape(1, 1, frameskip * 7))
                act_collapsed = torch.cat(act_parts, dim=1)

                cond_inputs = {c: all_embds[c][:, hist_idx, :, :] for c in condition_cameras}
                inp_states = states[:, hist_idx, :]

                preds, _pred_state_out = transition(cond_inputs, inp_states, act_collapsed)

                for k in range(num_pred):
                    tgt = t_last + (k + 1) * frameskip
                    if tgt >= T:
                        continue
                    if save_pred:
                        pred_valid[tgt] = 1
                    for i, cam in enumerate(predict_cameras):
                        pred_emb = preds[cam][:, k : k + 1]
                        if save_pred and cam in pred_emb_np:
                            pred_emb_np[cam][tgt] = pred_emb[0, 0].detach().cpu().numpy().astype(np.float32)
                        if cam not in decoders:
                            continue
                        pred_img, _ = decoders[cam](pred_emb)
                        pred_img = rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)
                        pred_img = pred_img[0, 0].cpu().numpy().transpose(1, 2, 0)
                        pred_imgs[i][tgt] = np.clip(pred_img, 0, 1)

        # Match eval_dino_wm_tactile pred embeds: dense (T, N, D) with GT where not predicted
        if save_pred and pred_emb_np and pred_valid is not None:
            for cam in predict_cameras:
                if cam not in pred_emb_np:
                    continue
                gt_src = gt_emb_np.get(cam)
                if gt_src is None:
                    gt_src = predict_grids[cam][0].detach().cpu().numpy().astype(np.float32)
                for t in range(T):
                    if pred_valid[t] == 0:
                        pred_emb_np[cam][t] = gt_src[t]

        # Build video: gt | pred | diff [| decoded_gt]
        frames = []
        for t in range(T):
            gt_row = np.concatenate([all_gt_imgs[i][t] for i in range(len(predict_cameras))], axis=1)
            pred_row = np.concatenate([pred_imgs[i][t] for i in range(len(predict_cameras))], axis=1)
            diff_row = np.abs(gt_row.astype(np.float32) - pred_row.astype(np.float32))
            if diff_row.shape[-1] == 1:
                diff_row = np.repeat(diff_row, 3, axis=-1)
            rows = [gt_row, pred_row, diff_row]
            if decoded_gt_imgs is not None:
                dec_gt_row = np.concatenate([decoded_gt_imgs[i][t] for i in range(len(predict_cameras))], axis=1)
                rows.append(dec_gt_row)
            frame = np.vstack(rows)
            frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

        video = np.stack(frames, axis=0)
        if args.wandb and HAS_WANDB:
            video_wandb = np.transpose(video, (0, 3, 1, 2))
            videos[f"episode_{ep_idx}"] = wandb.Video(video_wandb, fps=args.video_fps, format="mp4")

        save_dir = args.output_dir or "./eval_k_step_videos"
        os.makedirs(save_dir, exist_ok=True)
        # Episode-file datasets identify trajectories by their full input path.
        # Keep that path out of output filenames, while preserving group IDs
        # for consolidated datasets.
        trajectory_name = Path(traj_id).stem if args.no_consolidated else str(traj_id)
        base = f"episode_{ep_idx}_traj_{trajectory_name}"
        video_path = os.path.join(save_dir, f"{base}.mp4")
        save_rgb_mp4(video_path, video, args.video_fps)
        print(f"Saved: {video_path}")

        if save_gt or save_pred:
            # Same as eval_dino_wm_tactile.py --save_embeds: *_gt_embeds.npz / *_pred_embeds.npz, np.savez
            emb_root = args.embeddings_output_dir or save_dir
            os.makedirs(emb_root, exist_ok=True)
            if save_gt and gt_emb_np:
                gt_path = os.path.join(emb_root, f"{base}_gt_embeds.npz")
                gt_dict = {cam: gt_emb_np[cam] for cam in predict_cameras if cam in gt_emb_np}
                np.savez(gt_path, **gt_dict)
                print(f"Saved embeddings: {gt_path}")
            if save_pred and pred_emb_np:
                pred_path = os.path.join(emb_root, f"{base}_pred_embeds.npz")
                pred_dict = {
                    cam: pred_emb_np[cam] for cam in predict_cameras if cam in pred_emb_np
                }
                np.savez(pred_path, **pred_dict)
                print(f"Saved embeddings: {pred_path}")

    return videos


def parse_args():
    from dino_wm.config import load_config

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs" / "liquid.yaml")
    )
    config_args, _ = config_parser.parse_known_args()
    config_path = config_args.config
    if not Path(config_path).is_file():
        config_parser.error(f"Config file does not exist: {config_path}")
    cfg_wm = load_config("eval_wm", config_path)
    cfg_k = load_config("eval_wm_k_step", config_path)
    cfg = {**cfg_wm, **cfg_k}

    p = argparse.ArgumentParser(
        description="Evaluate the active liquid k-step concat model with recorded history."
    )
    p.add_argument("--config", type=str, default=config_path)
    p.add_argument("--wm_checkpoint", type=str, required=True)
    p.add_argument("--decoder_dir", type=str, required=True)
    p.add_argument("--hdf5_path", type=str, required=True)
    p.add_argument("--norm_stats_json", type=str, default=None)
    p.add_argument("--cameras", type=str, default=cfg.get("cameras", "camera_0,camera_1,camera_2"))
    p.add_argument("--condition_cameras", type=str, default=cfg.get("condition_cameras"))
    p.add_argument("--predict_cameras", type=str, default=cfg.get("predict_cameras"))
    p.add_argument("--num_hist", type=int, default=cfg.get("num_hist", 1))
    p.add_argument("--num_pred", type=int, default=cfg.get("num_pred", 1))
    p.add_argument("--frameskip", type=int, default=cfg.get("frameskip", 8))
    p.add_argument(
        "--temporal_mode",
        type=str,
        choices=["lpb", "consecutive"],
        default=cfg.get("temporal_mode", "lpb"),
        help='Train/test split for TactileKStepDataset trajectory list (same as training).',
    )
    p.add_argument(
        "--eval_hist_mode",
        type=str,
        choices=["lpb", "consecutive"],
        default=cfg.get("eval_hist_mode", "lpb"),
        help=(
            "Prediction video: recorded history layout. "
            '"lpb" = subsampled raw indices every frameskip (default). '
            '"consecutive" = dense raw indices t_last-nh+1..t_last (match temporal_mode consecutive training).'
        ),
    )
    p.add_argument("--num_test", type=int, default=cfg.get("num_test", 100))
    p.add_argument("--device", type=str, default=cfg.get("device", "cuda:0"))
    p.add_argument("--output_dir", type=str, default=cfg.get("output_dir"))
    p.add_argument("--no_consolidated", action="store_true")
    p.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=cfg.get("wandb_project", "dino-wm-tactile-k-step"))
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--num_episode_videos", type=int, default=cfg.get("num_episode_videos", 5))
    p.add_argument("--max_episode_len", type=int, default=cfg.get("max_episode_len", 300))
    p.add_argument("--video_fps", type=int, default=cfg.get("video_fps", 5))
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--normalize_states", action="store_true")
    p.set_defaults(normalize_states=cfg.get("normalize_states", False))
    p.add_argument(
        "--save_gt_embeddings",
        action="store_true",
        default=cfg.get("save_gt_embeddings", False),
        help="Save GT patch embeddings (per predict camera, shape T x N x D) in .npz next to videos.",
    )
    p.add_argument(
        "--save_pred_embeddings",
        action="store_true",
        default=cfg.get("save_pred_embeddings", False),
        help="Save predicted patch embeddings; unpredicted context slots are copied from GT.",
    )
    p.add_argument(
        "--save_embeddings",
        action="store_true",
        default=cfg.get("save_embeddings", False),
        help="Shorthand: enable both --save_gt_embeddings and --save_pred_embeddings.",
    )
    p.add_argument(
        "--embeddings_output_dir",
        type=str,
        default=cfg.get("embeddings_output_dir"),
        help="Directory for *_gt_embeds.npz / *_pred_embeds.npz (default: --output_dir).",
    )
    p.add_argument(
        "--decode_gt_embeddings",
        action="store_true",
        default=cfg.get("decode_gt_embeddings", False),
        help="Decode GT embeddings through the decoder and add as a fourth row in the video.",
    )
    p.add_argument(
        "--model_arch",
        type=str,
        choices=["concat"],
        default=cfg.get("model_arch", "concat"),
        help="The released architecture is TactileKStepTransformer (concat).",
    )

    args = p.parse_args()
    if args.model_arch != "concat":
        p.error("This release supports only model_arch=concat; use the active liquid config")
    if args.num_pred != 1:
        p.error("The released k-step heads predict one step; use --num_pred 1")
    if args.save_embeddings:
        args.save_gt_embeddings = True
        args.save_pred_embeddings = True
    hdf5_dir = args.hdf5_path if os.path.isdir(args.hdf5_path) else os.path.dirname(os.path.abspath(args.hdf5_path))
    args.norm_stats_path = args.norm_stats_json or os.path.join(hdf5_dir, "norm_stats.json")
    return args


def main():
    args = parse_args()

    import dino_wm.dino_decoder as _dino_dec_mod
    if not hasattr(_dino_dec_mod, "dist_fn"):
        _dino_dec_mod.dist_fn = type("DistFn", (), {"all_reduce": lambda x: None})()

    cameras = [c.strip() for c in args.cameras.split(",")]
    condition_cameras = cameras if not args.condition_cameras else [c.strip() for c in args.condition_cameras.split(",")]
    predict_cameras = cameras if not args.predict_cameras else [c.strip() for c in args.predict_cameras.split(",")]

    for c in cameras + condition_cameras + predict_cameras:
        if c not in CAMERA_CONFIG:
            raise ValueError(f"Unknown camera: {c}")

    device = torch.device(args.device)
    norm_stats = NormStats(args.norm_stats_path, str(args.device))

    camera_dims = {c: CAMERA_EMB_DIMS.get(c, 768) for c in cameras}
    model_kwargs = dict(
        cameras=cameras,
        camera_dims=camera_dims,
        condition_cameras=condition_cameras,
        predict_cameras=predict_cameras,
        common_dim=384,
        ac_dim=64,
        state_dim=8,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
        frameskip=args.frameskip,
        patches_per_frame=PATCH_SIDE_DINOV3 * PATCH_SIDE_DINOV3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
    )
    transition = TactileKStepTransformer(**model_kwargs).to(device)

    print(f"Evaluating concat checkpoint: {args.wm_checkpoint}")
    wm_state = torch.load(args.wm_checkpoint, map_location=device)
    transition.load_state_dict(_strip_compile_prefix(wm_state))
    transition.eval()
    if not args.no_compile and hasattr(torch, "compile"):
        transition = torch.compile(transition, mode="reduce-overhead")
        print("Using torch.compile on world model")

    decoders = load_decoders(args.decoder_dir, predict_cameras, device)
    if not decoders:
        raise ValueError(f"No decoders found in {args.decoder_dir}")
    print(f"Loaded decoders: {list(decoders.keys())}")

    eval_ds = TactileKStepDataset(
        args.hdf5_path,
        cameras=cameras,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
        frameskip=args.frameskip,
        split="test",
        num_test=args.num_test,
        is_consolidated=not args.no_consolidated,
        seed=args.seed,
        temporal_mode=args.temporal_mode,
    )

    if args.wandb and HAS_WANDB:
        run_name = args.wandb_run_name or f"eval_k_step_h{args.num_hist}_p{args.num_pred}_fs{args.frameskip}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    videos = create_episode_videos(
        transition, decoders, eval_ds, predict_cameras,
        condition_cameras, args, device, norm_stats,
    )

    if args.wandb and HAS_WANDB and videos:
        wandb.log(videos)

    print("Done.")


if __name__ == "__main__":
    main()
