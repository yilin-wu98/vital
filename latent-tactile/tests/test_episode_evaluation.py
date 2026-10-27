from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from dino_wm.dino_models import NormStats
from dino_wm.eval_dino_wm_tactile_k_step import create_episode_videos, parse_args
from dino_wm.tactile_k_step_dataset import TactileKStepDataset


class EpisodeEvaluationTests(unittest.TestCase):
    def test_consolidated_and_episode_inputs_write_local_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = root / "input" / "episodes"
            episodes.mkdir(parents=True)
            consolidated = root / "source.hdf5"
            arrays = {
                "actions": np.zeros((9, 7), dtype=np.float32),
                "states": np.zeros((9, 8), dtype=np.float32),
                "cam_tactile_patch_embd": np.ones((9, 196, 4), dtype=np.float32),
                "camera_2": np.zeros((9, 224, 224, 3), dtype=np.uint8),
            }
            for path, group in (
                (consolidated, "trajectory_0"),
                (episodes / "trajectory_0.hdf5", "data"),
            ):
                with h5py.File(path, "w") as handle:
                    data = handle.create_group(group)
                    for name, value in arrays.items():
                        data[name] = value

            def transition(embeddings, states, actions):
                return {"camera_2": embeddings["camera_2"] + 1}, states

            def decoder(embeddings):
                return torch.zeros(1, 3, 224, 224), None

            def write_video(path, frames, fps):
                # Exercise the output path without requiring a video codec.
                Path(path).write_bytes(b"video fixture")

            for separate in (False, True):
                with self.subTest(episode_files=separate):
                    output = root / ("separate" if separate else "consolidated")
                    source = episodes if separate else consolidated
                    argv = [
                        "evaluate", "--hdf5_path", str(source),
                        "--wm_checkpoint", "unused.pth", "--decoder_dir", "unused",
                        "--output_dir", str(output), "--cameras", "camera_2",
                        "--condition_cameras", "camera_2", "--predict_cameras", "camera_2",
                        "--num_test", "1", "--num_episode_videos", "1",
                        "--max_episode_len", "9", "--device", "cpu", "--no_compile",
                    ]
                    if separate:
                        argv.append("--no_consolidated")
                    with patch("sys.argv", argv):
                        args = parse_args()
                    dataset = TactileKStepDataset(
                        str(source), cameras=["camera_2"], split="test",
                        num_hist=1, num_pred=1, frameskip=8, num_test=1, seed=42,
                        is_consolidated=not separate,
                    )
                    with patch(
                        "dino_wm.eval_dino_wm_tactile_k_step.save_rgb_mp4", write_video,
                    ):
                        create_episode_videos(
                            transition, {"camera_2": decoder}, dataset,
                            ["camera_2"], ["camera_2"], args, "cpu",
                            NormStats(device="cpu"),
                        )
                    base = "episode_0_traj_trajectory_0"
                    self.assertEqual(
                        {path.name for path in output.iterdir()},
                        {f"{base}.mp4", f"{base}_gt_embeds.npz", f"{base}_pred_embeds.npz"},
                    )
                    with np.load(output / f"{base}_pred_embeds.npz") as predicted:
                        np.testing.assert_array_equal(predicted["camera_2"][:8], 1)
                        np.testing.assert_array_equal(predicted["camera_2"][8:], 2)


if __name__ == "__main__":
    unittest.main()
