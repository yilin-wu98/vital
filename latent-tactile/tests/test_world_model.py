import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from dino_wm.dino_models import NormStats
from dino_wm.dino_models_k_step import TactileKStepTransformer
from dino_wm.hdf5_to_dataset_tactile import stage_trajectories
from dino_wm.tactile_k_step_dataset import TactileKStepDataset


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "episodes.hdf5"
        with h5py.File(self.dataset, "w") as handle:
            for index in range(4):
                group = handle.create_group(f"trajectory_{index}")
                time = np.arange(16, dtype=np.float32)
                group["actions"] = np.repeat(time[:, None], 7, axis=1)
                group["states"] = np.repeat(time[:, None], 8, axis=1)
                group["cam_tactile_patch_embd"] = np.broadcast_to(
                    time[:, None, None], (16, 196, 8),
                )

    def tearDown(self):
        self.temp.cleanup()

    def test_trajectory_split_and_lpb_action_alignment(self):
        common = dict(
            cameras=["camera_2"], num_hist=1, num_pred=2, frameskip=2,
            num_test=1, seed=42,
        )
        train = TactileKStepDataset(str(self.dataset), split="train", **common)
        test = TactileKStepDataset(str(self.dataset), split="test", **common)
        self.assertFalse(set(train.trajectory_ids) & set(test.trajectory_ids))
        row = train[0]
        torch.testing.assert_close(row["state"][:, 0], torch.tensor([0., 2., 4.]))
        torch.testing.assert_close(row["action"][:4, 0], torch.arange(4.))
        # The unused terminal action slot repeats the preceding action, as in training.
        torch.testing.assert_close(row["action"][4:, 0], torch.tensor([3., 3.]))

    def test_consecutive_history_and_future_targets(self):
        dataset = TactileKStepDataset(
            str(self.dataset), cameras=["camera_2"], num_hist=3, num_pred=2,
            frameskip=2, num_test=1, temporal_mode="consecutive",
        )
        row = dataset[0]
        torch.testing.assert_close(row["state"][:, 0], torch.tensor([0., 1., 2., 4., 6.]))
        torch.testing.assert_close(row["action"][:6, 0], torch.tensor([0., 1., 1., 2., 2., 3.]))

    def test_forward_shapes_and_gradient_reaches_actions(self):
        torch.manual_seed(4)
        model = TactileKStepTransformer(
            cameras=["camera_2"], camera_dims={"camera_2": 8}, common_dim=12,
            ac_dim=8, num_hist=1, num_pred=1, frameskip=2, patches_per_frame=4,
            depth=1, heads=2, dim_head=8, mlp_dim=32, dropout=0,
        ).eval()
        actions = torch.randn(2, 1, 14, requires_grad=True)
        predictions, state = model(
            {"camera_2": torch.randn(2, 1, 4, 8)}, torch.randn(2, 1, 8), actions,
        )
        self.assertEqual(predictions["camera_2"].shape, (2, 1, 4, 8))
        self.assertEqual(state.shape, (2, 1, 8))
        (predictions["camera_2"].square().mean() + state.square().mean()).backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(actions.grad.abs().sum().item(), 0)

    def test_normalization_roundtrip_and_missing_file(self):
        stats = self.root / "norm.json"
        stats.write_text(json.dumps({"mean_acs": [1] * 7, "std_acs": [2] * 7}))
        normalizer = NormStats(str(stats), device="cpu")
        actions = torch.randn(3, 4, 7)
        torch.testing.assert_close(
            normalizer.unnormalize_acs(normalizer.normalize_acs(actions)), actions,
        )
        with self.assertRaises(FileNotFoundError):
            NormStats(str(self.root / "missing.json"), device="cpu")

    def test_preprocessing_stages_copies_and_refuses_existing_destination(self):
        original = self.dataset.read_bytes()
        staged = stage_trajectories([self.root], self.root.parent / (self.root.name + "-staged"))
        destination = Path(staged[0]).parent
        try:
            copied = Path(staged[0]) / self.dataset.name
            copied.write_bytes(b"changed working copy")
            self.assertEqual(self.dataset.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                stage_trajectories([self.root], destination)
        finally:
            import shutil
            shutil.rmtree(destination)


if __name__ == "__main__":
    unittest.main()
