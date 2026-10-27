import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import h5py
import numpy as np

from dino_wm.export_dataset import export_dataset, sha256_file
from dino_wm.tactile_k_step_dataset import TactileKStepDataset
from dino_wm.tactile_dataset import load_full_episode


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.hdf5"
        self.normalization = self.root / "normalization.json"
        self.normalization.write_text(json.dumps({"mean_acs": [0] * 7, "std_acs": [1] * 7}))
        self.output = self.root / "exported"
        with h5py.File(self.source, "w") as handle:
            for index in (0, 1, 10, 2):
                group = handle.create_group(f"trajectory_{index}")
                group["actions"] = np.arange(16 * 7, dtype=np.float64).reshape(16, 7)
                group["states"] = np.full((16, 8), index, dtype=np.float64)
                for name in ("cam_0_patch_embd", "cam_1_patch_embd", "cam_tactile_patch_embd"):
                    group[name] = np.full((16, 196, 4), index / 10, dtype=np.float32)
                for name in ("camera_0", "camera_1", "camera_2"):
                    group[name] = np.full((16, 4, 4, 3), index, dtype=np.uint8)
        self.source_hash = sha256_file(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def export(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return export_dataset(
                self.source, self.normalization, self.output,
                task="pipette", num_test=1, workers=1, **kwargs,
            )

    def test_lossless_export_and_existing_loader_agree(self):
        metadata = self.export()
        self.assertTrue(metadata["complete"])
        self.assertEqual(metadata["episodes"], 4)
        self.assertEqual(sha256_file(self.source), self.source_hash)
        with h5py.File(self.source, "r") as original:
            for name in original:
                with h5py.File(self.output / "episodes" / f"{name}.hdf5", "r") as copied:
                    for field in original[name]:
                        self.assertEqual(copied["data"][field].dtype, original[name][field].dtype)
                        self.assertEqual(
                            copied["data"][field][()].tobytes(), original[name][field][()].tobytes(),
                        )
        options = dict(cameras=["camera_2"], num_hist=1, num_pred=1, frameskip=8, num_test=1, seed=42)
        consolidated = TactileKStepDataset(str(self.source), split="test", **options)
        episodes = TactileKStepDataset(
            str(self.output / "episodes"), split="test", is_consolidated=False, **options,
        )
        self.assertEqual(consolidated.trajectory_ids, [Path(value).stem for value in episodes.trajectory_ids])
        for key, value in consolidated[0].items():
            np.testing.assert_array_equal(value.numpy(), episodes[0][key].numpy())
        split = json.loads((self.output / "splits.json").read_text())
        expected = sorted(["trajectory_0", "trajectory_1", "trajectory_10", "trajectory_2"])
        random.Random(42).shuffle(expected)
        self.assertEqual(split["test"], expected[:1])
        self.assertEqual(split["train"], expected[1:])

    def test_partial_export_can_resume_without_overwriting_completed_episode(self):
        self.assertFalse(self.export(limit=1)["complete"])
        episode = self.output / "episodes/trajectory_0.hdf5"
        before = episode.stat().st_mtime_ns
        self.assertTrue(self.export()["complete"])
        self.assertEqual(episode.stat().st_mtime_ns, before)

    def test_changed_output_is_rejected(self):
        self.export(limit=1)
        episode = self.output / "episodes/trajectory_0.hdf5"
        with h5py.File(episode, "r+") as handle:
            handle["data/actions"][0, 0] = -123.0
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.export()

    def test_unrecognized_output_is_preserved(self):
        self.output.mkdir()
        existing = self.output / "important.txt"
        existing.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(existing.read_text(), "keep")

    def test_parallel_export_verifies_all_episodes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = export_dataset(
                self.source, self.normalization, self.output,
                task="pipette", num_test=1, workers=2,
            )
        self.assertTrue(metadata["complete"])
        self.assertEqual(metadata["episodes"], 4)
        self.assertEqual(metadata["nonfinite_values"], 0)

    def test_extra_terminal_observation_is_preserved_and_loader_uses_action_length(self):
        with h5py.File(self.source, "r+") as handle:
            group = handle["trajectory_0"]
            images = np.concatenate([
                group["camera_0"][()], np.full((1, 4, 4, 3), 255, dtype=np.uint8),
            ])
            del group["camera_0"]
            group["camera_0"] = images
            group["gripper_states"] = np.arange(17, dtype=np.float64)
        metadata = self.export()
        self.assertEqual(metadata["episodes_with_terminal_observations"], 1)
        episode = self.output / "episodes/trajectory_0.hdf5"
        with h5py.File(episode, "r") as handle:
            np.testing.assert_array_equal(handle["data/camera_0"][()], images)
            self.assertEqual(handle["data/gripper_states"].shape, (17,))
        loaded = load_full_episode(
            str(self.output / "episodes"), str(episode), ["camera_0"],
            is_consolidated=False, resize_to_224=False,
        )
        self.assertEqual(loaded["camera_0_image"].shape[0], 16)
        self.assertEqual(loaded["camera_0_embd"].shape[0], 16)

    def test_controller_configuration_attribute_is_preserved(self):
        config = json.dumps({
            "controller_type": "OSC_POSE",
            "controller_cfg": {"controller_type": "OSC_POSITION", "is_delta": True},
        })
        with h5py.File(self.source, "r+") as handle:
            handle["trajectory_0"].attrs["config"] = config
        metadata = self.export()
        self.assertEqual(metadata["episodes_with_controller_config"], 1)
        with h5py.File(self.output / "episodes/trajectory_0.hdf5", "r") as handle:
            self.assertEqual(handle["data"].attrs["config"], config)
            self.assertEqual(handle["data"].attrs["original_trajectory_id"], "trajectory_0")
        records = [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]
        self.assertIn("config", records[0]["attributes"])

    def test_unknown_attributes_stop_export_without_changing_source(self):
        with h5py.File(self.source, "r+") as handle:
            handle["trajectory_0"].attrs["unreviewed"] = "fixture"
        before = sha256_file(self.source)
        with self.assertRaisesRegex(ValueError, "Review source attributes"):
            self.export()
        self.assertEqual(sha256_file(self.source), before)
        self.assertFalse((self.output / "episodes/trajectory_0.hdf5").exists())


if __name__ == "__main__":
    unittest.main()
