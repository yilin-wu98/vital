from collections import deque
import argparse
import ast
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from liquid_tactile_control import robometer_client
from liquid_tactile_control.policy import _local_targets
from liquid_tactile_control.run import METHODS, build_options, options_to_args, validate_paths
from liquid_tactile_control.robot_config import robot_config_parse_args
from liquid_tactile_control.sampling_context import (
    build_k_step_state_tensor, k_step_hist_frame_indices,
)


class OfflineTests(unittest.TestCase):
    def test_history_alignment_and_start_padding(self):
        self.assertEqual(k_step_hist_frame_indices(9, 3, 4, "lpb"), [1, 5, 9])
        self.assertEqual(k_step_hist_frame_indices(1, 3, 4, "lpb"), [0, 0, 1])
        self.assertEqual(k_step_hist_frame_indices(9, 3, 4, "consecutive"), [7, 8, 9])
        states = deque(np.full(8, i, dtype=np.float32) for i in range(10))
        output = build_k_step_state_tensor(states, 3, 4, "lpb", np.zeros(8), "cpu", 2)
        torch.testing.assert_close(output[:, :, 0], torch.tensor([[1., 5., 9.], [1., 5., 9.]]))

    def test_bundled_policy_hydra_targets(self):
        original = OmegaConf.create({
            "_target_": "data4robotics.models.diffusion_unet.DiffusionUnetAgent",
            "features": {"_target_": "data4robotics.models.resnet.RobomimicResNet"},
        })
        remapped = _local_targets(original)
        self.assertTrue(remapped._target_.startswith("liquid_tactile_control.policy_backend."))
        self.assertTrue(original._target_.startswith("data4robotics."))

    def test_all_presets_build_without_hardware_imports(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/liquid.example.json"
        config = json.loads(config_path.read_text())
        for method in METHODS:
            options = build_options(config, method, config_path.parent)
            args = options_to_args(options)
            flags = args[::]  # Flag uniqueness also catches duplicate shell overrides.
            flags = [item for item in flags if item.startswith("--")]
            self.assertEqual(len(flags), len(set(flags)))
            self.assertIn("--interface-cfg", args)
            if method != "baseline":
                self.assertEqual(options["frameskip"], 8)
                self.assertEqual(options["num_hist"], 1)
            # Execute only the driver's argument parser. Its hardware imports
            # and main loop are deliberately excluded from this integration test.
            source = config_path.parents[1] / "liquid_tactile_control" / "drivers" / (METHODS[method] + ".py")
            tree = ast.parse(source.read_text())
            parser_function = next(node for node in tree.body
                                   if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
            namespace = {"argparse": argparse, "Path": Path,
                         "robot_config_parse_args": robot_config_parse_args}
            exec(compile(ast.Module(body=[parser_function], type_ignores=[]), str(source), "exec"), namespace)
            with patch("sys.argv", [str(source), *args]):
                namespace["parse_args"]()

    def test_missing_paths_stop_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                validate_paths({"interface-cfg": str(Path(directory) / "absent.yml")})

    def test_candidate_requests_match_streaming_protocol(self):
        response = Mock()
        response.json.return_value = {"rewards": [0.2, 0.9], "success_probs": [0.1, 0.8]}
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        with patch.object(robometer_client.requests, "post", return_value=response) as post:
            rewards, success = robometer_client.stream_evaluate_candidates_batch(
                "http://localhost:8000", "test-session", frames,
            )
        self.assertEqual(post.call_args.args[0],
                         "http://localhost:8000/stream/test-session/evaluate_candidates")
        upload = post.call_args.kwargs["files"]["frames"][1]
        np.testing.assert_array_equal(np.load(BytesIO(upload), allow_pickle=False), frames)
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(success.shape, (2,))
        response.raise_for_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
