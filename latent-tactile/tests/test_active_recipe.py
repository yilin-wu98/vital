import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dino_wm.eval_dino_wm_tactile_k_step import parse_args as parse_evaluation
from dino_wm.train_dino_decoder_tactile import get_cameras_to_train
from dino_wm.train_dino_decoder_tactile import parse_args as parse_decoder
from dino_wm.train_dino_wm_tactile_k_step import parse_args as parse_training


class ActiveRecipeTests(unittest.TestCase):
    def parse(self, parser, *extra):
        arguments = ["program", "--hdf5_path", "/unused/liquid.hdf5"]
        if parser is parse_evaluation:
            arguments += [
                "--wm_checkpoint", "/unused/wm_kstep_concat_h1_p1_fs8_iter67000.pth",
                "--decoder_dir", "/unused/decoders",
            ]
        with patch("sys.argv", arguments + list(extra)):
            return parser()

    def test_training_defaults_match_the_selected_saved_run(self):
        path = Path(__file__).resolve().parents[1] / "docs/recovered_runs.json"
        run = next(run for run in json.loads(path.read_text()) if run["run_id"] == "xrjdyidg")
        saved = run["effective_config"]
        args = self.parse(parse_training)
        for key in (
            "model_arch", "num_hist", "num_pred", "frameskip", "temporal_mode",
            "loss_mode", "multi_step_rollout_min", "multi_step_rollout_max",
            "noised", "action_noise_geom_p", "action_noise_timesteps",
            "normalize_states", "segment_sampling", "seed", "num_test",
            "batch_size", "eval_batch_size", "iters", "eval_every",
        ):
            with self.subTest(setting=key):
                self.assertEqual(getattr(args, key), saved[key])
        self.assertEqual(args.cameras.split(","), saved["cameras"])
        for key in ("condition_cameras", "predict_cameras"):
            self.assertEqual(
                [value.strip() for value in getattr(args, key).split(",")],
                [value.strip() for value in saved[key].split(",")],
            )
        self.assertEqual(Path(args.config).name, "liquid.yaml")
        self.assertFalse(args.wandb)

    def test_evaluation_matches_training_model_and_held_out_split(self):
        train = self.parse(parse_training)
        evaluation = self.parse(parse_evaluation)
        for key in (
            "config", "model_arch", "cameras", "condition_cameras",
            "predict_cameras", "num_hist", "num_pred", "frameskip",
            "temporal_mode", "normalize_states", "seed", "num_test",
        ):
            with self.subTest(setting=key):
                self.assertEqual(getattr(evaluation, key), getattr(train, key))
        self.assertEqual(evaluation.eval_hist_mode, train.temporal_mode)
        self.assertTrue(evaluation.save_gt_embeddings)
        self.assertTrue(evaluation.save_pred_embeddings)

    def test_removed_architecture_is_rejected_in_cli_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "old.yaml"
            config.write_text(
                "train_wm_k_step:\n  model_arch: xattn\n"
                "eval_wm_k_step:\n  model_arch: xattn\n"
            )
            for parser in (parse_training, parse_evaluation):
                for extra in (
                    ["--model_arch", "xattn"],
                    ["--config", str(config)],
                    [f"--config={config}"],
                ):
                    with self.subTest(parser=parser.__module__, arguments=extra):
                        with contextlib.redirect_stderr(io.StringIO()):
                            with self.assertRaises(SystemExit) as error:
                                self.parse(parser, *extra)
                        self.assertEqual(error.exception.code, 2)

    def test_multi_frame_heads_are_rejected_before_loading_data(self):
        for parser in (parse_training, parse_evaluation):
            with self.subTest(parser=parser.__module__):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        self.parse(parser, "--num_pred", "2")
                self.assertEqual(error.exception.code, 2)

    def test_documented_decoder_mode_produces_the_required_camera_groups(self):
        args = self.parse(
            parse_decoder, "--mode", "tactile", "--loss", "rae",
            "--iters", "15000", "--resize_to_224", "--compile",
        )
        self.assertEqual(
            get_cameras_to_train(args), [["camera_0", "camera_1"], ["camera_2"]],
        )


if __name__ == "__main__":
    unittest.main()
