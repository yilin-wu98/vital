"""Configure the liquid experiments, or print their commands without importing hardware."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

METHODS = {
    "baseline": "baseline",
    "two-step": "two_step",
    "tactile-guidance": "tactile_guidance",
    "tactile-sampling": "tactile_sampling",
    "vision-sampling": "vision_sampling",
    "combined-sampling": "combined_sampling",
}
TACTILE_TEXT = {"reward_text": "grasp lightly", "reward_text_phase2": "grasp heavily"}
VISUAL_TEXT = {
    "reward_task": "transfer liquid to the yellow cup",
    "reward_task_phase2": "move the dropper back to the red cup",
}
PRESETS = {
    "baseline": {},
    "two-step": {
        **TACTILE_TEXT, **VISUAL_TEXT, "n_samples": 10, "ss": 1,
        "lpb_scale": True, "guidance_scale": 10, "normalize_grad": True,
        "compile_wm": True, "separate_phase": True, "mixed_reward": True,
    },
    "tactile-guidance": {
        **TACTILE_TEXT, "use_tactile": True, "ss": 1, "lpb_scale": True,
        "guidance_scale": 10, "normalize_grad": True, "guidance_start_step": 12,
        "guidance_every": 1, "compile_wm": True, "visualize_guidance": True,
        "separate_phase": True, "mixed_reward": True, "phase_threshold": 0.50,
    },
    "tactile-sampling": {
        **TACTILE_TEXT, "use_tactile": True, "action_select": "argmax",
        "mixed_reward": True,
    },
    "vision-sampling": {
        **VISUAL_TEXT, "reward_task_phase2": "move the dropper to the red cup",
        "use_tactile": True, "action_select": "argmax", "lookahead_steps": 2,
        "separate_phase": True, "phase_threshold": 0.7,
    },
    "combined-sampling": {
        **TACTILE_TEXT, **VISUAL_TEXT,
        "reward_task": "transfer liquid to the blue cup",
        "action_select": "argmax", "lookahead_steps": 2,
        "phase_threshold": 0.63, "mixed_reward": True, "separate_phase": True,
    },
}


def _path(value, base):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def build_options(config, method, base):
    paths = config.get("paths", {})
    required = ["interface_cfg", "policy_dir"]
    if method != "baseline":
        required += ["wm_checkpoint", "norm_stats_json", "anytouch_checkpoint",
                     "dinov3_hub_dir", "dinov3_weights"]
    if method in ("two-step", "vision-sampling", "combined-sampling"):
        required.append("vision_decoder")
    missing = [key for key in required if not paths.get(key)]
    if missing:
        raise ValueError("Missing configuration paths: " + ", ".join(missing))
    options = {
        "interface-cfg": _path(paths["interface_cfg"], base),
        "controller-cfg": config.get("controller_cfg", "osc-position-controller.yml"),
        "controller-type": config.get("controller_type", "OSC_POSE"),
        "folder": str(Path(_path(config.get("output_dir", "outputs"), base)) / method),
        "ckpt_dir": _path(paths["policy_dir"], base),
        "ckpt_iter": config.get("policy_iteration", 23999),
    }
    if method != "baseline":
        options.update({key: _path(paths[key], base) for key in required
                        if key not in ("interface_cfg", "policy_dir", "vision_decoder")})
        options.update({
            "wm_arch": "k_step", "wm_cameras": "camera_0,camera_1,camera_2",
            "num_hist": 1, "num_pred": 1, "frameskip": 8, "temporal_mode": "lpb",
            "eval_server_url": config.get("eval_server_url", "http://localhost:8000"),
        })
    if method in ("two-step", "vision-sampling", "combined-sampling"):
        option = "decoder_camera0" if method == "two-step" else "decoder_camera1"
        options[option] = _path(paths["vision_decoder"], base)
    options.update(PRESETS[method])
    overrides = config.get("method_options", {}).get(method, {})
    if not isinstance(overrides, dict):
        raise ValueError("method_options must map each method to an option dictionary")
    options.update(overrides)
    return {key: value for key, value in options.items() if value is not None}


def options_to_args(options):
    result = []
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                result.append("--" + key)
        elif isinstance(value, (str, int, float)):
            result.extend(("--" + key, str(value)))
        else:
            raise ValueError(f"Unsupported value for option {key}: {value!r}")
    return result


def validate_paths(options):
    """Check inputs locally, without talking to a robot, camera, or reward server."""
    problems = []
    files = ("interface-cfg", "wm_checkpoint", "norm_stats_json",
             "anytouch_checkpoint", "dinov3_weights", "decoder_camera0", "decoder_camera1")
    for key in files:
        if key in options and not Path(options[key]).is_file():
            problems.append(f"{key}: file not found: {options[key]}")
    for key in ("ckpt_dir", "dinov3_hub_dir"):
        if key in options and not Path(options[key]).is_dir():
            problems.append(f"{key}: directory not found: {options[key]}")
    if "ckpt_dir" in options:
        root = Path(options["ckpt_dir"])
        for name in ("agent_config.yaml", "exp_config.yaml", "obs_config.yaml",
                     "ob_norm.json", "ac_norm.json", ".hydra/config.yaml",
                     f"itr_{options['ckpt_iter']}.ckpt"):
            if not (root / name).is_file():
                problems.append(f"policy input not found: {root / name}")
    if options.get("mixed_reward") and not options.get("reward_text_phase2"):
        problems.append("mixed_reward requires reward_text_phase2")
    if problems:
        raise ValueError("\n".join(problems))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="two-step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command; do not load models or contact hardware")
    parser.add_argument("--check", action="store_true",
                        help="Check local input files and exit; do not contact hardware")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text())
        options = build_options(config, args.method, args.config.resolve().parent)
        command = [sys.executable, "-m",
                   "liquid_tactile_control.drivers." + METHODS[args.method],
                   *options_to_args(options)]
        if args.dry_run:
            print(shlex.join(command))
            return 0
        validate_paths(options)
        if args.check:
            print(f"{args.method}: local input files are present; hardware was not contacted.")
            return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
