"""Argument definitions without importing or starting the robot interface."""
import argparse
from pathlib import Path


def robot_config_parse_args(parser=None):
    own_parser = parser is None
    if own_parser:
        parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", required=True,
                        help="Path to this robot's Deoxys interface YAML")
    parser.add_argument("--folder", type=Path, required=True,
                        help="Output directory for this experiment")
    parser.add_argument("--ckpt", type=int, default=999)
    parser.add_argument("--controller-type", default="OSC_POSE")
    parser.add_argument("--controller-cfg", default="osc-position-controller.yml")
    return parser.parse_args() if own_parser else None
