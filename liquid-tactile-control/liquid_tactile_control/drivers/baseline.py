# Publication cleanup (2026-10-27): adapted from deoxys_control/deoxys/examples/run_dp_model_tactile_liquid_log.py.
"""Replay demonstration trajectories."""

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from liquid_tactile_control.robot_config import robot_config_parse_args
from liquid_tactile_control.policy import RealPolicy
import imageio

import logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    robot_config_parse_args(parser)
    # parser.add_argument('--ckpt_dir', type=str, default=None)
    # parser.add_argument('--ckpt', type=str, default='itr_400000.ckpt')
    # parser.add_argument('--ckpt_dir', type=str, default=None)
    # parser.add_argument('--ckpt_dir', type=str, default=None)
    # parser.add_argument('--ckpt_dir', type=str, default=None)

    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ckpt_iter', type=str, default=23999)#31999)
    # parser.add_argument('--ckpt_iter', type=str, default=49999)
    parser.add_argument('--use_tactile', action='store_true', default=False)
    return parser.parse_args()

def eef_pose_to_state(T, gripper):
    # Extract translation
    x, y, z = T[:3, 3]

    # Extract rotation matrix and convert to quaternion
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # Returns [qx, qy, qz, qw]

    # Concatenate into final state
    eef_state = np.concatenate(([x, y, z], quat, [gripper]))
    return eef_state

def main():
    args = parse_args()

    global logger
    from easydict import EasyDict
    from deoxys import config_root
    from deoxys.experimental.motion_utils import follow_joint_traj, reset_joints_to
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils import YamlConfig
    from deoxys.utils.input_utils import input2action
    from deoxys.utils.log_utils import get_deoxys_example_logger
    from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface
    logger = get_deoxys_example_logger()

    # Setting up robot, teleop, and camera interfaces
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg), gripper_mode='continuous_smooth')
    policy = RealPolicy(['state'], args.ckpt_dir, args.ckpt_iter)
    #device.start_control()
    camera_ids = [0, 1, 2]
    camera_types = ['rs', 'zed', 'tactile']
    camera_info = EasyDict({
        "camera_id": camera_ids,
        "camera_type": camera_types,
        "camera_name": [f"camera_{camera_types[id]}_{camera_ids[id]}" for id,_ in enumerate(camera_ids)]
    })
    cr_interfaces = {}
    for camera_id in camera_ids:
        print(camera_id, 'id')
        if 'tactile' in camera_types[camera_id]:
            cr_interface = CameraRedisSubInterface(camera_info=camera_info, camera_id=camera_id, use_color=False, use_tactile=True)
        else:
            cr_interface = CameraRedisSubInterface(camera_info=camera_info, camera_id=camera_id)

        cr_interface.start()
        cr_interfaces[camera_id] = cr_interface

    # demo_file = h5py.File(demo_file_name, "w")
    controller_type = args.controller_type
    print(controller_type)
    controller_cfg = YamlConfig(
        os.path.join(config_root, args.controller_cfg)
    ).as_easydict()

    args.folder.mkdir(parents=True, exist_ok=True)
    experiment_id = 0
    logger.info(f"Saving to {args.folder}")
    init_joints = [-0.14091554, 0.02724072, 0.33824686, -2.48239114, -0.12527768, 2.53220921, -2.87378524]
    # init_joints = [-0.20178348, 0.39381444, 0.22824561, -1.90957881, -0.12470368, 2.22629113, -2.17852913]
    # init_joints = [-0.18941418455646322, 0.3986268416401311, 0.37023725867044477, -1.8986898980216536, -0.1220696096038897, 2.1729692311931763, -2.1428127053565933]
    episode_id = 0

    # init_joints = np.array([0.00727177, -0.11309989, -0.01733192, -2.51448585, -0.024387, 2.56470902, 0.79139705])
    while True:
        data = {"actions": [], "ee_states": [], "joint_states": [], "gripper_states": []}
        for camera_id in camera_ids:
            data[f"camera_{camera_id}"] = []

        i = 0
        start = False

        previous_state_dict = None
        reset_joints_to(robot_interface, init_joints)
        time.sleep(1)
        # ee_state = eef_pose_to_state(robot_interface.last_eef_pose, robot_interface.last_gripper_q)
        # print('ee_state', ee_state)
        # robot_interface.control(
                    # controller_type=controller_type,
                    # action=[0.1, 0, 0, 0, 0, 0, 0],
                    # controller_cfg=controller_cfg,
                # )
        # time.sleep(1)
        ee_state = eef_pose_to_state(robot_interface.last_eef_pose, robot_interface.last_gripper_q)
        print('ee_state', ee_state)
        # breakpoint()
        print('Beginning Trajectory')
        interrupted = False
        try:
            while True:
                start_time = time.time()
                last_state = robot_interface._state_buffer[-1]
                last_gripper_state = robot_interface._gripper_state_buffer[-1]

                obs_dict = {}
                obs_dict['state'] = eef_pose_to_state(np.array(last_state.O_T_EE).reshape(4,4).T, np.array(last_gripper_state.width))
                ## save the state, joint, gripper, action to the data dictionary

                data["ee_states"].append(np.array(last_state.O_T_EE))
                data["joint_states"].append(np.array(last_state.q))
                data["gripper_states"].append(np.array(last_gripper_state.width))
                for camera_id in camera_ids:
                    original_img = cr_interfaces[camera_id].get_img()
                    if camera_id == 0:
                        color_img = original_img['color']
                        img  = cv2.resize(color_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    if camera_id == 1:
                        color_img = original_img['color']
                        img  = cv2.resize(color_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    if camera_id == 2:
                        color_img = original_img['tactile']
                        img = cv2.resize(color_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    data[f"camera_{camera_id}"].append(color_img)

                    if not args.use_tactile and camera_id == 2:
                        continue
                    obs_dict[f"camera_{camera_id}"] = img
                end_time = time.time_ns()
                action, log, record = policy.get_action(obs_dict, binary_gripper=False)
                data["actions"].append(action)
                print(f"step {i} action: {action}")


                # if record:
                i += 1

                # set unused orientation dims to 0
                if controller_type == "OSC_YAW":
                    action[3:5] = 0.0
                elif controller_type == "OSC_POSITION":
                    action[3:6] = 0.0


                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )

                if i > 900:
                    break
        except KeyboardInterrupt:
            print('\nKeyboard interrupt detected, saving trajectory...')
            interrupted = True

        ## ask for user input about labels
        label = input("Enter the label for the trajectory: (0 is left success, 1 is left failure, 2 is right success, 3 is right failure) ")

        ## save the data to a h5 file
        ## list the traj_xxx.hdf5 files in the folder and get the max episode_id
        # episode_id = 0

        ## count how many files in the folder
        episode_id = len(list(Path(args.folder).glob("traj_*.hdf5")))
        ## create the video folders to save the videos in mp4 format
        video_folder = args.folder / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        episode_video_folder = args.folder / "videos" / f"episode_{episode_id:04d}"
        episode_video_folder.mkdir(parents=True, exist_ok=True)
        for camera_id in camera_ids:
            video_folder = episode_video_folder / f"camera_{camera_id}.mp4"
            ## use imageio to save the videos

            images = data[f"camera_{camera_id}"]
            imageio.mimsave(video_folder, images, fps=10)
        print(f'Videos saved to {video_folder}')

        with h5py.File(args.folder / f"traj_{episode_id:04d}.hdf5", "w") as f:
            grp = f.create_group("data")
            grp.attrs["label"] = label
            ## remove the camera (0, 1,2) from the data dictionary
            data.pop('camera_0')
            data.pop('camera_1')
            data.pop('camera_2')
            ## add the video path to the data dictionary
            data['video_path'] = episode_video_folder.as_posix()
            for key, value in data.items():
                grp.create_dataset(key, data=value)
        print(f'Trajectory saved to {args.folder / f"traj_{episode_id:04d}.hdf5"}')
        print('Trajectory ended')
        episode_id += 1
        reset_joints_to(robot_interface, init_joints)
        time.sleep(1)
        if interrupted:
            robot_interface.close()
            print('Exiting...')
            break

        robot_interface.close()


if __name__ == "__main__":
    main()
