# Publication cleanup (2026-10-27): adapted from deoxys_control/deoxys/deoxys/utils/policy_callback/real_policy.py.
"""Diffusion-policy checkpoint adapter for the liquid experiments."""
import threading
import time
from collections import namedtuple

import numpy as np

# from oculus_reader import OculusReader
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import json
from omegaconf import DictConfig, OmegaConf
import hydra
from collections import deque, defaultdict
from scipy.spatial.transform import Rotation
import cv2

PRED_HORIZON = 16
ACT_HORIZON = 8
EXP_WEIGHT = 0
GRIP_THRESH = 0.55

def _local_targets(config):
    """Resolve the checkpoint's Hydra targets to the bundled policy backend."""
    data = OmegaConf.to_container(config, resolve=False)
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "_target_" and isinstance(item, str) and item.startswith("data4robotics."):
                    value[key] = item.replace(
                        "data4robotics.", "liquid_tactile_control.policy_backend.", 1
                    )
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(data)
    return OmegaConf.create(data)

def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(torch.Tensor(rot_mat)).as_quat()
    return quat


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X



class RealPolicy:
    """
    A wrapper for using a robot policy on hardware
    """
    def __init__(self, obs_keys, ckpt_dir, ckpt = None, log=False, logger=None):
        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method('spawn')
        agent_yaml_path = Path(ckpt_dir, "agent_config.yaml")
        exp_config_path = Path(ckpt_dir, "exp_config.yaml")
        obs_config_path = Path(ckpt_dir, "obs_config.yaml")
        ob_dict_path = Path(ckpt_dir, "ob_norm.json")
        ac_dict_path = Path(ckpt_dir, "ac_norm.json")

        f = open(ob_dict_path, 'r')
        data = json.load(f)
        self.ob_max = np.array(data['maximum'])
        self.ob_min = np.array(data['minimum'])

        f = open(ac_dict_path, 'r')
        data = json.load(f)
        self.ac_max = np.array(data['maximum'])
        self.ac_min = np.array(data['minimum'])


        hydra_config_path = Path(ckpt_dir, ".hydra", "config.yaml")
        print('agent yaml')
        print(agent_yaml_path)
        agent_config = _local_targets(OmegaConf.load(agent_yaml_path))
        exp_config = OmegaConf.load(exp_config_path)
        obs_config = _local_targets(OmegaConf.load(obs_config_path))
        hydra_config = OmegaConf.load(hydra_config_path)

        # The complete trained state is loaded below, so downloading ImageNet
        # initialization weights would be redundant and prevent offline use.
        if OmegaConf.select(agent_config, "features.weights") is not None:
            agent_config.features.weights = None
        agent = hydra.utils.instantiate(agent_config)
        if exp_config.params.exp_name is not None:
            model_name = exp_config.params.exp_name
        else:
            model_name = exp_config.exp_name
        print(f'Loading itr {ckpt}')
        load_dict = torch.load(Path(ckpt_dir, f"itr_{ckpt}.ckpt"), weights_only=False)
        agent.load_state_dict(load_dict["model"])


        # Load transforms

        loaded_transform = obs_config.transform
        self.transform = hydra.utils.instantiate(loaded_transform)

        self.model_name = model_name
        self.agent = agent.eval().cuda()
        self.obs_keys = obs_keys
        self.cam_indices = hydra_config.task.train_buffer.cam_indexes

        self.act_history = []
        self.prev_img = defaultdict(lambda: None)
        self.prev_obs = None
        self.img_chunk = exp_config.params.img_chunk

    def on_begin_traj(self, traj_idx):
        print(f"beginning new trajectory")

    def _proc_image(self, img):
        rgb_img = img[:, :, :3]

        rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255

        return self.transform(rgb_img)[None].cuda()

    def _get_images_and_states(self, obs):
        """
        Return images and states given observations
        """
        images = {}
        state = np.empty(0)
        for key in self.obs_keys:
            state = np.append(state, obs[key])

        cam_keys = [key for key in obs.keys() if "cam" in key]
        # cam_keys = ['camera_0', 'camera_1']
        for i, key in enumerate(cam_keys):
            if i in self.cam_indices:
                img = obs[key]
                cur_img = self._proc_image(img)
                if self.prev_img[key] is None:
                    self.prev_img[key] = torch.clone(cur_img)
                if self.img_chunk == 2:
                    images['cam'+str(i)] = torch.cat((cur_img, self.prev_img[key]), dim = 0).unsqueeze(0)
                else:
                    images['cam'+str(i)] = cur_img
                self.prev_img[key] = cur_img

        state = (state - self.ob_min)/(self.ob_max-self.ob_min)
        state = state*2 - 1
        if self.prev_obs is None:
            self.prev_obs = state

        self.prev_obs = state

        state = torch.from_numpy(state.astype(np.float32))[None].cuda()

        return images, state

    def get_action(self, obs, reward=0, pred_action=None, binary_gripper=True):
        """
        Called at the end of each step.
        """
        images, state = self._get_images_and_states(obs)

        if self.act_history:
            ac = self.act_history.pop(0)
        else:
            with torch.no_grad():
                action = self.agent.get_actions(images, state)
            ac_list = action[0].cpu().numpy().astype(np.float32)[:ACT_HORIZON]
            self.act_history = list(ac_list)
            ac = self.act_history.pop(0)

        if len(ac) == 7:
            ac = (ac+1)/2
            ac = ac*(self.ac_max-self.ac_min)+ self.ac_min

        assert len(ac) == 7, "Assuming 7d action dim!"
        if binary_gripper:
            ac[-1] = np.sign(ac[-1])
        return ac, None, None

    def get_action_with_guidance(self, obs, reward_fn, guidance_scale=1.5, ss=4, n_steps=None, guidance_every=1, guidance_min_alpha=0.0, grad_clip=None, lpb_scale=False, guidance_start_step=0, normalize_grad=False, binary_gripper=True):
        """
        Generate an action chunk via guided DDIM denoising and return the first action.

        Implements the DynaGuide (Du & Song, 2025) stochastic-sampling guidance scheme:
        at each denoising timestep the guidance gradient is applied ``ss`` times before
        proceeding to the next noise level.

        Args:
            obs: observation dict (same format as get_action).
            reward_fn: callable(x0_pred_normalized) -> scalar tensor with grad.
                x0_pred_normalized is (1, ac_chunk, ac_dim) in [-1, 1] policy space.
            guidance_scale: gradient multiplier (DynaGuide default: 1.5).
            ss: stochastic-sampling repetitions per timestep (DynaGuide default: 4).
            n_steps: override eval diffusion steps (None = use model default).
        Returns:
            (action, None, None) — same interface as get_action.
        """
        images, state = self._get_images_and_states(obs)

        if self.act_history:
            ac = self.act_history.pop(0)
        else:
            # Run guided diffusion for a single sample (B=1)
            time_start = time.time()
            actions = self.agent.get_actions_with_guidance(
                images, state, reward_fn,
                guidance_scale=guidance_scale,
                ss=ss,
                n_steps=n_steps,
                guidance_every=guidance_every,
                guidance_min_alpha=guidance_min_alpha,
                grad_clip=grad_clip,
                lpb_scale=lpb_scale,
                guidance_start_step=guidance_start_step,
                normalize_grad=normalize_grad,
            )
            ac_list = actions[0].detach().cpu().numpy().astype(np.float32)[:ACT_HORIZON]
            self.act_history = list(ac_list)
            ac = self.act_history.pop(0)
            time_end = time.time()
            print(f"Guided denoising took {time_end - time_start} seconds")
        if len(ac) == 7:
            ac = (ac + 1) / 2
            ac = ac * (self.ac_max - self.ac_min) + self.ac_min

        assert len(ac) == 7, "Assuming 7d action dim!"
        if binary_gripper:
            ac[-1] = np.sign(ac[-1])
        return ac, None, None

    def get_action_samples(self, obs, N, binary_gripper=True):
        """
        Return N predicted action samples from self.agent in parallel.
        Batches the observation N times, runs forward pass once, unnormalizes,
        and returns (N, ACT_HORIZON, 7) action chunks.
        """
        images, state = self._get_images_and_states(obs)

        # Batch observations for parallel sampling
        batch_images = {}
        for key, val in images.items():
            batch_images[key] = val.repeat(N, *([1] * (val.dim() - 1)))
        batch_state = state.repeat(N, 1)

        with torch.no_grad():
            actions = self.agent.get_actions(batch_images, batch_state)  # (N, ac_chunk, ac_dim)

        # Take ACT_HORIZON actions of each chunk -> (N, ACT_HORIZON, 7)
        ac = actions[:, :ACT_HORIZON, :].cpu().numpy().astype(np.float32)

        # Unnormalize
        ac = (ac + 1) / 2
        ac = ac * (self.ac_max - self.ac_min) + self.ac_min

        # Binarize gripper
        if binary_gripper:
            ac[:, :, -1] = np.sign(ac[:, :, -1])

        return ac  # (N, ACT_HORIZON, 7)

    def on_end_traj(self, traj_idx, traj_status):
        pass

    def reset_logger(self):
        pass

    def log_data(self, obs, action, reward=0):
        pass

    def log_obs(self, obs: dict, action: dict):
        pass

    def on_step(self, traj_idx, step_idx):
        return None
