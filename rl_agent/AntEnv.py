import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

from rl_agent.environment import CustomEnvironment

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


def define_function_from_string(
    function_string: str,
) -> Tuple[Optional[Callable], List[str]]:
    """Define a callable function from a string.

    Args:
        function_string: A string representation of the reward function.

    Returns:
        A tuple of (callable function, list of arguments).
    """
    import inspect
    import torch

    namespace = {}
    additional_globals = {
        "torch": torch,
        "np": np,
        "Tuple": Tuple,
        "List": List,
        "Callable": Callable,
        "Optional": Optional,
        "Dict": Dict,
    }
    namespace.update(additional_globals)
    exec(function_string, namespace)
    function = next(
        (
            val for key, val in namespace.items()
            if key == "compute_reward"
        ),
        None,
    )
    args = inspect.getfullargspec(function).args if function else []
    return function, args


def call_reward_func_dynamically(reward_func, env_state):
    """Call the reward function dynamically using arguments in env_state.

    Args:
        reward_func: The callable reward function.
        env_state: A dictionary containing environment state variables.

    Returns:
        A tuple of (reward, reward_components).
    """
    import inspect

    params = inspect.signature(reward_func).parameters
    args_to_pass = {
        param: env_state[param]
        for param in params
        if param in env_state
    }
    reward, reward_components = reward_func(**args_to_pass)
    return reward, reward_components


class AntEnv(MujocoEnv, utils.EzPickle):
    """Custom Ant environment that computes rewards using an LLM-generated function."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        reward_func_str: str,
        counter: int,
        generation_id: int,
        island_id: str,
        reward_history_file: str,
        velocity_file: str,
        model_checkpoint_file: str,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            reward_func_str,
            counter,
            generation_id,
            island_id,
            reward_history_file,
            velocity_file,
            model_checkpoint_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self.reward_func, _ = define_function_from_string(reward_func_str)
        self.counter = counter
        self.iteration = generation_id
        self.island_id = island_id
        self.reward_history_file = reward_history_file
        self.model_checkpoint_file = model_checkpoint_file
        self.velocity_file = velocity_file

        os.makedirs(os.path.dirname(self.reward_history_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_checkpoint_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.velocity_file), exist_ok=True)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._use_contact_forces = use_contact_forces
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.total_steps = 0
        self.custom_env = CustomEnvironment()
        self.rewards = []
        self.reward_components_log = {}

        obs_shape = 27
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            "ant.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = (
            (not self.is_healthy)
            if self._terminate_when_unhealthy
            else False
        )
        if self.total_steps > 1000:
            terminated = True
            self.total_steps = 0
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate((position, velocity))
        return observation

    def step(self, action):
        self.total_steps = self.total_steps + 1
        xy_position_before = self.get_body_com("torso")[:2].copy()

        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        observation = self._get_obs()
        self.custom_env.update_state(observation)

        reward, reward_components = call_reward_func_dynamically(
            self.reward_func, self.custom_env.env_state
        )

        self.rewards.append(reward)
        for key, value in reward_components.items():
            if key not in self.reward_components_log:
                self.reward_components_log[key] = []
            self.reward_components_log[key].append(value)

        terminated = self.terminated

        info = {
            "reward_components": reward_components,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }
        if terminated:
            info["episode"] = {
                "r": sum(self.rewards),
                "l": len(self.rewards),
                "t": time.time() - self.episode_start_time,
            }

            episode_summary = {
                "total_reward": sum(self.rewards),
                "episode_components": {
                    key: sum(values)
                    for key, values in self.reward_components_log.items()
                },
            }

            with open(self.reward_history_file, "a") as file:
                json.dump(episode_summary, file)
                file.write("\n")

            self.rewards = []
            self.reward_components_log = {
                key: [] for key in reward_components.keys()
            }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        self.episode_start_time = time.time()
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        self.rewards = []
        self.reward_components_log = {}

        observation = self._get_obs()
        self.custom_env.update_state(observation)

        return observation
