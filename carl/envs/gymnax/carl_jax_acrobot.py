from __future__ import annotations

from typing import Dict, List, Optional, Union

import gymnasium
import gymnax
import jax.numpy as jnp
import numpy as np
from gymnax.environments.classic_control.acrobot import Acrobot

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.envs.gymnax.carl_gymnax_env import CARLGymnaxEnv
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts

DEFAULT_CONTEXT = {
    "link_length_1": 1,
    "link_length_2": 1,
    "link_mass_1": 1,
    "link_mass_2": 1,
    "link_com_pos_1": 0.5,
    "link_com_pos_2": 0.5,
    "link_moi": 1,
    "max_vel_1": 4 * jnp.pi,
    "max_vel_2": 9 * jnp.pi,
    "torque_noise_max": 0.0,
    "max_steps_in_episode": 500,
}

CONTEXT_BOUNDS = {
    "link_length_1": (
        0.1,
        10,
        float,
    ),  # Links can be shrunken and grown by a factor of 10
    "link_length_2": (0.1, 10, float),
    "link_mass_1": (
        0.1,
        10,
        float,
    ),  # Link mass can be shrunken and grown by a factor of 10
    "link_mass_2": (0.1, 10, float),
    "link_com_pos_1": (
        0,
        1,
        float,
    ),  # Center of mass can move from one end to the other
    "link_com_pos_2": (0, 1, float),
    "link_moi": (
        0.1,
        10,
        float,
    ),  # Moments on inertia can be shrunken and grown by a factor of 10
    "max_vel_1": (
        0.4 * np.pi,
        40 * np.pi,
        float,
    ),  # Velocity can vary by a factor of 10 in either direction
    "max_vel_2": (0.9 * np.pi, 90 * np.pi, float),
    "torque_noise_max": (
        -1.0,
        1.0,
        float,
    ),  # torque is either {-1., 0., 1}. Applying noise of 1. would be quite extreme
    "max_steps_in_episode": (1, jnp.inf, int),
}


class CARLJaxAcrobotEnv(CARLGymnaxEnv):
    env_name: str = "Acrobot-v1"
    max_episode_steps: int = DEFAULT_CONTEXT["max_steps_in_episode"]
    DEFAULT_CONTEXT: Context = DEFAULT_CONTEXT

    def _update_context(self) -> None:
        content = self.env.env.env_params.__dict__
        content.update(self.context)
        # We cannot directly set attributes of env_params because it is a frozen dataclass
        self.env.env.env_params = gymnax.environments.classic_control.acrobot.EnvParams(
            **content
        )

        high = jnp.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                self.env.env.env_params.max_vel_1,
                self.env.env.env_params.max_vel_2,
            ],
            dtype=jnp.float32,
        )
        low = -high
        self.build_observation_space(low, high, CONTEXT_BOUNDS)
