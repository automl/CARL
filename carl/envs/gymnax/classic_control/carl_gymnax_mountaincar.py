from __future__ import annotations

import gymnax
import jax.numpy as jnp

from carl.envs.gymnax.carl_gymnax_env import CARLGymnaxEnv
from carl.utils.types import Context

DEFAULT_CONTEXT = {
    "min_position": -1.2,
    "max_position": 0.6,
    "max_speed": 0.07,
    "goal_position": 0.5,
    "goal_velocity": 0,
    "force": 0.001,
    "gravity": 0.0025,
    "max_steps_in_episode": 200,
}

CONTEXT_BOUNDS = {
    "min_position": (-jnp.inf, jnp.inf, float),
    "max_position": (-jnp.inf, jnp.inf, float),
    "max_speed": (0, jnp.inf, float),
    "goal_position": (-jnp.inf, jnp.inf, float),
    "goal_velocity": (-jnp.inf, jnp.inf, float),
    "force": (-jnp.inf, jnp.inf, float),
    "gravity": (0, jnp.inf, float),
    "max_steps_in_episode": (1, jnp.inf, int),
}


class CARLGymnaxMountainCar(CARLGymnaxEnv):
    env_name: str = "MountainCar-v0"
    max_episode_steps: int = int(DEFAULT_CONTEXT["max_steps_in_episode"])
    DEFAULT_CONTEXT: Context = DEFAULT_CONTEXT

    def _update_context(self) -> None:
        self.env.env.env_params = (
            gymnax.environments.classic_control.mountain_car.EnvParams(**self.context)
        )

        self.low = jnp.array(
            [self.env.env.env_params.min_position, -self.env.env.env_params.max_speed],
            dtype=jnp.float32,
        ).squeeze()
        self.high = jnp.array(
            [self.env.env.env_params.max_position, self.env.env.env_params.max_speed],
            dtype=jnp.float32,
        ).squeeze()

        self.build_observation_space(self.low, self.high, CONTEXT_BOUNDS)


class CARLGymnaxMountainCarContinuous(CARLGymnaxMountainCar):
    env_name: str = "MountainCarContinuous-v0"
    max_episode_steps: int = 999
    DEFAULT_CONTEXT: Context = DEFAULT_CONTEXT
