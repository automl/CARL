from __future__ import annotations

from typing import Dict, List, Optional, Union

import jax.numpy as jnp
from gymnax.environments.classic_control.pendulum import EnvParams, Pendulum

from carl.envs.gymnax.carl_gymnax_env import CARLGymnaxEnv
from carl.utils.types import Context, Contexts

DEFAULT_CONTEXT = {
    "max_speed": 8.0,
    "max_torque": 2.0,
    "dt": 0.05,
    "g": 10.0,
    "m": 1.0,
    "l": 1.0,
    "max_steps_in_episode": 200,
}

CONTEXT_BOUNDS = {
    "max_speed": (-jnp.inf, jnp.inf, float),
    "max_torque": (-jnp.inf, jnp.inf, float),
    "dt": (0, jnp.inf, float),
    "g": (0, jnp.inf, float),
    "m": (1e-6, jnp.inf, float),
    "l": (1e-6, jnp.inf, float),
    "max_steps_in_episode": (1, jnp.inf, int),
}


class CARLJaxPendulumEnv(CARLGymnaxEnv):
    env_name: str = "Pendulum-v1"
    max_episode_steps: int = DEFAULT_CONTEXT["max_steps_in_episode"]
    DEFAULT_CONTEXT: Context = DEFAULT_CONTEXT

    def _update_context(self) -> None:
        self.env.env.env_params = EnvParams(**self.context)

        high = jnp.array(
            [1.0, 1.0, self.env.env.env_params.max_speed], dtype=jnp.float32
        )
        self.build_observation_space(-high, high, CONTEXT_BOUNDS)
