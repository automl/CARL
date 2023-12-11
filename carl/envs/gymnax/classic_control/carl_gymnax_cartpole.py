from __future__ import annotations

import gymnax
import jax.numpy as jnp

from carl.envs.gymnax.carl_gymnax_env import CARLGymnaxEnv
from carl.utils.types import Context

DEFAULT_CONTEXT = {
    "gravity": 9.8,
    "masscart": 1.0,
    "masspole": 0.1,
    "length": 0.5,
    "force_mag": 10.0,
    "tau": 0.02,
    "polemass_length": None,
    "total_mass": None,
    "max_steps_in_episode": 500,
}

CONTEXT_BOUNDS = {
    "gravity": (5.0, 15.0, float),
    "masscart": (0.5, 2.0, float),
    "masspole": (0.05, 0.2, float),
    "length": (0.25, 1.0, float),
    "force_mag": (5.0, 15.0, float),
    "tau": (0.01, 0.05, float),
    "polemass_length": (0, jnp.inf, float),
    "total_mass": (0, jnp.inf, float),
    "max_steps_in_episode": (1, jnp.inf, int),
}


class CARLGymnaxCartPole(CARLGymnaxEnv):
    env_name: str = "CartPole-v1"
    max_episode_steps: int = int(DEFAULT_CONTEXT["max_steps_in_episode"])  # type: ignore[arg-type]
    DEFAULT_CONTEXT: Context = DEFAULT_CONTEXT

    def _update_context(self) -> None:
        self.context["polemass_length"] = (
            self.context["masspole"] * self.context["length"]
        )
        self.context["total_mass"] = self.context["masscart"] + self.context["masspole"]

        self.env.env.env_params = (
            gymnax.environments.classic_control.cartpole.EnvParams(**self.context)
        )

        high = jnp.array(
            [
                self.env.env.env_params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                self.env.env.env_params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ],
            dtype=jnp.float32,
        )
        low = -high
        self.build_observation_space(low, high, CONTEXT_BOUNDS)
