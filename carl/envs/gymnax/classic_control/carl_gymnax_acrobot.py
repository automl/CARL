from __future__ import annotations

import gymnax
import jax.numpy as jnp
import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.gymnax.carl_gymnax_env import CARLGymnaxEnv


class CARLGymnaxAcrobot(CARLGymnaxEnv):
    env_name: str = "Acrobot-v1"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "link_length_1": UniformFloatContextFeature(
                "link_length_1", lower=0.1, upper=10, default_value=1
            ),  # Links can be shrunken and grown by a factor of 10
            "link_length_2": UniformFloatContextFeature(
                "link_length_2", lower=0.1, upper=10, default_value=1
            ),  # Links can be shrunken and grown by a factor of 10
            "link_mass_1": UniformFloatContextFeature(
                "link_mass_1", lower=0.1, upper=10, default_value=1
            ),  # Link mass can be shrunken and grown by a factor of 10
            "link_mass_2": UniformFloatContextFeature(
                "link_mass_2", lower=0.1, upper=10, default_value=1
            ),  # Link mass can be shrunken and grown by a factor of 10
            "link_com_pos_1": UniformFloatContextFeature(
                "link_com_pos_1", lower=0, upper=1, default_value=0.5
            ),  # Center of mass can move from one end to the other
            "link_com_pos_2": UniformFloatContextFeature(
                "link_com_pos_2", lower=0, upper=1, default_value=0.5
            ),  # Center of mass can move from one end to the other
            "link_moi": UniformFloatContextFeature(
                "link_moi", lower=0.1, upper=10, default_value=1
            ),  # Moments on inertia can be shrunken and grown by a factor of 10
            "max_vel_1": UniformFloatContextFeature(
                "max_vel_1",
                lower=0.4 * jnp.pi,
                upper=40 * jnp.pi,
                default_value=4 * jnp.pi,
            ),  # Velocity can vary by a factor of 10 in either direction
            "max_vel_2": UniformFloatContextFeature(
                "max_vel_2",
                lower=0.9 * np.pi,
                upper=90 * np.pi,
                default_value=9 * np.pi,
            ),  # Velocity can vary by a factor of 10 in either direction
            "torque_noise_max": UniformFloatContextFeature(
                "torque_noise_max", lower=-1, upper=1, default_value=0
            ),  # torque is either {-1., 0., 1}. Applying noise of 1. would be quite extreme
        }

    def _update_context(self) -> None:
        content = self.env.env_params.__dict__
        content.update(self.context)
        # We cannot directly set attributes of env_params because it is a frozen dataclass
        self.env.env.env_params = gymnax.environments.classic_control.acrobot.EnvParams(
            **content
        )
