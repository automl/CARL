from __future__ import annotations

import jax.numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.gymnax.carl_gymnax_env import CARLGymnaxEnv


class CARLGymnaxMountainCar(CARLGymnaxEnv):
    env_name: str = "MountainCar-v0"
    module: str = "classic_control.mountain_car"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "max_speed": UniformFloatContextFeature(
                "max_speed", lower=1e-3, upper=10, default_value=0.07
            ),
            "goal_position": UniformFloatContextFeature(
                "goal_position", lower=-2, upper=2, default_value=0.45
            ),
            "goal_velocity": UniformFloatContextFeature(
                "goal_velocity", lower=-10, upper=10, default_value=0
            ),
            "force": UniformFloatContextFeature(
                "force", lower=-10, upper=10, default_value=0.001
            ),
            "gravity": UniformFloatContextFeature(
                "gravity", lower=-10, upper=10, default_value=0.0025
            ),
        }


class CARLGymnaxMountainCarContinuous(CARLGymnaxMountainCar):
    env_name: str = "MountainCarContinuous-v0"
    module: str = "classic_control.continuous_mountain_car"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "max_speed": UniformFloatContextFeature(
                "max_speed", lower=0, upper=np.inf, default_value=0.07
            ),
            "goal_position": UniformFloatContextFeature(
                "goal_position", lower=-np.inf, upper=np.inf, default_value=0.45
            ),
            "goal_velocity": UniformFloatContextFeature(
                "goal_velocity", lower=-np.inf, upper=np.inf, default_value=0
            ),
            "power": UniformFloatContextFeature(
                "power", lower=1e-6, upper=10, default_value=0.001
            ),
            "gravity": UniformFloatContextFeature(
                "gravity", lower=-10, upper=10, default_value=0.0025
            ),
        }
