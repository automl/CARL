from __future__ import annotations

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv


class CARLMountainCar(CARLGymnasiumEnv):
    env_name: str = "MountainCar-v0"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "min_position": UniformFloatContextFeature(
                "min_position", lower=-np.inf, upper=np.inf, default_value=-1.2
            ),
            "max_position": UniformFloatContextFeature(
                "max_position", lower=-np.inf, upper=np.inf, default_value=0.6
            ),
            "max_speed": UniformFloatContextFeature(
                "max_speed", lower=0, upper=np.inf, default_value=0.07
            ),
            "goal_position": UniformFloatContextFeature(
                "goal_position", lower=-np.inf, upper=np.inf, default_value=0.45
            ),
            "goal_velocity": UniformFloatContextFeature(
                "goal_velocity", lower=-np.inf, upper=np.inf, default_value=0
            ),
            "force": UniformFloatContextFeature(
                "force", lower=-np.inf, upper=np.inf, default_value=0.001
            ),
            "gravity": UniformFloatContextFeature(
                "gravity", lower=0, upper=np.inf, default_value=0.0025
            ),
            "min_position_start": UniformFloatContextFeature(
                "min_position_start", lower=-np.inf, upper=np.inf, default_value=-0.6
            ),
            "max_position_start": UniformFloatContextFeature(
                "max_position_start", lower=-np.inf, upper=np.inf, default_value=-0.4
            ),
            "min_velocity_start": UniformFloatContextFeature(
                "min_velocity_start", lower=-np.inf, upper=np.inf, default_value=0
            ),
            "max_velocity_start": UniformFloatContextFeature(
                "max_velocity_start", lower=-np.inf, upper=np.inf, default_value=0
            ),
        }
