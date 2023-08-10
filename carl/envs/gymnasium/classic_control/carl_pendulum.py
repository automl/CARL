from __future__ import annotations

from typing import Optional
import numpy as np
from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv


class CARLPendulum(CARLGymnasiumEnv):
    env_name: str = "Pendulum-v1"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "gravity": UniformFloatContextFeature(
                "gravity", lower=-np.inf, upper=np.inf, default_value=8.0
            ),
            "dt": UniformFloatContextFeature(
                "dt", lower=0, upper=np.inf, default_value=0.05
            ),
            "g": UniformFloatContextFeature(
                "g", lower=0, upper=np.inf, default_value=10
            ),
            "m": UniformFloatContextFeature(
                "m", lower=1e-6, upper=np.inf, default_value=1
            ),
            "l": UniformFloatContextFeature(
                "l", lower=1e-6, upper=np.inf, default_value=1
            ),
            "initial_angle_max": UniformFloatContextFeature(
                "initial_angle_max", lower=0, upper=np.inf, default_value=np.pi
            ),
            "initial_velocity_max": UniformFloatContextFeature(
                "initial_velocity_max", lower=0, upper=np.inf, default_value=1
            ),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        theta = self.env.np_random.uniform(high=self.context["initial_angle_max"])
        thetadot = self.env.np_random.uniform(high=self.context["initial_velocity_max"])
        self.env.unwrapped.state = np.array([theta, thetadot], dtype=np.float32)
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32), {}
