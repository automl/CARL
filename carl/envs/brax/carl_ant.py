from __future__ import annotations

import numpy as np

from carl.context.context_space import (
    CategoricalContextFeature,
    ContextFeature,
    UniformFloatContextFeature,
)
from carl.envs.brax.brax_walker_goal_wrapper import directions
from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxAnt(CARLBraxEnv):
    env_name: str = "ant"
    asset_path: str = "envs/assets/ant.xml"
    metadata = {"render_modes": []}

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "gravity": UniformFloatContextFeature(
                "gravity", lower=-1000, upper=-1e-6, default_value=-9.8
            ),
            "friction": UniformFloatContextFeature(
                "friction", lower=0, upper=100, default_value=1
            ),
            "elasticity": UniformFloatContextFeature(
                "elasticity", lower=0, upper=100, default_value=0
            ),
            "ang_damping": UniformFloatContextFeature(
                "ang_damping", lower=-np.inf, upper=np.inf, default_value=-0.05
            ),
            "mass_torso": UniformFloatContextFeature(
                "mass_torso", lower=1e-6, upper=np.inf, default_value=10
            ),
            "viscosity": UniformFloatContextFeature(
                "viscosity", lower=0, upper=np.inf, default_value=0
            ),
            "target_distance": UniformFloatContextFeature(
                "target_distance", lower=0, upper=np.inf, default_value=100
            ),
            "target_direction": CategoricalContextFeature(
                "target_direction", choices=directions, default_value=1
            ),
            "target_radius": UniformFloatContextFeature(
                "target_radius", lower=0.1, upper=np.inf, default_value=5
            ),
        }
