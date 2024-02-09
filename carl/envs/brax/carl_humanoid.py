from __future__ import annotations

import numpy as np

from carl.context.context_space import (
    CategoricalContextFeature,
    ContextFeature,
    UniformFloatContextFeature,
)
from carl.envs.brax.brax_walker_goal_wrapper import directions
from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxHumanoid(CARLBraxEnv):
    env_name: str = "humanoid"
    asset_path: str = "envs/assets/humanoid.xml"
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
            "viscosity": UniformFloatContextFeature(
                "viscosity", lower=0, upper=np.inf, default_value=0
            ),
            "mass_torso": UniformFloatContextFeature(
                "mass_torso", lower=1e-6, upper=np.inf, default_value=10
            ),
            "mass_lwaist": UniformFloatContextFeature(
                "mass_lwaist", lower=1e-6, upper=np.inf, default_value=2.2619467
            ),
            "mass_pelvis": UniformFloatContextFeature(
                "mass_pelvis", lower=1e-6, upper=np.inf, default_value=6.6161942
            ),
            "mass_right_thigh": UniformFloatContextFeature(
                "mass_right_thigh", lower=1e-6, upper=np.inf, default_value=4.751751
            ),
            "mass_right_shin": UniformFloatContextFeature(
                "mass_right_shin", lower=1e-6, upper=np.inf, default_value=4.522842
            ),
            "mass_left_thigh": UniformFloatContextFeature(
                "mass_left_thigh", lower=1e-6, upper=np.inf, default_value=4.751751
            ),
            "mass_left_shin": UniformFloatContextFeature(
                "mass_left_shin", lower=1e-6, upper=np.inf, default_value=4.522842
            ),
            "mass_right_upper_arm": UniformFloatContextFeature(
                "mass_right_upper_arm",
                lower=1e-6,
                upper=np.inf,
                default_value=1.6610805,
            ),
            "mass_right_lower_arm": UniformFloatContextFeature(
                "mass_right_lower_arm",
                lower=1e-6,
                upper=np.inf,
                default_value=1.2295402,
            ),
            "mass_left_upper_arm": UniformFloatContextFeature(
                "mass_left_upper_arm", lower=1e-6, upper=np.inf, default_value=1.6610805
            ),
            "mass_left_lower_arm": UniformFloatContextFeature(
                "mass_left_lower_arm", lower=1e-6, upper=np.inf, default_value=1.2295402
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
