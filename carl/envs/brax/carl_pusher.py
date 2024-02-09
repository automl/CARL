from __future__ import annotations

from copy import deepcopy

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxPusher(CARLBraxEnv):
    env_name: str = "pusher"
    asset_path: str = "envs/assets/pusher.xml"
    metadata = {"render_modes": []}

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            # General physics
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
            # Masses of the pusher robot
            "mass_r_shoulder_pan_link": UniformFloatContextFeature(
                "mass_r_shoulder_pan_link",
                lower=1e-6,
                upper=np.inf,
                default_value=7.2935214e00,
            ),
            "mass_r_shoulder_lift_link": UniformFloatContextFeature(
                "mass_r_shoulder_lift_link",
                lower=1e-6,
                upper=np.inf,
                default_value=np.pi,
            ),
            "mass_r_upper_arm_roll_link": UniformFloatContextFeature(
                "mass_r_upper_arm_roll_link",
                lower=1e-6,
                upper=np.inf,
                default_value=1.7140529,
            ),
            "mass_r_elbow_flex_link": UniformFloatContextFeature(
                "mass_r_elbow_flex_link",
                lower=1e-6,
                upper=np.inf,
                default_value=4.0715042e-01,
            ),
            "mass_r_forearm_roll_link": UniformFloatContextFeature(
                "mass_r_forearm_roll_link",
                lower=1e-6,
                upper=np.inf,
                default_value=9.2818356e-01,
            ),
            "mass_r_wrist_flex_link": UniformFloatContextFeature(
                "mass_r_wrist_flex_link",
                lower=1e-6,
                upper=np.inf,
                default_value=5.0265482e-03,
            ),
            "mass_r_wrist_roll_link": UniformFloatContextFeature(
                "mass_r_wrist_roll_link",
                lower=1e-6,
                upper=np.inf,
                default_value=1.8346901e-01,
            ),
            # Mass of the object to be pushed
            "mass_object": UniformFloatContextFeature(
                "mass_object", lower=1e-6, upper=np.inf, default_value=1.8325957e-03
            ),
            "goal_position_x": UniformFloatContextFeature(
                "goal_position_x", lower=0, upper=np.inf, default_value=0.45
            ),
            "goal_position_y": UniformFloatContextFeature(
                "goal_position_y", lower=0, upper=np.inf, default_value=0.05
            ),
            "goal_position_z": UniformFloatContextFeature(
                "goal_position_z", lower=0, upper=np.inf, default_value=0.05
            ),
        }

    def _update_context(self) -> None:
        goal_x = self.context["goal_position_x"]
        goal_y = self.context["goal_position_y"]
        goal_z = self.context["goal_position_z"]
        context = deepcopy(self.context)
        del self.context["goal_position_x"]
        del self.context["goal_position_y"]
        del self.context["goal_position_z"]
        super()._update_context()
        self.env._goal_pos = np.array([goal_x, goal_y, goal_z])
        self.context = context
