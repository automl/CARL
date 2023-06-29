from __future__ import annotations

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv


class CARLAcrobot(CARLGymnasiumEnv):
    env_name: str = "Acrobot-v1"

    def _update_context(self) -> None:
        self.env.LINK_LENGTH_1 = self.context["link_length_1"]
        self.env.LINK_LENGTH_2 = self.context["link_length_2"]
        self.env.LINK_MASS_1 = self.context["link_mass_1"]
        self.env.LINK_MASS_2 = self.context["link_mass_2"]
        self.env.LINK_COM_POS_1 = self.context["link_com_1"]
        self.env.LINK_COM_POS_2 = self.context["link_com_2"]
        self.env.LINK_MOI = self.context["link_moi"]
        self.env.MAX_VEL_1 = self.context["max_velocity_1"]
        self.env.MAX_VEL_2 = self.context["max_velocity_2"]
        self.env.torque_noise_max = self.context["torque_noise_max"]
        self.env.INITIAL_ANGLE_LOWER = self.context["initial_angle_lower"]
        self.env.INITIAL_ANGLE_UPPER = self.context["initial_angle_upper"]
        self.env.INITIAL_VELOCITY_LOWER = self.context["initial_velocity_lower"]
        self.env.INITIAL_VELOCITY_UPPER = self.context["initial_velocity_upper"]

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "link_length_1": UniformFloatContextFeature(
                "link_length_1", lower=0.1, upper=10, default_value=1
            ),
            "link_length_2": UniformFloatContextFeature(
                "link_length_2", lower=0.1, upper=10, default_value=1
            ),
            "link_mass_1": UniformFloatContextFeature(
                "link_mass_1", lower=0.1, upper=10, default_value=1
            ),
            "link_mass_2": UniformFloatContextFeature(
                "link_mass_2", lower=0.1, upper=10, default_value=1
            ),
            "link_com_1": UniformFloatContextFeature(
                "link_com_1", lower=0, upper=1, default_value=0.5
            ),
            "link_com_2": UniformFloatContextFeature(
                "link_com_2", lower=0, upper=1, default_value=0.5
            ),
            "link_moi": UniformFloatContextFeature(
                "link_moi", lower=0.1, upper=10, default_value=1
            ),
            "max_velocity_1": UniformFloatContextFeature(
                "max_velocity_1", lower=0.4 * np.pi, upper=40 * np.pi, default_value=4 * np.pi
            ),
            "max_velocity_2": UniformFloatContextFeature(
                "max_velocity_2", lower=0.9 * np.pi, upper=90 * np.pi, default_value=9 * np.pi
            ),
            "torque_noise_max": UniformFloatContextFeature(
                "torque_noise_max", lower=-1, upper=1, default_value=0
            ),
            "initial_angle_lower": UniformFloatContextFeature(
                "initial_angle_lower", lower=-np.inf, upper=np.inf, default_value=-0.1
            ),
            "initial_angle_upper": UniformFloatContextFeature(
                "initial_angle_upper", lower=-np.inf, upper=np.inf, default_value=0.1
            ),
            "initial_velocity_lower": UniformFloatContextFeature(
                "initial_velocity_lower", lower=-np.inf, upper=np.inf, default_value=-0.1
            ),
            "initial_velocity_upper": UniformFloatContextFeature(
                "initial_velocity_upper", lower=-np.inf, upper=np.inf, default_value=0.1
            ),
        }
