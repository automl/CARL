from __future__ import annotations

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxHopper(CARLBraxEnv):
    env_name: str = "hopper"
    asset_path: str = "envs/assets/hopper.xml"

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
            "mass_thigh": UniformFloatContextFeature(
                "mass_thigh", lower=1e-6, upper=np.inf, default_value=4.0578904
            ),
            "mass_leg": UniformFloatContextFeature(
                "mass_leg", lower=1e-6, upper=np.inf, default_value=2.7813568
            ),
            "mass_foot": UniformFloatContextFeature(
                "mass_foot", lower=1e-6, upper=np.inf, default_value=5.3155746
            ),
        }
