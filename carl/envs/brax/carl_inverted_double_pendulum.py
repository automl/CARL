from __future__ import annotations

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxInvertedDoublePendulum(CARLBraxEnv):
    env_name: str = "inverted_double_pendulum"
    asset_path: str = "envs/assets/inverted_double_pendulum.xml"
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
            "mass_cart": UniformFloatContextFeature(
                "mass_cart", lower=1e-6, upper=np.inf, default_value=1
            ),
            "mass_pole": UniformFloatContextFeature(
                "mass_pole", lower=1e-6, upper=np.inf, default_value=1
            ),
            "mass_pole2": UniformFloatContextFeature(
                "mass_pole", lower=1e-6, upper=np.inf, default_value=1
            ),
            "ang_damping": UniformFloatContextFeature(
                "ang_damping", lower=-np.inf, upper=np.inf, default_value=-0.05
            ),
            "viscosity": UniformFloatContextFeature(
                "viscosity", lower=0, upper=np.inf, default_value=0
            ),
        }
