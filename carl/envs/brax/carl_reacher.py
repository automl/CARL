from __future__ import annotations

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxReacher(CARLBraxEnv):
    env_name: str = "reacher"
    asset_path: str = "envs/assets/reacher.xml"
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
            "mass_body0": UniformFloatContextFeature(
                "mass_body0", lower=1e-6, upper=np.inf, default_value=0.03560472
            ),
            "mass_body1": UniformFloatContextFeature(
                "mass_body1", lower=1e-6, upper=np.inf, default_value=0.03979351
            ),
        }
