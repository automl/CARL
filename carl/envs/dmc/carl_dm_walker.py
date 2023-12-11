import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv


class CARLDmcWalkerEnv(CARLDmcEnv):
    domain = "walker"
    task = "walk_context"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "gravity": UniformFloatContextFeature(
                "gravity", lower=-np.inf, upper=-0.1, default_value=-9.81
            ),
            "friction_torsional": UniformFloatContextFeature(
                "friction_torsional", lower=0, upper=np.inf, default_value=1.0
            ),
            "friction_rolling": UniformFloatContextFeature(
                "friction_rolling", lower=0, upper=np.inf, default_value=1.0
            ),
            "friction_tangential": UniformFloatContextFeature(
                "friction_tangential", lower=0, upper=np.inf, default_value=1.0
            ),
            "timestep": UniformFloatContextFeature(
                "timestep", lower=0.001, upper=0.1, default_value=0.0025
            ),
            "joint_damping": UniformFloatContextFeature(
                "joint_damping", lower=0.0, upper=np.inf, default_value=1.0
            ),
            "joint_stiffness": UniformFloatContextFeature(
                "joint_stiffness", lower=0.0, upper=np.inf, default_value=0.0
            ),
            "actuator_strength": UniformFloatContextFeature(
                "actuator_strength", lower=0.0, upper=np.inf, default_value=1.0
            ),
            "density": UniformFloatContextFeature(
                "density", lower=0.0, upper=np.inf, default_value=0.0
            ),
            "viscosity": UniformFloatContextFeature(
                "viscosity", lower=0.0, upper=np.inf, default_value=0.0
            ),
            "geom_density": UniformFloatContextFeature(
                "geom_density", lower=0.0, upper=np.inf, default_value=1.0
            ),
            "wind_x": UniformFloatContextFeature(
                "wind_x", lower=-np.inf, upper=np.inf, default_value=0.0
            ),
            "wind_y": UniformFloatContextFeature(
                "wind_y", lower=-np.inf, upper=np.inf, default_value=0.0
            ),
            "wind_z": UniformFloatContextFeature(
                "wind_z", lower=-np.inf, upper=np.inf, default_value=0.0
            ),
        }
