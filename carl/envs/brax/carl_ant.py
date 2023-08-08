from __future__ import annotations
from __future__ import annotations

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxAnt(CARLBraxEnv):
    env_name: str = "ant"
    asset_path: str = "envs/assets/ant.xml"

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
        }

#     def _update_context(self) -> None:
#         #self.env: Ant
#         config = {}#copy.deepcopy(self.base_config)
#         config["gravity"] = jnp.array([0, 0, self.context["gravity"]])
#         #config["friction"] = self.context["friction"]
#         config["dt"] = self.context["dt"]
#         #for j in range(len(config["joints"])):
#         #    config["joints"][j]["angularDamping"] = self.context[
#         #        "joint_angular_damping"
#         #    ]
#         #    config["joints"][j]["stiffness"] = self.context["joint_stiffness"]
#         #for a in range(len(config["actuators"])):
#         #    config["actuators"][a]["strength"] = self.context["actuator_strength"]
#         #config["bodies"][0]["mass"] = self.context["torso_mass"]
#         # This converts the dict to a JSON String, then parses it into an empty brax config
#         #self.env.sys = brax.System(
#         #    json_format.Parse(json.dumps(config), brax.Config())
#         #)
#         self.env.sys = self.env.sys.replace(**config)