from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from brax.envs.ant import Ant


from carl.context.selection import AbstractSelector
# from carl.envs.carl_brax_env import CARLBraxEnv
from carl.utils.types import Context, Contexts
from carl.envs.brax.carl_brax_env import CARLBraxEnv

DEFAULT_CONTEXT = {
    "stiffness_factor": 1,
    "gravity": -9.81,
    "friction": 1,
    "damping_factor": 1,
    "actuator_strength_factor": 1,
    "torso_mass": 10,
    "dt": 0.01
}

CONTEXT_BOUNDS = {
    "stiffness_factor": (0, np.inf, float),
    "gravity": (-np.inf, -0.1, float),
    "friction": (-np.inf, np.inf, float),
    "damping_factor": (-np.inf, np.inf, float),
    "actuator_strength_factor": (1, np.inf, float),
    "torso_mass": (0.1, np.inf, float),
    "dt": (0.0001, 0.03, float),
}



class CARLAnt(CARLBraxEnv):
    env_name: str = "ant"
    DEFAULT_CONTEXT: Context = DEFAULT_CONTEXT

    def _update_context(self) -> None:
        self.env: Ant
        config = {}
        config["gravity"] = jnp.array([0, 0, self.context["gravity"]])
        config["dt"] = jnp.array(self.context["dt"])
        new_mass = self.env._env.sys.link.inertia.mass.at[0].set(self.context["torso_mass"])
        # TODO: do we want to implement this?
        #new_com = self.env.sys.link.inertia.transform
        #new_inertia = self.env.sys.link.inertia.i
        inertia = self.env._env.sys.link.inertia.replace(mass=new_mass)
        config["link"] = self.env._env.sys.link.replace(inertia=inertia)
        new_stiffness = self.context["stiffness_factor"]*self.env._env.sys.dof.stiffness
        new_damping = self.context["damping_factor"]*self.env._env.sys.dof.damping
        config["dof"] = self.env._env.sys.dof.replace(stiffness=new_stiffness, damping=new_damping)
        new_gear = self.context["actuator_strength_factor"]*self.env._env.sys.actuator.gear
        config["actuator"] = self.env._env.sys.actuator.replace(gear=new_gear)
        geoms = self.env._env.sys.geoms
        geoms[0] = geoms[0].replace(friction=jnp.array([self.context["friction"]]))
        config["geoms"] = geoms
        self.env._env.sys = self.env._env.sys.replace(**config)


# # NOTE: this is not up to date!
# class CARLBraxAnt(CARLBraxEnv):
#     def __init__(
#         self,
#         env: Ant = Ant(),
#         contexts: Contexts = {},
#         state_context_features: list[str] | None = None,
#         dict_observation: bool = False,
#         context_selector: AbstractSelector | type[AbstractSelector] | None = None,
#         context_selector_kwargs: dict = None,
#     ):
#         super().__init__(
#             env=env,
#             contexts=contexts,
#             state_context_features=state_context_features,
#             dict_observation=dict_observation,
#             context_selector=context_selector,
#             context_selector_kwargs=context_selector_kwargs
#         )

#         #self.base_config = MessageToDict(
#         #    text_format.Parse(_SYSTEM_CONFIG_SPRING, brax.Config())
#         #)

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