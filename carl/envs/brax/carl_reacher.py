from typing import Any, Dict, List, Optional, Union

import numpy as np
import jax.numpy as jnp
from brax.envs.reacher import Reacher
from brax.envs import create
from carl.envs.braxenvs.brax_wrappers import GymWrapper, VectorGymWrapper

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts
from carl.envs.braxenvs.carl_brax_env import CARLBraxEnv

DEFAULT_CONTEXT = {
    "stiffness_factor": 1,
    "gravity": -9.81,
    "friction": 1,
    "damping_factor": 1,
    "actuator_strength_factor": 1,
    "body_mass_0": 0.036,
    "body_mass_1": 0.04,
    "dt": 0.01
}

CONTEXT_BOUNDS = {
    "stiffness_factor": (0, np.inf, float),
    "gravity": (-np.inf, -0.1, float),
    "friction": (-np.inf, np.inf, float),
    "damping_factor": (-np.inf, np.inf, float),
    "actuator_strength_factor": (1, np.inf, float),
    "body_mass_0": (0.1, np.inf, float),
    "body_mass_1": (0.1, np.inf, float),
    "dt": (0.0001, 0.03, float),
}



class CARLReacher(CARLBraxEnv):
    env_name: str = "reacher"
    DEFAULT_CONTEXT: Context = DEFAULT_CONTEXT

    def _update_context(self) -> None:
        self.env: Reacher
        config = {}
        config["gravity"] = jnp.array([0, 0, self.context["gravity"]])
        config["dt"] = jnp.array(self.context["dt"])
        new_mass = self.env._env.sys.link.inertia.mass.at[0].set(self.context["body_mass_0"])
        new_mass = new_mass.at[1].set(self.context["body_mass_1"])
        # TODO: do we wReacher to implement this?
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
