from typing import Any, Dict, List, Optional, Union

import numpy as np
import jax.numpy as jnp
from brax.envs.pusher import Pusher
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
    "friction": 0.8,
    "damping_factor": 1,
    "actuator_strength_factor": 1,
    "dt": 0.01
}

CONTEXT_BOUNDS = {
    "stiffness_factor": (0, np.inf, float),
    "gravity": (-np.inf, -0.1, float),
    "friction": (-np.inf, np.inf, float),
    "damping_factor": (-np.inf, np.inf, float),
    "actuator_strength_factor": (1, np.inf, float),
    "dt": (0.0001, 0.03, float),
}



class CARLPusher(CARLBraxEnv):
    env_name: str = "pusher"
    DEFAULT_CONTEXT: Context = DEFAULT_CONTEXT

    def _update_context(self) -> None:
        self.env: Pusher
        config = {}
        config["gravity"] = jnp.array([0, 0, self.context["gravity"]])
        config["dt"] = jnp.array(self.context["dt"])
        new_stiffness = self.context["stiffness_factor"]*self.env._env.sys.dof.stiffness
        new_damping = self.context["damping_factor"]*self.env._env.sys.dof.damping
        config["dof"] = self.env._env.sys.dof.replace(stiffness=new_stiffness, damping=new_damping)
        new_gear = self.context["actuator_strength_factor"]*self.env._env.sys.actuator.gear
        config["actuator"] = self.env._env.sys.actuator.replace(gear=new_gear)
        geoms = self.env._env.sys.geoms
        geoms[0] = geoms[0].replace(friction=jnp.array([self.context["friction"]]))
        config["geoms"] = geoms
        self.env._env.sys = self.env._env.sys.replace(**config)
