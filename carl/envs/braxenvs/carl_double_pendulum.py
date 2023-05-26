from typing import Any, Dict, List, Optional, Union

import numpy as np
import jax.numpy as jnp
from brax.envs import create
from brax.envs.inverted_double_pendulum import InvertedDoublePendulum
from carl.envs.braxenvs.brax_wrappers import GymWrapper, VectorGymWrapper

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts

DEFAULT_CONTEXT = {
    "stiffness_factor": 1,
    "gravity_x": 1e-5,
    "gravity_z": -9.81,
    "friction": 0.8,
    "damping_factor": 1,
    "actuator_strength_factor": 1,
    "mass_cart": 10.5,
    "mass_pole_0": 4.2,
    "mass_pole_1": 4.2,
    "dt": 0.01
}

CONTEXT_BOUNDS = {
    "stiffness_factor": (0, np.inf, float),
    "gravity_x": (-np.inf, np.inf, float),
    "gravity_z": (-np.inf, -0.1, float),
    "friction": (-np.inf, np.inf, float),
    "damping_factor": (-np.inf, np.inf, float),
    "actuator_strength_factor": (1, np.inf, float),
    "mass_cart": (0.1, np.inf, float),
    "mass_pole_0": (0.1, np.inf, float),
    "mass_pole_1": (0.1, np.inf, float),
    "dt": (0.0001, 0.03, float),
}



class CARLInvertedDoublePendulum(CARLEnv):
    def __init__(
        self,
        env: InvertedDoublePendulum = None,
        n_envs: int = 1,
        contexts: Contexts = {},
        hide_context: bool = True,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Context] = DEFAULT_CONTEXT,
        state_context_features: Optional[List[str]] = None,
        context_mask: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[
            Union[AbstractSelector, type[AbstractSelector]]
        ] = None,
        context_selector_kwargs: Optional[Dict] = None,
        max_episode_length = 1000,
    ):
        if env is None:
            env = create("inverted_double_pendulum", batch_size=n_envs)

        self.n_envs=n_envs
        if n_envs == 1:
            env = GymWrapper(env)
        else:
            env = VectorGymWrapper(env, n_envs)

        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        super().__init__(
            env=env,
            n_envs=n_envs,
            contexts=contexts,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features=scale_context_features,
            default_context=default_context,
            state_context_features=state_context_features,
            dict_observation_space=dict_observation_space,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            context_mask=context_mask,
            max_episode_length=max_episode_length,
        )
        self.whitelist_gaussian_noise = list(
            DEFAULT_CONTEXT.keys()
        )  # allow to augment all values

    def _update_context(self) -> None:
        self.env: InvertedDoublePendulum
        config = {}
        config["gravity"] = jnp.array([self.context["gravity_x"], 0, self.context["gravity_z"]])
        config["dt"] = jnp.array(self.context["dt"])
        new_mass = self.env._env.sys.link.inertia.mass.at[0].set(self.context["mass_cart"])
        new_mass = new_mass.at[1].set(self.context["mass_pole_0"])
        new_mass = new_mass.at[2].set(self.context["mass_pole_1"])
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

    def __getattr__(self, name: str) -> Any:
        if name in ["sys", "__getstate__"]:
            return getattr(self.env._environment, name)
        else:
            return getattr(self, name)