from typing import Any, Dict, List, Optional, Union

import numpy as np

from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector
from carl.envs.dmc.wrappers import MujocoToGymWrapper
from carl.envs.dmc.utils import load_dmc_env
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv


DEFAULT_CONTEXT = {
    "joint_stiffness": 5000,
    "gravity_x": 0.,
    "gravity_y": 0.,
    "gravity_z": -9.81,
    "friction_tangential": 0.7,
    "friction_torsional": 0.1,
    "friction_rolling": 0.1,
    "actuator_strength": 1, # scale all actuators by this factor
    "joint_damping": 0.1,
    # "torso_mass": 10, # TODO find out if mass can be modified
    "timestep": 0.005,  # Seconds between updates
    "magnetic_x": 0., # TODO decide if this is useful
    "magnetic_y": -0.5, 
    "magnetic_z": 0.,
    "wind_x": 0., # TODO decide if this is useful
    "wind_y": 0.,
    "wind_z": 0.,
}

CONTEXT_BOUNDS = {
    "joint_stiffness": (1, np.inf, float),
    "gravity_x": (-0.1, -np.inf, float),
    "gravity_y": (-0.1, -np.inf, float),
    "gravity_z": (-0.1, -np.inf, float),
    "friction_tangential": (-np.inf, np.inf, float), # TODO can friction be negative here?
    "friction_torsional": (-np.inf, np.inf, float),
    "friction_rolling": (-np.inf, np.inf, float),
    "actuator_strength": (-np.inf, np.inf, float),
    "joint_damping": (0, np.inf, float),
    # "torso_mass": (0.1, np.inf, float),
    "timestep": (0.001, 0.1, float,),  # TODO not sure how much it should be varied
    "magnetic_x": (-np.inf, np.inf, float),
    "magnetic_y": (-np.inf, np.inf, float),
    "magnetic_z": (-np.inf, np.inf, float),
    "wind_x": (-np.inf, np.inf, float),
    "wind_y": (-np.inf, np.inf, float),
    "wind_z": (-np.inf, np.inf, float),
}

class CARLDmcQuadrupedEnv(CARLDmcEnv):
    def __init__(
        self,
        domain: str = "quadruped",
        task: str = "walk_context",
        contexts: Dict[Any, Dict[Any, Any]] = {},
        hide_context: bool = False,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Dict] = DEFAULT_CONTEXT,
        max_episode_length: int = 500,  # from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        state_context_features: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[Union[AbstractSelector, type(AbstractSelector)]] = None,
        context_selector_kwargs: Optional[Dict] = None,
    ):
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        self.domain = domain
        self.task = task
        if dict_observation_space:
            raise NotImplementedError
        else:
            env = load_dmc_env(domain_name=domain, task_name=task, context={}, environment_kwargs={"flat_observation": True})
            env = MujocoToGymWrapper(env)
        super().__init__(
            env=env,
            contexts=contexts,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features=scale_context_features,
            default_context=default_context,
            max_episode_length=max_episode_length,
            state_context_features=state_context_features,
            dict_observation_space=dict_observation_space,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
        )
        # TODO check gaussian noise on context features
        self.whitelist_gaussian_noise = list(
            DEFAULT_CONTEXT.keys()
        )  # allow to augment all values
