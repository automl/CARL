from typing import Any, Dict, List, Optional, Union

import numpy as np

from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector
from carl.envs.dmc.wrappers import MujocoToGymWrapper
from carl.envs.dmc.utils import load_dmc_env
from carl.envs.carl_env import CARLEnv


"""
Physics options (defaults for CartPole):
|           apirate = 100.0                                                │
│         collision = 0                                                    │
│              cone = 0                                                    │
│           density = 0.0                                                  │
│      disableflags = 0                                                    │
│       enableflags = 0                                                    │
│           gravity = array([ 0.  ,  0.  , -9.81])                         │
│          impratio = 1.0                                                  │
│        integrator = 0                                                    │
│        iterations = 100                                                  │
│          jacobian = 2                                                    │
│          magnetic = array([ 0. , -0.5,  0. ])                            │
│    mpr_iterations = 50                                                   │
│     mpr_tolerance = 1e-06                                                │
│ noslip_iterations = 0                                                    │
│  noslip_tolerance = 1e-06                                                │
│          o_margin = 0.0                                                  │
│          o_solimp = array([9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00]) │
│          o_solref = array([0.02, 1.  ])                                  │
│            solver = 2                                                    │
│          timestep = 0.0025                                               │
│         tolerance = 1e-08                                                │
│         viscosity = 0.0                                                  │
│              wind = array([0., 0., 0.])                                  |
"""

DEFAULT_CONTEXT = {
    "gravity": [0., 0., -9.81],
    "masscart": 1.0,  # Should be seen as 100% and scaled accordingly
    "masspole": 0.1,  # Should be seen as 100% and scaled accordingly
    "pole_length": 1.0,  # Should be seen as 100% and scaled accordingly
    "force_magnifier": 10.0,
    "timestep": 0.01,  # Seconds between updates
    "magnetic": [0., -0.5, 0.],
    "wind": [0., 0., 0.],
}

CONTEXT_BOUNDS = {
    "gravity": [(0.1, -np.inf, float), (0.1, -np.inf, float), (0.1, -np.inf, float)],  # Negative gravity
    "masscart": (0.1, 10, float),  # Cart mass can be varied by a factor of 10
    "masspole": (0.01, 1, float),  # Pole mass can be varied by a factor of 10
    "pole_length": (0.05, 5, float),  # Pole length can be varied by a factor of 10
    "force_magnifier": (1, 100, int),  # Force magnifier can be varied by a factor of 10
    "timestep": (
        0.001,
        0.1,
        float,
    ),  # TODO not sure how much it can be varied ...Update interval can be varied by a factor of 10
    "magnetic": [(-np.inf, np.inf, float), (-np.inf, np.inf, float), (-np.inf, np.inf, float)],
    "wind": [(-np.inf, np.inf, float), (-np.inf, np.inf, float), (-np.inf, np.inf, float)],
}


class CARLDmcCartPoleEnv(CARLEnv):
    def __init__(
        self,
        domain: str = "cartpole",
        task: str = "swingup_context",
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
    
    def _update_context(self) -> None:
        if self.dict_observation_space:
            raise NotImplementedError
        else:
            env = load_dmc_env(domain_name=self.domain, task_name=self.task, context=self.context, environment_kwargs={"flat_observation": True})
            self.env = MujocoToGymWrapper(env)
