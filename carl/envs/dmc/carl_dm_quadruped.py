from typing import Any, Dict, List, Optional, Union

import numpy as np

from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv


DEFAULT_CONTEXT = {
    "gravity": -9.81,
    "friction_tangential": 1.,  # Scaling factor for tangential friction of all geoms (objects)
    "friction_torsional": 1.,  # Scaling factor for torsional friction of all geoms (objects)
    "friction_rolling": 1.,  # Scaling factor for rolling friction of all geoms (objects)
    "timestep": 0.005,  # Seconds between updates
    "joint_damping": 1.,  # Scaling factor for all joints
    "joint_stiffness": 0.,
    "actuator_strength": 1,  # Scaling factor for all actuators in the model
    "density": 0.,
    "viscosity": 0.,
    "geom_density": 1.,  # Scaling factor for all geom (objects) densities
    "wind_x": 0.,
    "wind_y": 0.,
    "wind_z": 0.,
}

CONTEXT_BOUNDS = {
    "gravity": (-0.1, -np.inf, float),
    "friction_tangential": (0, np.inf, float),
    "friction_torsional": (0, np.inf, float),
    "friction_rolling": (0, np.inf, float),
    "timestep": (0.001, 0.1, float,),
    "joint_damping": (0, np.inf, float),
    "joint_stiffness": (0, np.inf, float),
    "actuator_strength": (0, np.inf, float),
    "density": (0, np.inf, float),
    "viscosity": (0, np.inf, float),
    "geom_density": (0, np.inf, float),
    "wind_x": (-np.inf, np.inf, float),
    "wind_y": (-np.inf, np.inf, float),
    "wind_z": (-np.inf, np.inf, float),
}

CONTEXT_MASK = [
    "wind_x",
    "wind_y",
    "wind_z",
]


class CARLDmcQuadrupedEnv(CARLDmcEnv):
    def __init__(
        self,
        domain: str = "quadruped",
        task: str = "walk_context",
        contexts: Dict[Any, Dict[Any, Any]] = {},
        context_mask: Optional[List[str]] = [],
        hide_context: bool = True,
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
        super().__init__(
            domain=domain,
            task=task,
            contexts=contexts,
            context_mask=context_mask,
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
