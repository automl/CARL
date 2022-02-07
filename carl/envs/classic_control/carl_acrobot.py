from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
from gym.envs.classic_control import AcrobotEnv

from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector

DEFAULT_CONTEXT = {
    "link_length_1": 1,  # should be seen as 100% default and scaled
    "link_length_2": 1,  # should be seen as 100% default and scaled
    "link_mass_1": 1,  # should be seen as 100% default and scaled
    "link_mass_2": 1,  # should be seen as 100% default and scaled
    "link_com_1": 0.5,  # Percentage of the length of link one
    "link_com_2": 0.5,  # Percentage of the length of link one
    "link_moi": 1,  # should be seen as 100% default and scaled
    "max_velocity_1": 4 * np.pi,
    "max_velocity_2": 9 * np.pi,
    "torque_noise_max": 0.0,  # optional noise on torque, sampled uniformly from [-torque_noise_max, torque_noise_max]
}

CONTEXT_BOUNDS = {
    "link_length_1": (
        0.1,
        10,
        float,
    ),  # Links can be shrunken and grown by a factor of 10
    "link_length_2": (0.1, 10, float),
    "link_mass_1": (
        0.1,
        10,
        float,
    ),  # Link mass can be shrunken and grown by a factor of 10
    "link_mass_2": (0.1, 10, float),
    "link_com_1": (0, 1, float),  # Center of mass can move from one end to the other
    "link_com_2": (0, 1, float),
    "link_moi": (
        0.1,
        10,
        float,
    ),  # Moments on inertia can be shrunken and grown by a factor of 10
    "max_velocity_1": (
        0.4 * np.pi,
        40 * np.pi,
        float,
    ),  # Velocity can vary by a factor of 10 in either direction
    "max_velocity_2": (0.9 * np.pi, 90 * np.pi, float),
    "torque_noise_max": (
        -1.0,
        1.0,
        float,
    ),  # torque is either {-1., 0., 1}. Applying noise of 1. would be quite extreme
}


class CARLAcrobotEnv(CARLEnv):
    def __init__(
        self,
        env: gym.Env = AcrobotEnv(),
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
        self.whitelist_gaussian_noise = list(
            DEFAULT_CONTEXT.keys()
        )  # allow to augment all values

    def _update_context(self) -> None:
        self.env.LINK_LENGTH_1 = self.context["link_length_1"]
        self.env.LINK_LENGTH_2 = self.context["link_length_2"]
        self.env.LINK_MASS_1 = self.context["link_mass_1"]
        self.env.LINK_MASS_2 = self.context["link_mass_2"]
        self.env.LINK_COM_POS_1 = self.context["link_com_1"]
        self.env.LINK_COM_POS_2 = self.context["link_com_2"]
        self.env.LINK_MOI = self.context["link_moi"]
        self.env.MAX_VEL_1 = self.context["max_velocity_1"]
        self.env.MAX_VEL_2 = self.context["max_velocity_2"]
        self.env.torque_noise_max = self.context["torque_noise_max"]

        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.env.MAX_VEL_1, self.env.MAX_VEL_2],
            dtype=np.float32,
        )
        low = -high
        self.build_observation_space(low, high, CONTEXT_BOUNDS)
