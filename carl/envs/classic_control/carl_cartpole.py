from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
from gym.envs.classic_control import CartPoleEnv

from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector

DEFAULT_CONTEXT = {
    "gravity": 9.8,
    "masscart": 1.0,  # Should be seen as 100% and scaled accordingly
    "masspole": 0.1,  # Should be seen as 100% and scaled accordingly
    "pole_length": 0.5,  # Should be seen as 100% and scaled accordingly
    "force_magnifier": 10.0,
    "update_interval": 0.02,  # Seconds between updates
}

CONTEXT_BOUNDS = {
    "gravity": (0.1, np.inf, float),  # Positive gravity
    "masscart": (0.1, 10, float),  # Cart mass can be varied by a factor of 10
    "masspole": (0.01, 1, float),  # Pole mass can be varied by a factor of 10
    "pole_length": (0.05, 5, float),  # Pole length can be varied by a factor of 10
    "force_magnifier": (1, 100, int),  # Force magnifier can be varied by a factor of 10
    "update_interval": (
        0.002,
        0.2,
        float,
    ),  # Update interval can be varied by a factor of 10
}


class CARLCartPoleEnv(CARLEnv):
    def __init__(
        self,
        env: gym.Env = CartPoleEnv(),
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
        self.env.gravity = self.context["gravity"]
        self.env.masscart = self.context["masscart"]
        self.env.masspole = self.context["masspole"]
        self.env.length = self.context["pole_length"]
        self.env.force_mag = self.context["force_magnifier"]
        self.env.tau = self.context["update_interval"]

        high = np.array(
            [
                self.env.x_threshold * 2,
                np.finfo(np.float32).max,
                self.env.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        low = -high
        self.build_observation_space(low, high, CONTEXT_BOUNDS)
