from typing import Dict, List, Optional, Union

import gymnasium.envs.classic_control as gccenvs
import numpy as np

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts

DEFAULT_CONTEXT = {
    "max_speed": 8.0,
    "dt": 0.05,
    "g": 10.0,
    "m": 1.0,
    "l": 1.0,
    "initial_angle_max": np.pi,  # Upper bound for uniform distribution to sample from
    "initial_velocity_max": 1,  # Upper bound for uniform distribution to sample from
    # The lower bound will be the negative value.
}

CONTEXT_BOUNDS = {
    "max_speed": (-np.inf, np.inf, float),
    "dt": (0, np.inf, float),
    "g": (0, np.inf, float),
    "m": (1e-6, np.inf, float),
    "l": (1e-6, np.inf, float),
    "initial_angle_max": (0, np.inf, float),
    "initial_velocity_max": (0, np.inf, float),
}


class CustomPendulum(gccenvs.pendulum.PendulumEnv):
    def __init__(self, g: float = 10.0):
        super(CustomPendulum, self).__init__(g=g)
        self.initial_angle_max = DEFAULT_CONTEXT["initial_angle_max"]
        self.initial_velocity_max = DEFAULT_CONTEXT["initial_velocity_max"]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
        super().reset(seed=seed)
        high = np.array([self.initial_angle_max, self.initial_velocity_max])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}


class CARLPendulumEnv(CARLEnv):
    def __init__(
        self,
        env: CustomPendulum = CustomPendulum(),
        contexts: Contexts = {},
        hide_context: bool = True,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Context] = DEFAULT_CONTEXT,
        max_episode_length: int = 200,  # from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        state_context_features: Optional[List[str]] = None,
        context_mask: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[
            Union[AbstractSelector, type[AbstractSelector]]
        ] = None,
        context_selector_kwargs: Optional[Dict] = None,
    ):
        """
        Max torque is not a context feature because it changes the action space.

        Parameters
        ----------
        env
        contexts
        instance_mode
        hide_context
        add_gaussian_noise_to_context
        gaussian_noise_std_percentage
        """
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
            context_mask=context_mask,
        )
        self.whitelist_gaussian_noise = list(
            DEFAULT_CONTEXT.keys()
        )  # allow to augment all values

    def _update_context(self) -> None:
        self.env: CustomPendulum
        self.env.max_speed = self.context["max_speed"]
        self.env.dt = self.context["dt"]
        self.env.l = self.context["l"]  # noqa: E741 ambiguous variable name
        self.env.m = self.context["m"]
        self.env.g = self.context["g"]
        self.env.initial_angle_max = self.context["initial_angle_max"]
        self.env.initial_velocity_max = self.context["initial_velocity_max"]

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.build_observation_space(-high, high, CONTEXT_BOUNDS)
