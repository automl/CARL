import numpy as np
from typing import Dict, Optional, List
import gym
import gym.envs.classic_control as gccenvs

from src.envs.carl_env import CARLEnv
from src.training.trial_logger import TrialLogger


DEFAULT_CONTEXT = {
    "max_speed": 8.,
    "dt":  0.05,
    "g": 10.0,
    "m": 1.,
    "l": 1.,
}

CONTEXT_BOUNDS = {
    "max_speed": (-np.inf, np.inf, float),
    "dt": (0, np.inf, float),
    "g": (0, np.inf, float),
    "m": (1e-6, np.inf, float),
    "l": (1e-6, np.inf, float),
}


class CARLPendulumEnv(CARLEnv):
    def __init__(
            self,
            env: gym.Env = gccenvs.pendulum.PendulumEnv(),
            contexts: Dict[str, Dict] = {},
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = False,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None,
            scale_context_features: str = "no",
            default_context: Optional[Dict] = DEFAULT_CONTEXT,
            max_episode_length: int = 200,  # from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
            state_context_features: Optional[List[str]] = None,
            dict_observation_space: bool = False,
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
            instance_mode=instance_mode,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features=scale_context_features,
            default_context=default_context,
            max_episode_length=max_episode_length,
            state_context_features=state_context_features,
            dict_observation_space=dict_observation_space
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT.keys())  # allow to augment all values

    def _update_context(self):
        self.env.max_speed = self.context["max_speed"]
        self.env.dt = self.context["dt"]
        self.env.l = self.context["l"]
        self.env.m = self.context["m"]
        self.env.g = self.context["g"]

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.build_observation_space(-high, high, CONTEXT_BOUNDS)
