import gym
import numpy as np
from typing import Optional, Dict
from gym.envs.classic_control import CartPoleEnv
from src.envs.meta_env import MetaEnv
from src.trial_logger import TrialLogger

DEFAULT_CONTEXT = {
    "gravity": 9.8,
    "masscart": 1., # Should be seen as 100% and scaled accordingly
    "masspole":  0.1, # Should be seen as 100% and scaled accordingly
    "pole_length": 0.5, # Should be seen as 100% and scaled accordingly
    "force_magnifier": 10.,
    "update_interval": 0.02, # Seconds between updates
}

CONTEXT_BOUNDS = {
    "gravity": (0.1, np.inf, float), # Positive gravity
    "masscart": (0.1, 10, float), # Cart mass can be varied by a factor of 10
    "masspole":  (0.01, 1, float), # Pole mass can be varied by a factor of 10
    "pole_length": (0.05, 5, float), # Pole length can be varied by a factor of 10
    "force_magnifier": (1, 100, int), # Force magnifier can be varied by a factor of 10
    "update_interval": (0.002, 0.2, float), # Update interval can be varied by a factor of 10
}


class MetaCartPoleEnv(MetaEnv):
    def __init__(
            self,
            env: gym.Env = CartPoleEnv(),
            contexts: Dict[str, Dict] = {},
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None,
    ):
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
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT.keys())  # allow to augment all values
        self._update_context()

    def _update_context(self):
        self.env.gravity = self.context["gravity"]
        self.env.masscart = self.context["masscart"]
        self.env.masspole = self.context["masspole"]
        self.env.length = self.context["pole_length"]
        self.env.force_mag = self.context["force_magnifier"]
        self.env.tau = self.context["update_interval"]

        high = np.array([self.env.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.env.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        low = -high
        self.build_observation_space(low, high, CONTEXT_BOUNDS)

