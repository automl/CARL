import gym
import numpy as np
from typing import Optional, Dict
from .mario_env import MarioEnv
from src.envs.meta_env import MetaEnv
from src.trial_logger import TrialLogger

DEFAULT_CONTEXT = {
    
}

CONTEXT_BOUNDS = {
   
}


class MetaMarioEnv(MetaEnv):
    def __init__(
            self,
            env: gym.Env = MarioEnv(levels=[]), # TODO(frederik): set default levels
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
            logger=logger
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT.keys())  # allow to augment all values
        self._update_context()

    def _update_context(self):
        # TODO(frederik): update levels in env according to context
        self.build_observation_space(low, high, CONTEXT_BOUNDS)

