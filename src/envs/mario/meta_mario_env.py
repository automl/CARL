from typing import Dict, Optional

import gym
import numpy as np
from src.envs.meta_env import MetaEnv
from src.trial_logger import TrialLogger

from .mario_env import MarioEnv

DEFAULT_CONTEXT = {
    "level_idx": 0,
    "width": 200 # TODO(frederik): lookup correct with of level 1-1
    # TODO(frederik): make height also part of context?
}

CONTEXT_BOUNDS = {
    "level_idx": (0, 14, int),
    "width": (16, 2000, int)
}


class MetaMarioEnv(MetaEnv):
    def __init__(
            self,
            env: gym.Env = MarioEnv(levels=[]), # TODO(frederik): set default levels
            contexts: Dict[int, Dict] = {},
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
        # TODO(frederik): generate levels according to context using TOAD-GAN
        # TODO(frederik): filter generated levels by reachability?
        # TODO(frederik): update levels in env

