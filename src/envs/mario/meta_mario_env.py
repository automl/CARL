from typing import Dict, Optional

import gym
import numpy as np
from src.envs.mario.mario_env import MarioEnv
from src.envs.mario.toad_gan import generate_level
from src.envs.meta_env import MetaEnv
from src.trial_logger import TrialLogger

DEFAULT_CONTEXT = {"level_index": 0, "width": 200, "height": 16}

CONTEXT_BOUNDS = {
    "level_index": (0, 14, int),
    "width": (16, np.inf, int),
    "height": (8, 32, int),
}


class MetaMarioEnv(MetaEnv):
    def __init__(
        self,
        env: gym.Env = MarioEnv(levels=[]),
        contexts: Dict[int, Dict] = {},
        instance_mode: str = "rr",
        hide_context: bool = False,
        add_gaussian_noise_to_context: bool = True,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Dict] = DEFAULT_CONTEXT,
    ):
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        super().__init__(
            env=env,
            contexts=contexts,
            instance_mode=instance_mode,
            hide_context=True,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features=scale_context_features,
            default_context=default_context,
        )
        self.whitelist_gaussian_noise = list(
            DEFAULT_CONTEXT.keys()
        )  # allow to augment all values
        self._update_context()

    def _update_context(self):
        level = generate_level(
            int(self.context["width"]),
            int(self.context["height"]),
            int(self.context["level_index"]),
        )
        self.env.levels = [level]
