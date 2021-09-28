from typing import Dict, List, Optional

import gym
import numpy as np
from src.envs.mario.mario_env import MarioEnv
from src.envs.mario.toad_gan import generate_initial_noise, generate_level
from src.envs.carl_env import CARLEnv
from src.training.trial_logger import TrialLogger

INITIAL_WIDTH = 100
INITIAL_LEVEL_INDEX = 0
INITIAL_HEIGHT = 16
DEFAULT_CONTEXT = {
    "level_index": INITIAL_LEVEL_INDEX,
    "noise": generate_initial_noise(INITIAL_WIDTH, INITIAL_HEIGHT, INITIAL_LEVEL_INDEX),
    "mario_state": 0,
    "mario_inertia": 0.89
}

CONTEXT_BOUNDS = {
    "level_index": (None, None, "categorical", np.arange(0, 14)),
    "noise": (-1.0, 1.0, float),
    "mario_state": (None, None, "categorical", [0, 1, 2]),
    "mario_inertia": (0.5, 1.5, float)
}
CATEGORICAL_CONTEXT_FEATURES = ["level_index", "mario_state"]


class CARLMarioEnv(CARLEnv):
    def __init__(
        self,
        env: gym.Env = MarioEnv(levels=[]),
        contexts: Dict[int, Dict] = {},
        instance_mode: str = "rr",
        hide_context: bool = False,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.05,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Dict] = DEFAULT_CONTEXT,
        state_context_features: Optional[List[str]] = None,
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
            scale_context_features="no",
            default_context=default_context,
        )
        self.levels = []
        for context in contexts.values():
            level = generate_level(
                width=INITIAL_WIDTH,
                height=INITIAL_HEIGHT,
                level_index=context["level_index"],
                initial_noise=context["noise"],
                filter_unplayable=True,
            )
            self.levels.append(level)
        self._update_context()

    def _update_context(self):
        self.env.mario_state = self.context["mario_state"]
        self.env.levels = [self.levels[self.context_index]]

    def _log_context(self):
        if self.logger:
            loggable_context = {k: v for k, v in self.context.items() if k != "noise"}
            self.logger.write_context(
                self.episode_counter, self.total_timestep_counter, loggable_context
            )

