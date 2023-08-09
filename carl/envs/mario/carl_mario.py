from __future__ import annotations

from typing import List

import numpy as np

from carl.context.context_space import (
    CategoricalContextFeature,
    ContextFeature,
    UniformFloatContextFeature,
)
from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.envs.mario.mario_env import MarioEnv
from carl.envs.mario.toad_gan import generate_level
from carl.utils.types import Contexts

try:
    from carl.envs.mario.toad_gan import generate_initial_noise
except FileNotFoundError:
    from torch import Tensor

    def generate_initial_noise(width: int, height: int, level_index: int) -> Tensor:
        return Tensor()


INITIAL_HEIGHT = 16
INITIAL_WIDTH = 100


class CARLMarioEnv(CARLEnv):
    def __init__(
        self,
        env: MarioEnv = None,
        contexts: Contexts | None = None,
        obs_context_features: list[str]
        | None = None,  # list the context features which should be added to the state
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        **kwargs,
    ):
        if env is None:
            env = MarioEnv(levels=[])
        super().__init__(
            env=env,
            contexts=contexts,
            obs_context_features=obs_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )
        self.levels: List[str] = []
        self._update_context()

    def _update_context(self) -> None:
        self.env: MarioEnv
        if not self.levels:
            for context in self.contexts.values():
                level = generate_level(
                    width=INITIAL_WIDTH,
                    height=INITIAL_HEIGHT,
                    level_index=context["level_index"],
                    initial_noise=context["noise"],
                    filter_unplayable=True,
                )
                self.levels.append(level)
        self.env.mario_state = self.context["mario_state"]
        self.env.mario_inertia = self.context["mario_inertia"]
        self.env.levels = [self.levels[self.context_index]]

    def _log_context(self) -> None:
        if self.logger:
            loggable_context = {k: v for k, v in self.context.items() if k != "noise"}
            self.logger.write_context(
                self.episode_counter, self.total_timestep_counter, loggable_context
            )

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "level_index": CategoricalContextFeature(
                "level_index", choices=np.arange(0, 14), default_value=0
            ),
            "noise": UniformFloatContextFeature(
                "noise",
                lower=-1.0,
                upper=1.0,
                default_value=generate_initial_noise(INITIAL_WIDTH, INITIAL_HEIGHT, 0),
            ),
            "mario_state": CategoricalContextFeature(
                "mario_state", choices=[0, 1, 2], default_value=0
            ),
            "mario_inertia": UniformFloatContextFeature(
                "mario_inertia", lower=0.5, upper=1.5, default_value=0.89
            ),
        }
