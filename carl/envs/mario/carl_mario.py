from __future__ import annotations

from typing import List

import numpy as np

from carl.context.context_space import (
    CategoricalContextFeature,
    ContextFeature,
    UniformFloatContextFeature,
    UniformIntegerContextFeature,
)
from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.envs.mario.pcg_smb_env import MarioEnv
from carl.envs.mario.pcg_smb_env.toadgan.toad_gan import generate_level
from carl.utils.types import Contexts

LEVEL_HEIGHT = 16


class CARLMarioEnv(CARLEnv):
    metadata = {
        "render_modes": ["rgb_array", "tiny_rgb_array"],
        "render_fps": 24,
    }

    def __init__(
        self,
        env: MarioEnv = None,
        contexts: Contexts | None = None,
        obs_context_features: (
            list[str] | None
        ) = None,  # list the context features which should be added to the state
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

    def _update_context(self) -> None:
        self.env: MarioEnv
        self.context = CARLMarioEnv.get_context_space().insert_defaults(self.context)
        if not self.levels:
            for context in self.contexts.values():
                level, _ = generate_level(
                    width=context["level_width"],
                    height=LEVEL_HEIGHT,
                    level_index=context["level_index"],
                    seed=context["noise_seed"],
                    filter_unplayable=True,
                )
                self.levels.append(level)
        self.env.mario_state = self.context["mario_state"]
        self.env.mario_inertia = self.context["mario_inertia"]
        self.env.levels = [self.levels[self.context_id]]

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "level_width": UniformIntegerContextFeature(
                "level_width", 16, 1000, default_value=100
            ),
            "level_index": CategoricalContextFeature(
                "level_index", choices=np.arange(0, 14), default_value=0
            ),
            "noise_seed": UniformIntegerContextFeature(
                "noise_seed", 0, 2**31 - 1, default_value=0
            ),
            "mario_state": CategoricalContextFeature(
                "mario_state", choices=[0, 1, 2], default_value=0
            ),
            "mario_inertia": UniformFloatContextFeature(
                "mario_inertia", lower=0.5, upper=1.5, default_value=0.89
            ),
        }
