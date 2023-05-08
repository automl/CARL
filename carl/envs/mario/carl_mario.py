from typing import Dict, List, Optional, Union

import gymnasium as gym

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.envs.mario.carl_mario_definitions import (
    DEFAULT_CONTEXT,
    INITIAL_HEIGHT,
    INITIAL_WIDTH,
)
from carl.envs.mario.mario_env import MarioEnv
from carl.envs.mario.toad_gan import generate_level
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts


class CARLMarioEnv(CARLEnv):
    def __init__(
        self,
        env: gym.Env = MarioEnv(levels=[]),
        contexts: Contexts = {},
        hide_context: bool = True,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.05,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Context] = DEFAULT_CONTEXT,
        state_context_features: Optional[List[str]] = None,
        context_mask: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[
            Union[AbstractSelector, type[AbstractSelector]]
        ] = None,
        context_selector_kwargs: Optional[Dict] = None,
    ):
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        super().__init__(
            env=env,
            contexts=contexts,
            hide_context=True,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features="no",
            default_context=default_context,
            dict_observation_space=dict_observation_space,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            context_mask=context_mask,
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
