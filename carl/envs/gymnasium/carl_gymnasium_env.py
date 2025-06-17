from __future__ import annotations

import gymnasium
import pygame
from gymnasium.core import Env

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.types import Contexts

try:
    pygame.display.init()
except:  # noqa:E722
    import os  # pragma: no cover

    os.environ["SDL_VIDEODRIVER"] = "dummy"  # pragma: no cover


class CARLGymnasiumEnv(CARLEnv):
    env_name: str
    render_mode: str = "rgb_array"

    def __init__(
        self,
        env: Env | None = None,
        contexts: Contexts | None = None,
        obs_context_features: (
            list[str] | None
        ) = None,  # list the context features which should be added to the state
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        **kwargs,
    ) -> None:
        """
        CARL Gymnasium Environment.

        Parameters
        ----------

        env : Env | None
            Gymnasium environment, the default is None.
            If None, instantiate the env with gymnasium's make function and
            `self.env_name` which is defined in each child class.
        contexts : Contexts | None, optional
            Context set, by default None. If it is None, we build the
            context set with the default context.
        obs_context_features : list[str] | None, optional
            Context features which should be included in the observation, by default None.
            If they are None, add all context features.
        context_selector: AbstractSelector | type[AbstractSelector] | None, optional
            The context selector (class), after each reset selects a new context to use.
             If None, use a round robin selector.
        context_selector_kwargs : dict, optional
            Optional keyword arguments for the context selector, by default None.
            Only used when `context_selector` is not None.

        Attributes
        ----------
        env_name: str
            The registered gymnasium environment name.
        """
        if env is None:
            env = gymnasium.make(id=self.env_name, render_mode=self.render_mode)
        super().__init__(
            env=env,
            contexts=contexts,
            obs_context_features=obs_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )

    def _update_context(self) -> None:
        for k, v in self.context.items():
            setattr(self.env.unwrapped, k, v)
