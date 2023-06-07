from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import gymnasium

from CARL.carl.envs.gymnax.utils import make_gymnax_env
from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts


class CARLGymnaxEnv(CARLEnv):
    env_name: str
    DEFAULT_CONTEXT: Context
    max_episode_steps: int

    def __init__(
        self,
        env: gymnasium.Env | None = None,
        contexts: Contexts = {},
        hide_context: bool = True,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Context] = None,
        state_context_features: Optional[List[str]] = None,
        context_mask: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[
            Union[AbstractSelector, type[AbstractSelector]]
        ] = None,
        context_selector_kwargs: Optional[Dict] = None,
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
        if env is None:
            env = make_gymnax_env(env_name=self.env_name)

        if not contexts:
            contexts = {0: self.DEFAULT_CONTEXT}

        if not default_context:
            default_context = self.DEFAULT_CONTEXT

        super().__init__(
            env=env,
            contexts=contexts,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features=scale_context_features,
            default_context=default_context,
            max_episode_length=self.max_episode_steps,
            state_context_features=state_context_features,
            dict_observation_space=dict_observation_space,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            context_mask=context_mask,
        )
        self.whitelist_gaussian_noise = list(
            self.DEFAULT_CONTEXT.keys()
        )  # allow to augment all values

    def _update_context(self) -> None:
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        if name in ["sys", "__getstate__"]:
            return getattr(self.env._environment, name)
        else:
            return getattr(self, name)
