from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from brax.envs import create
from carl.envs.braxenvs.brax_wrappers import GymWrapper, VectorGymWrapper

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts


class CARLBraxEnv(CARLEnv):
    env_name: str
    DEFAULT_CONTEXT: Context

    def __init__(
        self,
        env=None,
        n_envs: int = 1,
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
        max_episode_length = 1000,
    ):
        if env is None:
            batch_size = None if n_envs == 1 else n_envs  # TODO check if batched env works with concat state
            env = create(self.env_name, batch_size=batch_size)
            
        self.n_envs=n_envs
        if n_envs == 1:
            env = GymWrapper(env)
        else:
            env = VectorGymWrapper(env, n_envs)

        if not contexts:
            contexts = {0: self.DEFAULT_CONTEXT}
        if not default_context:
            default_context = self.DEFAULT_CONTEXT
        super().__init__(
            env=env,
            n_envs=n_envs,
            contexts=contexts,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features=scale_context_features,
            default_context=default_context,
            state_context_features=state_context_features,
            dict_observation_space=dict_observation_space,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            context_mask=context_mask,
            max_episode_length=max_episode_length,
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


