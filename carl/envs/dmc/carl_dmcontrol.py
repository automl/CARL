from typing import Any, Dict, List, Union, Optional

from carl.envs.carl_env import CARLEnv
from carl.envs.dmc.wrappers import MujocoToGymWrapper
from carl.envs.dmc.loader import load_dmc_env
from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector
from carl.context_encoders import ContextEncoder


class CARLDmcEnv(CARLEnv):
    def __init__(
        self,
        domain: str,
        task: str,
        contexts: Dict[Any, Dict[Any, Any]],
        context_mask: Optional[List[str]],
        hide_context: bool,
        add_gaussian_noise_to_context: bool,
        gaussian_noise_std_percentage: float,
        logger: Optional[TrialLogger],
        scale_context_features: str,
        default_context: Optional[Dict],
        max_episode_length: int,
        state_context_features: Optional[List[str]],
        dict_observation_space: bool,
        context_selector: Optional[Union[AbstractSelector, type(AbstractSelector)]],
        context_selector_kwargs: Optional[Dict],
        context_encoder: Optional[ContextEncoder] = None,
    ):
        # TODO can we have more than 1 env?
        # env = MujocoToGymWrapper(env)
        if not contexts:
            contexts = {0: default_context}
        self.domain = domain
        self.task = task
        if dict_observation_space:
            raise NotImplementedError
        else:
            env = load_dmc_env(
                domain_name=self.domain,
                task_name=self.task,
                context={},
                context_mask=[],
                environment_kwargs={"flat_observation": True}
            )
            env = MujocoToGymWrapper(env)
        super().__init__(
            env=env,
            contexts=contexts,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features=scale_context_features,
            default_context=default_context,
            max_episode_length=max_episode_length,
            state_context_features=state_context_features,
            dict_observation_space=dict_observation_space,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            context_mask=context_mask,
            context_encoder=context_encoder,
        )
        # TODO check gaussian noise on context features
        self.whitelist_gaussian_noise = list(
            default_context.keys()
        )  # allow to augment all values

    def _update_context(self) -> None:
        if self.dict_observation_space:
            raise NotImplementedError
        else:
            env = load_dmc_env(
                domain_name=self.domain,
                task_name=self.task,
                context=self.context,
                context_mask=self.context_mask,
                environment_kwargs={"flat_observation": True}
            )
            self.env = MujocoToGymWrapper(env)
