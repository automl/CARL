from typing import Any, Dict, List, Optional, Union

import gym

from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv, DEFAULT_CONTEXT, load_dmc_env


class CARLDmcCartpoleEnv(CARLDmcEnv):
    def __init__(
        self,
        domain: str = "cartpole",
        task: str = "swingup",
        contexts: Dict[Any, Dict[Any, Any]] = {},
        hide_context: bool = False,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Dict] = DEFAULT_CONTEXT,
        max_episode_length: int = 500,  # from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        state_context_features: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[Union[AbstractSelector, type(AbstractSelector)]] = None,
        context_selector_kwargs: Optional[Dict] = None,
    ):
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        if dict_observation_space:
            raise NotImplementedError
        else:
            env = load_dmc_env(domain_name=domain, task_name=task, environment_kwargs={"flat_observation": True})
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
        )
    
    def _update_context(self) -> None:
        pass
        # self.env.gravity = self.context["gravity"]
        # print(self.env.env._physics)
        # print(self.env.env.__dict__)

        # "gravity",
        # "wind",
        # "magnetic",
        # "density",
        # "viscosity",
        # high = np.array(
        #     [
        #         self.env.x_threshold * 2,
        #         np.finfo(np.float32).max,
        #         self.env.theta_threshold_radians * 2,
        #         np.finfo(np.float32).max,
        #     ],
        #     dtype=np.float32,
        # )
        # low = -high
        # print(low)
        # print(high)
        # self.build_observation_space(low, high, CONTEXT_BOUNDS)
