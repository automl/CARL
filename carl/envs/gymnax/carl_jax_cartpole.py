from typing import Dict, List, Optional, Union

import jax.numpy as jnp
from gymnax.environments.classic_control.cartpole import CartPole

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts

DEFAULT_CONTEXT = {
    "gravity": 9.8,
    "masscart": 1.0,
    "masspole": 0.1,
    "length": 0.5,
    "force_mag": 10.0,
    "tau": 0.02,
}

CONTEXT_BOUNDS = {
    "gravity": (5.0, 15.0, float),
    "masscart": (0.5, 2.0, float),
    "masspole": (0.05, 0.2, float),
    "length": (0.25, 1.0, float),
    "force_mag": (5.0, 15.0, float),
    "tau": (0.01, 0.05, float),
}


class CARLJaxCartPoleEnv(CARLEnv):
    def __init__(
        self,
        env: CartPole = CartPole(),
        contexts: Contexts = {},
        hide_context: bool = True,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Context] = DEFAULT_CONTEXT,
        max_episode_length: int = 500,  # from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
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
        )
        self.whitelist_gaussian_noise = list(
            DEFAULT_CONTEXT.keys()
        )  # allow to augment all values

    def _update_context(self) -> None:
        self.env: CartPole
        self.env.gravity = self.context["gravity"]
        self.env.masscart = self.context["masscart"]
        self.env.masspole = self.context["masspole"]
        self.env.length = self.context["length"]
        self.env.force_mag = self.context["force_mag"]
        self.env.tau = self.context["tau"]

        high = jnp.array(
            [
                self.env.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                self.env.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ],
            dtype=jnp.float32,
        )
        low = -high
        self.build_observation_space(low, high, CONTEXT_BOUNDS)
