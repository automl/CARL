from typing import Dict, List, Optional, Union

from gymnax.environments.classic_control.pendulum import EnvParams, Pendulum
import jax.numpy as jnp

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts

DEFAULT_CONTEXT = {
    "max_speed": 8.0,
    "max_torque": 2.0,
    "dt": 0.05,
    "g": 10.0,
    "m": 1.0,
    "l": 1.0,
    "max_steps_in_episode": 200,
}

CONTEXT_BOUNDS = {
    "max_speed": (-jnp.inf, jnp.inf, float),
    "max_torque": (-jnp.inf, jnp.inf, float),
    "dt": (0, jnp.inf, float),
    "g": (0, jnp.inf, float),
    "m": (1e-6, jnp.inf, float),
    "l": (1e-6, jnp.inf, float),
    "max_steps_in_episode": (1, jnp.inf, int),
}


class CARLJaxPendulumEnv(CARLEnv):
    def __init__(
        self,
        env: Pendulum = Pendulum(),
        contexts: Contexts = {},
        hide_context: bool = True,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Context] = DEFAULT_CONTEXT,
        max_episode_length: int = 200,  # from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
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
        self.env_params = EnvParams(**self.context)

        high = jnp.array([1.0, 1.0, self.max_speed], dtype=jnp.float32)
        self.build_observation_space(-high, high, CONTEXT_BOUNDS)
