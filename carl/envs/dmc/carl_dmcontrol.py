from __future__ import annotations

from gymnasium import spaces

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.envs.dmc.loader import load_dmc_env
from carl.envs.dmc.wrappers import MujocoToGymWrapper
from carl.utils.types import Contexts


class CARLDmcEnv(CARLEnv):
    """
    General class for the dm-control environments.

    Meta-class to change the context for the environments.

    Parameters
    ----------
    domain : str
        Dm-control domain that should be loaded.
    task : str
        Task within the specified domain.

    For descriptions of the other parameters see the parent class CARLEnv.

    Raises
    ------
    NotImplementedError
        Dict observation spaces are not implemented for dm-control yet.
    """

    def __init__(
        self,
        contexts: Contexts | None = None,
        obs_context_features: list[str] | None = None,
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        **kwargs,
    ):
        # TODO can we have more than 1 env?
        env = load_dmc_env(
            domain_name=self.domain,
            task_name=self.task,
            context={},
            environment_kwargs={"flat_observation": True},
        )
        env = MujocoToGymWrapper(env)
        env.observation_space = spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            dtype=env.observation_space.dtype,
        )

        super().__init__(
            env=env,
            contexts=contexts,
            obs_context_features=obs_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )
        # TODO check gaussian noise on context features
        self.whitelist_gaussian_noise = list(
            self.get_context_features().keys()  # type: ignore
        )  # allow to augment all values

    def _update_context(self) -> None:
        env = load_dmc_env(
            domain_name=self.domain,
            task_name=self.task,
            context=self.context,
            environment_kwargs={"flat_observation": True},
        )
        self.env = MujocoToGymWrapper(env)

    def render(self):
        return self.env.render(mode="rgb_array")
