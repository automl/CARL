# pylint: disable=missing-module-docstring  # isort: skip_file
from __future__ import annotations
from typing import Optional, List, Tuple, Any
import numpy as np
import gymnasium as gym
from itertools import chain, combinations
from carl.envs.carl_env import CARLEnv
from carl.envs.rna.parse_dot_brackets import parse_dot_brackets
from carl.envs.rna.rna_environment import (
    RnaDesignEnvironment,
    RnaDesignEnvironmentConfig,
)
from carl.utils.types import Contexts
from carl.context.context_space import (
    ContextFeature,
    UniformFloatContextFeature,
    CategoricalContextFeature,
)
from carl.context.selection import AbstractSelector

ACTION_SPACE = gym.spaces.Discrete(4)
OBSERVATION_SPACE = gym.spaces.Box(low=-np.inf * np.ones(11), high=np.inf * np.ones(11))


class CARLRnaDesignEnv(CARLEnv):
    def __init__(
        self,
        env: RnaDesignEnvironment | None = None,
        contexts: Contexts | None = None,
        obs_context_features: (
            list[str] | None
        ) = None,  # list the context features which should be added to the state
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        obs_low: Optional[int] = 11,
        obs_high: Optional[int] = 11,
        data_location: str = "carl/envs/rna/learna/data",
        **kwargs,
    ):
        """
        Parameters
        ----------
        env: gym.Env, optional
            Defaults to classic control environment mountain car from gym (MountainCarEnv).
        contexts: List[Dict], optional
            Different contexts / different environment parameter settings.
        instance_mode: str, optional
        """
        if env is None:
            context_space = self.get_context_features()
            env_config = RnaDesignEnvironmentConfig(
                mutation_threshold=context_space["mutation_threshold"].default_value,
                reward_exponent=context_space["reward_exponent"].default_value,
                state_radius=context_space["state_radius"].default_value,
            )
            dot_brackets = parse_dot_brackets(
                dataset=context_space["dataset"].default_value,  # type: ignore[arg-type]  # type: ignore[arg-type]
                data_dir=data_location,
                target_structure_ids=context_space[
                    "target_structure_ids"
                ].default_value,  # type: ignore[arg-type]
            )
            env = RnaDesignEnvironment(dot_brackets, env_config)

        env.action_space = ACTION_SPACE
        env.observation_space = OBSERVATION_SPACE
        env.reward_range = (-np.inf, np.inf)
        env.metadata = {}
        # The data_location in the RNA env refers to the place where the dataset is downloaded to, so it is not changed
        # with the context.
        env.data_location = data_location
        super().__init__(
            env=env,
            contexts=contexts,
            obs_context_features=obs_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )
        self.whitelist_gaussian_noise = list(self.get_context_features().keys())
        self.obs_low = obs_low
        self.obs_high = obs_high

    def step(self, action: np.ndarray) -> Tuple[List[int], float, Any, Any, Any]:
        # Step function has a different name in this env
        state, reward, terminated, truncated = self.env.execute(action)  # type: ignore[has-type]
        self.step_counter += 1
        return state, reward, terminated, truncated, {}

    def _update_context(self) -> None:
        dot_brackets = parse_dot_brackets(
            dataset=self.context["dataset"],
            data_dir=self.env.data_location,  # type: ignore[has-type]
            target_structure_ids=self.context["target_structure_ids"],
        )
        env_config = RnaDesignEnvironmentConfig(
            mutation_threshold=self.context["mutation_threshold"],
            reward_exponent=self.context["reward_exponent"],
            state_radius=self.context["state_radius"],
        )
        self.env = RnaDesignEnvironment(dot_brackets, env_config)
        # self.build_observation_space(
        #     env_lower_bounds=-np.inf * np.ones(self.obs_low),
        #     env_upper_bounds=np.inf * np.ones(self.obs_high),
        #     context_bounds=CONTEXT_BOUNDS,  # type: ignore[arg-type]
        # )

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        # TODO: these actually depend on the dataset, how to handle this?
        base_ids = list(range(1, 11))
        id_choices = list(
            chain(
                *map(lambda x: combinations(base_ids, x), range(0, len(base_ids) + 1))
            )
        ) + [False]
        return {
            "mutation_threshold": UniformFloatContextFeature(
                "mutation_threshold", lower=0.1, upper=np.inf, default_value=5
            ),
            "reward_exponent": UniformFloatContextFeature(
                "reward_exponent", lower=0.1, upper=np.inf, default_value=1
            ),
            "state_radius": UniformFloatContextFeature(
                "state_radius", lower=1, upper=np.inf, default_value=5
            ),
            "dataset": CategoricalContextFeature(
                "dataset",
                choices=["eterna", "rfam_learn", "rfam_taneda"],
                default_value="eterna",
            ),
            "target_structure_ids": CategoricalContextFeature(
                name="target_structure_ids", choices=id_choices, default_value=False
            ),
        }
