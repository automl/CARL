# pylint: disable=missing-module-docstring  # isort: skip_file
from typing import Optional, Dict, Union, List, Tuple, Any
import numpy as np
import gymnasium as gym

from carl.envs.carl_env import CARLEnv
from carl.envs.rna.parse_dot_brackets import parse_dot_brackets
from carl.envs.rna.rna_environment import (
    RnaDesignEnvironment,
    RnaDesignEnvironmentConfig,
)
from carl.utils.trial_logger import TrialLogger
from carl.envs.rna.carl_rna_definitions import (
    DEFAULT_CONTEXT,
    ACTION_SPACE,
    OBSERVATION_SPACE,
    CONTEXT_BOUNDS,
)
from carl.utils.types import Context, Contexts
from carl.context.selection import AbstractSelector


class CARLRnaDesignEnv(CARLEnv):
    def __init__(
        self,
        env: gym.Env = None,
        data_location: str = "carl/envs/rna/learna/data",
        contexts: Contexts = {},
        hide_context: bool = False,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Context] = DEFAULT_CONTEXT,
        max_episode_length: int = 500,
        state_context_features: Optional[List[str]] = None,
        context_mask: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[
            Union[AbstractSelector, type[AbstractSelector]]
        ] = None,
        context_selector_kwargs: Optional[Dict] = None,
        obs_low: Optional[int] = 11,
        obs_high: Optional[int] = 11,
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
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        if env is None:
            env_config = RnaDesignEnvironmentConfig(
                mutation_threshold=DEFAULT_CONTEXT["mutation_threshold"],
                reward_exponent=DEFAULT_CONTEXT["reward_exponent"],
                state_radius=DEFAULT_CONTEXT["state_radius"],
            )
            dot_brackets = parse_dot_brackets(
                dataset=DEFAULT_CONTEXT["dataset"],  # type: ignore[arg-type]
                data_dir=data_location,
                target_structure_ids=DEFAULT_CONTEXT["target_structure_ids"],  # type: ignore[arg-type]
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
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT)
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
        self.build_observation_space(
            env_lower_bounds=-np.inf * np.ones(self.obs_low),
            env_upper_bounds=np.inf * np.ones(self.obs_high),
            context_bounds=CONTEXT_BOUNDS,  # type: ignore[arg-type]
        )
