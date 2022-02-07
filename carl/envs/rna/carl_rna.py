from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np

from carl.envs.carl_env import CARLEnv
from carl.envs.rna.carl_rna_definitions import (
    ACTION_SPACE,
    DEFAULT_CONTEXT,
    OBSERVATION_SPACE,
)
from carl.envs.rna.learna.src.data.parse_dot_brackets import parse_dot_brackets
from carl.envs.rna.learna.src.learna.environment import (
    RnaDesignEnvironment,
    RnaDesignEnvironmentConfig,
)
from carl.utils.trial_logger import TrialLogger


class RnaGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)

    def step(self, action):
        state, done, reward = self.env.execute(action)
        return state, reward, done, {}


class CARLRnaDesignEnv(CARLEnv):
    def __init__(
        self,
        env: Optional[gym.Env] = None,
        data_location: str = "carl/envs/rna/learna/data",
        contexts: Dict[Any, Dict[Any, Any]] = {},
        instance_mode: str = "rr",
        hide_context: bool = False,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Dict] = DEFAULT_CONTEXT,
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
                dataset=DEFAULT_CONTEXT["dataset"],
                data_dir=data_location,
                target_structure_ids=DEFAULT_CONTEXT["target_structure_ids"],
            )
            env = RnaDesignEnvironment(dot_brackets, env_config)
        env.action_space = ACTION_SPACE
        env.observation_space = OBSERVATION_SPACE
        env.reward_range = (-np.inf, np.inf)
        env.metadata = {}
        env = RnaGymWrapper(env)
        # The data_location in the RNA env refers to the place where the dataset is downloaded to, so it is not changed
        # with the context.
        self.data_location = data_location
        super().__init__(
            env=env,
            contexts=contexts,
            instance_mode=instance_mode,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features=scale_context_features,
            default_context=default_context,
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT)

   # def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
   #     # Step function has a different name in this env
   #     state, reward, done = self.env.execute(action)
   #     if not self.hide_context:
   #         state = np.concatenate((state, np.array(list(self.context.values()))))
   #     self.step_counter += 1
   #     return state, reward, done, {}

    def _update_context(self) -> None:
        dot_brackets = parse_dot_brackets(
            dataset=self.context["dataset"],
            data_dir=self.data_location,
            target_structure_ids=self.context["target_structure_ids"],
        )
        env_config = RnaDesignEnvironmentConfig(
            mutation_threshold=self.context["mutation_threshold"],
            reward_exponent=self.context["reward_exponent"],
            state_radius=self.context["state_radius"],
        )
        self.env = RnaGymWrapper(RnaDesignEnvironment(dot_brackets, env_config))
