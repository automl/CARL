from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import gym.envs.classic_control as gccenvs
import numpy as np

from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector

DEFAULT_CONTEXT = {
    "min_position": -1.2,  # unit?
    "max_position": 0.6,  # unit?
    "max_speed": 0.07,  # unit?
    "goal_position": 0.5,  # unit?
    "goal_velocity": 0,  # unit?
    "force": 0.001,  # unit?
    "gravity": 0.0025,  # unit?
    "start_position": -0.5,
    "start_position_std": 0.1,
    "start_velocity": 0.0,
    "start_velocity_std": 0.0,
    # "min_position_start": -0.6,
    # "max_position_start": -0.4,
    # "min_velocity_start": 0.,
    # "max_velocity_start": 0.,
}

CONTEXT_BOUNDS = {
    "min_position": (-np.inf, np.inf, float),
    "max_position": (-np.inf, np.inf, float),
    "max_speed": (0, np.inf, float),
    "goal_position": (-np.inf, np.inf, float),
    "goal_velocity": (-np.inf, np.inf, float),
    "force": (-np.inf, np.inf, float),
    "gravity": (0, np.inf, float),
    "start_position": (-1.5, 0.5, float),  # start position inbetween hilltops
    "start_position_std": (0.1, np.inf, float),
    "start_velocity": (-np.inf, np.inf, float),
    "start_velocity_std": (0.1, np.inf, float)
    # "min_position_start": (-np.inf, np.inf, float),  # TODO need to check these!!!
    # "max_position_start": (-np.inf, np.inf, float),
    # "min_velocity_start": (-np.inf, np.inf, float),
    # "max_velocity_start": (-np.inf, np.inf, float),
}


class CustomMountainCarEnv(gccenvs.mountain_car.MountainCarEnv):
    def __init__(self, goal_velocity: float = 0.0):
        super(CustomMountainCarEnv, self).__init__(goal_velocity=goal_velocity)
        # self.min_position_start = -0.6
        # self.max_position_start = -0.4
        # self.min_velocity_start = 0.
        # self.max_velocity_start = 0.
        self.position_start = -0.5
        self.position_start_std = 0.1
        self.velocity_start = 0.0
        self.velocity_start_std = 0.0

    def reset_state(self) -> np.ndarray:
        return np.array(
            [
                self.np_random.normal(self.position_start, self.position_start_std),
                self.np_random.normal(self.velocity_start, self.velocity_start_std)
                # self.np_random.uniform(low=self.min_position_start, high=self.max_position_start),
                # self.np_random.uniform(low=self.min_velocity_start, high=self.max_velocity_start)
            ]
        )

    def reset(self) -> np.ndarray:
        self.state = self.reset_state().squeeze()
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        state, reward, done, info = super().step(action)
        return (
            state.squeeze(),
            reward,
            done,
            info,
        )  # TODO something weird is happening such that the state gets shape (2,1) instead of (2,)


class CARLMountainCarEnv(CARLEnv):
    def __init__(
        self,
        env: gym.Env = CustomMountainCarEnv(),
        contexts: Dict[Any, Dict[Any, Any]] = {},
        hide_context: bool = False,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Dict] = DEFAULT_CONTEXT,
        max_episode_length: int = 200,  # from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        state_context_features: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[Union[AbstractSelector, type(AbstractSelector)]] = None,
        context_selector_kwargs: Optional[Dict] = None,
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
        self.whitelist_gaussian_noise = list(
            DEFAULT_CONTEXT.keys()
        )  # allow to augment all values

    def _update_context(self) -> None:
        self.env.min_position = self.context["min_position"]
        self.env.max_position = self.context["max_position"]
        self.env.max_speed = self.context["max_speed"]
        self.env.goal_position = self.context["goal_position"]
        self.env.goal_velocity = self.context["goal_velocity"]
        self.env.position_start = self.context["start_position"]
        self.env.position_start_std = self.context["start_position_std"]
        self.env.velocity_start = self.context["start_velocity"]
        self.env.velocity_start_std = self.context["start_velocity_std"]
        # self.env.min_position_start = self.context["min_position_start"]
        # self.env.max_position_start = self.context["max_position_start"]
        # self.env.min_velocity_start = self.context["min_velocity_start"]
        # self.env.max_velocity_start = self.context["max_velocity_start"]
        self.env.force = self.context["force"]
        self.env.gravity = self.context["gravity"]

        self.low = np.array(
            [self.env.min_position, -self.env.max_speed], dtype=np.float32
        ).squeeze()
        self.high = np.array(
            [self.env.max_position, self.env.max_speed], dtype=np.float32
        ).squeeze()

        self.build_observation_space(self.low, self.high, CONTEXT_BOUNDS)
