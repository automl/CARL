import numpy as np
import math
import gym
import gym.envs.classic_control as gccenvs
from gym import spaces
from gym.utils import seeding
from typing import List, Dict
from src.meta_env import MetaEnv


DEFAULT_CONTEXT = {
    "min_position": -1.2,  # unit?
    "max_position": 0.6,  # unit?
    "max_speed": 0.07,  # unit?
    "goal_position": 0.5,  # unit?
    "goal_velocity": 0,  # unit?
    "force": 0.001,  # unit?
    "gravity": 0.0025,  # unit?
    "min_position_start": -0.6,
    "max_position_start": -0.4,
    "min_velocity_start": 0.,
    "max_velocity_start": 0.,
}

DEFAULT_CONTEXT_BOUNDS = {
    "gravity": (0, np.inf),
    # TODO add bounds for each env
}


class CustomMountainCarEnv(gccenvs.mountain_car.MountainCarEnv):
    def __init__(self, goal_velocity: float = 0.):
        super(CustomMountainCarEnv, self).__init__(goal_velocity=goal_velocity)
        self.min_position_start = -0.6
        self.max_position_start = -0.4
        self.min_velocity_start = 0.
        self.max_velocity_start = 0.
        
    def reset_state(self):
        return np.array([
            self.np_random.uniform(low=self.min_position_start, high=self.max_position_start),  # sample start position
            self.np_random.uniform(low=self.min_velocity_start, high=self.max_velocity_start)   # sample start velocity
        ])

    def reset(self):
        self.state = self.reset_state().squeeze()
        return self.state

    def step(self, action):
        state, reward, done, info = super().step(action)
        return state.squeeze(), reward, done, info  # TODO something weird is happening such that the state gets shape (2,1) instead of (2,)


class MetaMountainCarEnv(MetaEnv):
    def __init__(
            self,
            env: gym.Env = CustomMountainCarEnv(),
            contexts: Dict[str, Dict] = {},
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01
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
            instance_mode=instance_mode,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT.keys())  # allow to augment all values
        self._update_context()

    def _update_context(self):
        self.env.min_position = self.context["min_position"]
        self.env.max_position = self.context["max_position"]
        self.env.max_speed = self.context["max_speed"]
        self.env.goal_position = self.context["goal_position"]
        self.env.goal_velocity = self.context["goal_velocity"]
        self.env.min_position_start = self.context["min_position_start"]
        self.env.max_position_start = self.context["max_position_start"]
        self.env.min_velocity_start = self.context["min_velocity_start"]
        self.env.max_velocity_start = self.context["max_velocity_start"]
        self.env.force = self.context["force"]
        self.env.gravity = self.context["gravity"]

        self.low = np.array(
            [self.env.min_position, -self.env.max_speed], dtype=np.float32
        ).squeeze()
        self.high = np.array(
            [self.env.max_position, self.env.max_speed], dtype=np.float32
        ).squeeze()

        self.env.observation_space = spaces.Box(  # TODO check this for every env (".env.")
            self.low, self.high, dtype=np.float32,
        ) # TODO add context space if applicable (in every child env)
        self.observation_space = self.env.observation_space  # make sure it is the same object

        # TODO log context to debug and put into info object
