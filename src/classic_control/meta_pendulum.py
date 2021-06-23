import numpy as np
from typing import Dict
import gym
from gym import spaces
import gym.envs.classic_control as gccenvs

from src.meta_env import MetaEnv


DEFAULT_CONTEXT = {
    "max_speed": 8.,
    "max_torque": 2.,
    "dt":  0.05,
    "g": 10.0,
    "m": 1.,
    "l": 1.,
}


class MetaPendulumEnv(MetaEnv):
    def __init__(
            self,
            env: gym.Env = gccenvs.pendulum.PendulumEnv(),
            contexts: Dict[int, Dict] = {},  # ??? what should be the type of the dict keys?
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01
    ):
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
        self._update_context()    # TODO move this to MetaEnv as this is the same for each child meta env

    def _update_context(self):
        self.env.max_speed = self.context["max_speed"]
        self.env.max_torque = self.context["max_torque"]  # TODO: allow to change action space?
        self.env.dt = self.context["dt"]
        self.env.l = self.context["l"]
        self.env.m = self.context["m"]
        self.env.g = self.context["g"]

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
