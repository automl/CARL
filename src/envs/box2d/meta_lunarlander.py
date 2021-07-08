import numpy as np
import math
from typing import Dict, Optional

import gym
import Box2D
from gym.envs.box2d import lunar_lander
from gym.envs.box2d import lunar_lander as ll
from gym.envs.box2d.lunar_lander import heuristic
from gym import spaces
from gym.utils import seeding, EzPickle

from src.envs.meta_env import MetaEnv
from src.trial_logger import TrialLogger

# TODO debug/test this environment by looking at rendering!

DEFAULT_CONTEXT = {
    "FPS": 50,
    "SCALE": 30.0,   # affects how fast-paced the game is, forces should be adjusted as well

    # Engine powers
    "MAIN_ENGINE_POWER": 13.0,
    "SIDE_ENGINE_POWER": 0.6,

    # random force on lunar lander body on reset
    "INITIAL_RANDOM": 1000.0,   # Set 1500 to make game harder

    "GRAVITY_X": 0,
    "GRAVITY_Y": -10,

    # lunar lander body specification
    "LEG_AWAY": 20,
    "LEG_DOWN": 18,
    "LEG_W": 2,
    "LEG_H": 8,
    "LEG_SPRING_TORQUE": 40,
    "SIDE_ENGINE_HEIGHT": 14.0,
    "SIDE_ENGINE_AWAY": 12.0,

    # Size of world
    "VIEWPORT_W": 600,
    "VIEWPORT_H": 400,
}

CONTEXT_BOUNDS = {
    "FPS": (1, 500),
    "SCALE": (1, 100),   # affects how fast-paced the game is, forces should be adjusted as well
    "MAIN_ENGINE_POWER": (0, 50),
    "SIDE_ENGINE_POWER": (0, 50),

    # random force on lunar lander body on reset
    "INITIAL_RANDOM": (0, 2000),   # Set 1500 to make game harder

    "GRAVITY_X": (-20, 20),  # unit: m/s²
    "GRAVITY_Y": (-20, 0),   # the y-component of gravity is not allowed to be bigger than 0 because otherwise the
                             # lunarlander leaves the frame by going up

    # lunar lander body specification
    "LEG_AWAY": (0, 50),
    "LEG_DOWN": (0, 50),
    "LEG_W": (1, 10),
    "LEG_H": (1, 20),
    "LEG_SPRING_TORQUE": (0, 100),
    "SIDE_ENGINE_HEIGHT": (1, 20),
    "SIDE_ENGINE_AWAY": (1, 20),

    # Size of world
    "VIEWPORT_W": (400, 1000),
    "VIEWPORT_H": (200, 800),
}


class CustomLunarLanderEnv(lunar_lander.LunarLander):
    def __init__(self, gravity: (float, float) = (0, -10)):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World(gravity=gravity)
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.reset()


class MetaLunarLanderEnv(MetaEnv):
    def __init__(
            self,
            env: CustomLunarLanderEnv = CustomLunarLanderEnv(),
            contexts: Dict[str, Dict] = {},
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None,
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
            logger=logger
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT.keys())  # allow to augment all values
        self.gravity_x = None
        self.gravity_y = None
        self._update_context()

    def _update_context(self):
        lunar_lander.FPS = self.context["FPS"]
        lunar_lander.SCALE = self.context["SCALE"]
        lunar_lander.MAIN_ENGINE_POWER = self.context["MAIN_ENGINE_POWER"]
        lunar_lander.SIDE_ENGINE_POWER = self.context["SIDE_ENGINE_POWER"]

        lunar_lander.INITIAL_RANDOM = self.context["INITIAL_RANDOM"]

        lunar_lander.LEG_AWAY = self.context["LEG_AWAY"]
        lunar_lander.LEG_DOWN = self.context["LEG_DOWN"]
        lunar_lander.LEG_W = self.context["LEG_W"]
        lunar_lander.LEG_H = self.context["LEG_H"]
        lunar_lander.LEG_SPRING_TORQUE = self.context["LEG_SPRING_TORQUE"]
        lunar_lander.SIDE_ENGINE_HEIGHT = self.context["SIDE_ENGINE_HEIGHT"]
        lunar_lander.SIDE_ENGINE_AWAY = self.context["SIDE_ENGINE_AWAY"]

        lunar_lander.VIEWPORT_W = self.context["VIEWPORT_W"]
        lunar_lander.VIEWPORT_H = self.context["VIEWPORT_H"]

        self.gravity_x = self.context["GRAVITY_X"]
        self.gravity_y = self.context["GRAVITY_Y"]

        gravity = (self.gravity_x, self.gravity_y)
        # self.env.world = Box2D.b2World(gravity=gravity)

        # self.env.__init__(gravity=(self.gravity_x, self.gravity_y))

        print(lunar_lander.VIEWPORT_H)


def demo_heuristic_lander(env, seed=None, render=False):
    """
    Copied from LunarLander
    """
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if not still_open:
                break

        if done:# or steps % 20 == 0:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done:
            break
    return total_reward


if __name__ == '__main__':
    env = MetaLunarLanderEnv(hide_context=False)
    env.render()  # initialize viewer. otherwise weird bug.
    # env = ll.LunarLander()
    for i in range(5):
        demo_heuristic_lander(env, seed=0, render=True)
    env.close()
