from typing import Dict, Optional

import gym
from gym.envs.box2d import bipedal_walker
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

from src.envs.meta_env import MetaEnv
from src.trial_logger import TrialLogger

# TODO debug/test this environment by looking at rendering!

# TODO what about scale?

DEFAULT_CONTEXT = {
    "FPS": 50,
    "SCALE": 30.0,   # affects how fast-paced the game is, forces should be adjusted as well

    # surroundings
    "FRICTION": 2.5,
    "TERRAIN_STEP": 14/30.0,
    "TERRAIN_LENGTH": 200,  # in steps
    "TERRAIN_HEIGHT"

    # walker
    "MOTORS_TORQUE": 80,
    "SPEED_HIP": 4,
    "SPEED_KNEE": 6,
    "LIDAR_RANGE": 160/30.0,
    "LEG_DOWN": -8/30.0,
    "LEG_W": 8/30.0,
    "LEC_H": 34/30.0,

    # absolute value of random force applied to walker at start of episode
    "INITIAL_RANDOM": 5,


    # Size of world
    "VIEWPORT_W": 600,
    "VIEWPORT_H": 400,
}

# CONTEXT_BOUNDS = {
#     "FPS": (1, 500),
#     "SCALE": (1, 100),   # affects how fast-paced the game is, forces should be adjusted as well
#
#     "FRICTION": (0, 10),
#
#     # random force on lunar lander body on reset
#     "INITIAL_RANDOM": (0, 2000),   # Set 1500 to make game harder
#
#     # lunar lander body specification
#     "LEG_AWAY": (0, 50),
#     "LEG_DOWN": (0, 50),
#     "LEG_W": (1, 10),
#     "LEG_H": (1, 20),
#     "LEG_SPRING_TORQUE": (0, 100),
#     "SIDE_ENGINE_HEIGHT": (1, 20),
#     "SIDE_ENGINE_AWAY": (1, 20),
#
#     # Size of world
#     "VIEWPORT_W": (400, 1000),
#     "VIEWPORT_H": (200, 800),
# }


class MetaLunarLanderEnv(MetaEnv):
    def __init__(
            self,
            env: gym.Env = bipedal_walker.BipedalWalker(),
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
        self._update_context()

    def _update_context(self):
        bipedal_walker.FRICTION = self.context["FRICTION"]

        # Important for building terrain
        self.env.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            friction=bipedal_walker.FRICTION)
        self.env.fd_edge = fixtureDef(
            shape=edgeShape(vertices=
                            [(0, 0),
                             (1, 1)]),
            friction=bipedal_walker.FRICTION,
            categoryBits=0x0001,
        )

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
