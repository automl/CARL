from typing import Any, Dict, List, Optional, Tuple, Union

import Box2D
import numpy as np
from Box2D.b2 import edgeShape, fixtureDef, polygonShape
from gym import spaces
from gym.envs.box2d import bipedal_walker
from gym.envs.box2d import bipedal_walker as bpw
from gym.utils import EzPickle

from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector

DEFAULT_CONTEXT = {
    "FPS": 50,
    "SCALE": 30.0,  # affects how fast-paced the game is, forces should be adjusted as well
    "GRAVITY_X": 0,
    "GRAVITY_Y": -10,
    # surroundings
    "FRICTION": 2.5,
    "TERRAIN_STEP": 14 / 30.0,
    "TERRAIN_LENGTH": 200,  # in steps
    "TERRAIN_HEIGHT": 600 / 30 / 4,  # VIEWPORT_H/SCALE/4
    "TERRAIN_GRASS": 10,  # low long are grass spots, in steps
    "TERRAIN_STARTPAD": 20,  # in steps
    # walker
    "MOTORS_TORQUE": 80,
    "SPEED_HIP": 4,
    "SPEED_KNEE": 6,
    "LIDAR_RANGE": 160 / 30.0,
    "LEG_DOWN": -8 / 30.0,
    "LEG_W": 8 / 30.0,
    "LEG_H": 34 / 30.0,
    # absolute value of random force applied to walker at start of episode
    "INITIAL_RANDOM": 5,
    # Size of world
    "VIEWPORT_W": 600,
    "VIEWPORT_H": 400,
}

# TODO make bounds more generous for all Box2D envs?
CONTEXT_BOUNDS = {
    "FPS": (1, 500, float),
    "SCALE": (
        1,
        100,
        float,
    ),  # affects how fast-paced the game is, forces should be adjusted as well
    # surroundings
    "FRICTION": (0, 10, float),
    "TERRAIN_STEP": (0.25, 1, float),
    "TERRAIN_LENGTH": (100, 500, int),  # in steps
    "TERRAIN_HEIGHT": (3, 10, float),  # VIEWPORT_H/SCALE/4
    "TERRAIN_GRASS": (5, 15, int),  # low long are grass spots, in steps
    "TERRAIN_STARTPAD": (10, 30, int),  # in steps
    # walker
    "MOTORS_TORQUE": (0, 200, float),
    "SPEED_HIP": (1e-6, 15, float),
    "SPEED_KNEE": (1e-6, 15, float),
    "LIDAR_RANGE": (0.5, 20, float),
    "LEG_DOWN": (-2, -0.25, float),
    "LEG_W": (0.25, 0.5, float),
    "LEG_H": (0.25, 2, float),
    # absolute value of random force applied to walker at start of episode
    "INITIAL_RANDOM": (0, 50, float),
    # Size of world
    "VIEWPORT_W": (400, 1000, int),
    "VIEWPORT_H": (200, 800, int),
    "GRAVITY_X": (-20, 20, float),  # unit: m/s²
    "GRAVITY_Y": (
        -20,
        -0.01,
        float,
    ),  # the y-component of gravity must be smaller than 0 because otherwise the
    # body leaves the frame by going up
}


class CARLBipedalWalkerEnv(CARLEnv):
    def __init__(
        self,
        env: Optional[bipedal_walker.BipedalWalker] = None,
        contexts: Dict[Any, Dict[Any, Any]] = {},
        hide_context: bool = False,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.05,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Dict] = DEFAULT_CONTEXT,
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
        if env is None:
            env = bipedal_walker.BipedalWalker()
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
            state_context_features=state_context_features,
            dict_observation_space=dict_observation_space,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs

        )
        self.whitelist_gaussian_noise = list(
            DEFAULT_CONTEXT.keys()
        )  # allow to augment all values

    def _update_context(self):
        bpw.FPS = self.context["FPS"]
        bpw.SCALE = self.context["SCALE"]
        bpw.FRICTION = self.context["FRICTION"]
        bpw.TERRAIN_STEP = self.context["TERRAIN_STEP"]
        bpw.TERRAIN_LENGTH = int(
            self.context["TERRAIN_LENGTH"]
        )  # TODO do this automatically
        bpw.TERRAIN_HEIGHT = self.context["TERRAIN_HEIGHT"]
        bpw.TERRAIN_GRASS = self.context["TERRAIN_GRASS"]
        bpw.TERRAIN_STARTPAD = self.context["TERRAIN_STARTPAD"]
        bpw.MOTORS_TORQUE = self.context["MOTORS_TORQUE"]
        bpw.SPEED_HIP = self.context["SPEED_HIP"]
        bpw.SPEED_KNEE = self.context["SPEED_KNEE"]
        bpw.LIDAR_RANGE = self.context["LIDAR_RANGE"]
        bpw.LEG_DOWN = self.context["LEG_DOWN"]
        bpw.LEG_W = self.context["LEG_W"]
        bpw.LEG_H = self.context["LEG_H"]
        bpw.INITIAL_RANDOM = self.context["INITIAL_RANDOM"]
        bpw.VIEWPORT_W = self.context["VIEWPORT_W"]
        bpw.VIEWPORT_H = self.context["VIEWPORT_H"]

        gravity_x = self.context["GRAVITY_X"]
        gravity_y = self.context["GRAVITY_Y"]

        gravity = (gravity_x, gravity_y)
        self.env.world.gravity = gravity

        # Important for building terrain
        self.env.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=bipedal_walker.FRICTION,
        )
        self.env.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=bipedal_walker.FRICTION,
            categoryBits=0x0001,
        )

        bpw.HULL_FD = fixtureDef(
            shape=polygonShape(
                vertices=[(x / bpw.SCALE, y / bpw.SCALE) for x, y in bpw.HULL_POLY]
            ),
            density=5.0,
            friction=0.1,
            categoryBits=0x0020,
            maskBits=0x001,  # collide only with ground
            restitution=0.0,
        )  # 0.99 bouncy

        bpw.LEG_FD = fixtureDef(
            shape=polygonShape(box=(bpw.LEG_W / 2, bpw.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001,
        )

        bpw.LOWER_FD = fixtureDef(
            shape=polygonShape(box=(0.8 * bpw.LEG_W / 2, bpw.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001,
        )

        self.env.world.gravity = gravity


def demo_heuristic(env: Union[CARLBipedalWalkerEnv, bipedal_walker.BipedalWalker]) -> None:
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4]]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9]]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        contact0 = s[8]  # noqa: F841
        contact1 = s[13]  # noqa: F841
        moving_s_base = 4 + 5 * moving_leg
        supporting_s_base = 4 + 5 * supporting_leg

        hip_targ = [None, None]  # -0.8 .. +1.1
        knee_targ = [None, None]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state == STAY_ON_ONE_LEG:
            hip_targ[moving_leg] = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED:
                supporting_knee_angle += 0.03
            supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state == PUT_OTHER_DOWN:
            hip_targ[moving_leg] = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base + 4]:
                state = PUSH_OFF
                supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
        if state == PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]:
            hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]:
            hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]:
            knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]:
            knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5 * a, -1.0, 1.0)

        env.render()
        if done:
            break


if __name__ == "__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = CARLBipedalWalkerEnv(add_gaussian_noise_to_context=True)
    for i in range(3):
        demo_heuristic(env)
    env.close()
