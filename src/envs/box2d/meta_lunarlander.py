import numpy as np
import math
from typing import Dict, Optional, Type, List

# import pyglet
# pyglet.options["shadow_window"] = False

import gym
import Box2D
# from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from gym.envs.box2d import lunar_lander
from gym.envs.box2d import lunar_lander as ll
from gym.envs.box2d.lunar_lander import heuristic
from gym import spaces
from gym.utils import seeding, EzPickle

from src.envs.meta_env import MetaEnv
from src.trial_logger import TrialLogger
from src.envs.box2d.utils import safe_destroy

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
    "FPS": (1, 500, float),
    "SCALE": (1, 100, float),   # affects how fast-paced the game is, forces should be adjusted as well
    "MAIN_ENGINE_POWER": (0, 50, float),
    "SIDE_ENGINE_POWER": (0, 50, float),

    # random force on lunar lander body on reset
    "INITIAL_RANDOM": (0, 2000, float),   # Set 1500 to make game harder

    "GRAVITY_X": (-20, 20, float),  # unit: m/s²
    "GRAVITY_Y": (-20, -0.01, float),   # the y-component of gravity must be smaller than 0 because otherwise the
                                 # lunarlander leaves the frame by going up

    # lunar lander body specification
    "LEG_AWAY": (0, 50, float),
    "LEG_DOWN": (0, 50, float),
    "LEG_W": (1, 10, float),
    "LEG_H": (1, 20, float),
    "LEG_SPRING_TORQUE": (0, 100, float),
    "SIDE_ENGINE_HEIGHT": (1, 20, float),
    "SIDE_ENGINE_AWAY": (1, 20, float),

    # Size of world
    "VIEWPORT_W": (400, 1000, int),
    "VIEWPORT_H": (200, 800, int),
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

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        bodies = [self.moon, self.lander] + self.legs
        # safe destroy because before calling destroy we already created a new world
        # which does not have the bodies anymore
        safe_destroy(self.world, bodies)
        self.moon = None
        self.lander = None

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            safe_destroy(self.world, [self.particles.pop(0)])

    # def reset(self):
    #     self._destroy()
    #     self.world.contactListener_keepref = ll.ContactDetector(self)
    #     self.world.contactListener = self.world.contactListener_keepref
    #     self.game_over = False
    #     self.prev_shaping = None
    #
    #     W = ll.VIEWPORT_W/ll.SCALE
    #     H = ll.VIEWPORT_H/ll.SCALE
    #
    #     # terrain
    #     CHUNKS = 11
    #     height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,))
    #     chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
    #     self.helipad_x1 = chunk_x[CHUNKS//2-1]
    #     self.helipad_x2 = chunk_x[CHUNKS//2+1]
    #     self.helipad_y = H/4
    #     height[CHUNKS//2-2] = self.helipad_y
    #     height[CHUNKS//2-1] = self.helipad_y
    #     height[CHUNKS//2+0] = self.helipad_y
    #     height[CHUNKS//2+1] = self.helipad_y
    #     height[CHUNKS//2+2] = self.helipad_y
    #     smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]
    #
    #     self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
    #     self.sky_polys = []
    #     for i in range(CHUNKS-1):
    #         p1 = (chunk_x[i], smooth_y[i])
    #         p2 = (chunk_x[i+1], smooth_y[i+1])
    #         self.moon.CreateEdgeFixture(
    #             vertices=[p1,p2],
    #             density=0,
    #             friction=0.1)
    #         self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
    #
    #     self.moon.color1 = (0.5, 0.5, 0.5)
    #     self.moon.color2 = (0.0, 1, 0.0)
    #
    #     initial_y = ll.VIEWPORT_H/ll.SCALE
    #     self.lander = self.world.CreateDynamicBody(
    #         position=(ll.VIEWPORT_W/ll.SCALE/2, initial_y),
    #         angle=0.0,
    #         fixtures = fixtureDef(
    #             shape=polygonShape(vertices=[(x/ll.SCALE, y/ll.SCALE) for x, y in ll.LANDER_POLY]),
    #             density=5.0,
    #             friction=0.1,
    #             categoryBits=0x0010,
    #             maskBits=0x001,   # collide only with ground
    #             restitution=0.0)  # 0.99 bouncy
    #             )
    #     self.lander.color1 = (0.5, 0.4, 0.9)
    #     self.lander.color2 = (0.3, 0.3, 0.5)
    #     self.lander.ApplyForceToCenter( (
    #         self.np_random.uniform(-ll.INITIAL_RANDOM, ll.INITIAL_RANDOM),
    #         self.np_random.uniform(-ll.INITIAL_RANDOM, ll.INITIAL_RANDOM)
    #         ), True)
    #
    #     self.legs = []
    #     for i in [-1, +1]:
    #         leg = self.world.CreateDynamicBody(
    #             position=(ll.VIEWPORT_W/ll.SCALE/2 - i*ll.LEG_AWAY/ll.SCALE, initial_y),
    #             angle=(i * 0.05),
    #             fixtures=fixtureDef(
    #                 shape=polygonShape(box=(ll.LEG_W/ll.SCALE, ll.LEG_H/ll.SCALE)),
    #                 density=1.0,
    #                 restitution=0.0,
    #                 categoryBits=0x0020,
    #                 maskBits=0x001)
    #             )
    #         leg.ground_contact = False
    #         leg.color1 = (0.5, 0.4, 0.9)
    #         leg.color2 = (0.3, 0.3, 0.5)
    #         rjd = revoluteJointDef(
    #             bodyA=self.lander,
    #             bodyB=leg,
    #             localAnchorA=(0, 0),
    #             localAnchorB=(i * ll.LEG_AWAY/ll.SCALE, ll.LEG_DOWN/ll.SCALE),
    #             enableMotor=True,
    #             enableLimit=True,
    #             maxMotorTorque=ll.LEG_SPRING_TORQUE,
    #             motorSpeed=+0.3 * i  # low enough not to jump back into the sky
    #             )
    #         if i == -1:
    #             rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
    #             rjd.upperAngle = +0.9
    #         else:
    #             rjd.lowerAngle = -0.9
    #             rjd.upperAngle = -0.9 + 0.5
    #         leg.joint = self.world.CreateJoint(rjd)
    #         self.legs.append(leg)
    #
    #     self.drawlist = [self.lander] + self.legs
    #
    #     return self.step(np.array([0, 0]) if self.continuous else 0)[0]
    #
    # def render(self, mode='human'):
    #     from gym.envs.classic_control import rendering
    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(ll.VIEWPORT_W, ll.VIEWPORT_H)
    #         self.viewer.set_bounds(0, ll.VIEWPORT_W/ll.SCALE, 0, ll.VIEWPORT_H/ll.SCALE)
    #
    #     for obj in self.particles:
    #         obj.ttl -= 0.15
    #         obj.color1 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))
    #         obj.color2 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))
    #
    #     self._clean_particles(False)
    #
    #     for p in self.sky_polys:
    #         self.viewer.draw_polygon(p, color=(0, 0, 0))
    #
    #     for obj in self.particles + self.drawlist:
    #         for f in obj.fixtures:
    #             trans = f.body.transform
    #             if type(f.shape) is circleShape:
    #                 t = rendering.Transform(translation=trans*f.shape.pos)
    #                 self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
    #                 self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
    #             else:
    #                 path = [trans*v for v in f.shape.vertices]
    #                 self.viewer.draw_polygon(path, color=obj.color1)
    #                 path.append(path[0])
    #                 self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
    #
    #     for x in [self.helipad_x1, self.helipad_x2]:
    #         flagy1 = self.helipad_y
    #         flagy2 = flagy1 + 50/ll.SCALE
    #         self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
    #         self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/ll.SCALE), (x + 25/ll.SCALE, flagy2 - 5/ll.SCALE)],
    #                                  color=(0.8, 0.8, 0))
    #
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class MetaLunarLanderEnv(MetaEnv):
    def __init__(
            self,
            env: Optional[CustomLunarLanderEnv] = None,
            contexts: Dict[str, Dict] = {},
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = False,
            gaussian_noise_std_percentage: float = 0.05,
            logger: Optional[TrialLogger] = None,
            scale_context_features: str = "no",
            default_context: Optional[Dict] = DEFAULT_CONTEXT,
            state_context_features: Optional[List[str]] = None,
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
            env = lunar_lander.LunarLander()
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
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
            state_context_features=state_context_features,
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT.keys())  # allow to augment all values
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

        gravity_x = self.context["GRAVITY_X"]
        gravity_y = self.context["GRAVITY_Y"]

        gravity = (gravity_x, gravity_y)
        self.env.world.gravity = gravity


def demo_heuristic_lander(env, seed=None, render=False):
    """
    Copied from LunarLander
    """
    env.seed(seed)
    total_reward = 0
    steps = 0
    env.render()
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
            # print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done:
            break
    return total_reward


if __name__ == '__main__':
    env = MetaLunarLanderEnv(hide_context=False, add_gaussian_noise_to_context=True, gaussian_noise_std_percentage=0.1)
    # env.render()  # initialize viewer. otherwise weird bug.
    # env = ll.LunarLander()
    # env = CustomLunarLanderEnv()
    for i in range(5):
        demo_heuristic_lander(env, seed=1, render=True)
    env.close()
