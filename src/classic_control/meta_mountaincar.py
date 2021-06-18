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
}


class MountainCarEnv(MetaEnv):
    def __init__(
            self,
            env: gym.Env = gccenvs.mountain_car.MountainCarEnv,
            contexts: List[Dict] = [],
            instance_mode: str = "rr"
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
            contexts = [DEFAULT_CONTEXT]
        super().__init__(env=env, contexts=contexts, instance_mode=instance_mode)

        self._update_context()
        self.viewer = None
        self.seed()

    def _update_context(self):
        self.min_position = self.context["min_position"]
        self.max_position = self.context["max_position"]
        self.max_speed = self.context["max_speed"]
        self.goal_position = self.context["goal_position"]
        self.goal_velocity = self.context["goal_velocity"]
        self.force = self.context["force"]
        self.gravity = self.context["gravity"]

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0):
            velocity = 0

        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}


    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)


    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

