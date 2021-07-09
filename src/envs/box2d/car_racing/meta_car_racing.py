import numpy as np
from gym.envs.box2d import CarRacing

from parking_garage.race_car import RaceCar as Car
from parking_garage.race_car import FWDRaceCar  # as Car
from parking_garage.race_car import AWDRaceCar  # as Car
from parking_garage.race_car import RaceCarSmallTrailer  # as Car
from parking_garage.race_car import FWDRaceCarSmallTrailer  # as Car
from parking_garage.race_car import AWDRaceCarSmallTrailer  # as Car
from parking_garage.race_car import RaceCarLargeTrailer  # as Car
from parking_garage.race_car import FWDRaceCarLargeTrailer  # as Car
from parking_garage.race_car import AWDRaceCarLargeTrailer  # as Car

from parking_garage.street_car import StreetCar  # as Car
from parking_garage.street_car import FWDStreetCar  # as Car
from parking_garage.street_car import AWDStreetCar  # as Car
from parking_garage.street_car import StreetCarSmallTrailer  # as Car
from parking_garage.street_car import FWDStreetCarSmallTrailer  # as Car
from parking_garage.street_car import AWDStreetCarSmallTrailer  # as Car
from parking_garage.street_car import StreetCarLargeTrailer  # as Car
from parking_garage.street_car import FWDStreetCarLargeTrailer  # as Car
from parking_garage.street_car import AWDStreetCarLargeTrailer  # as Car

from parking_garage.trike import TukTuk  # as Car
from parking_garage.trike import TukTukSmallTrailer  # as Car

from parking_garage.bus import Bus  # as Car
from parking_garage.bus import FWDBus  # as Car
from parking_garage.bus import AWDBus  # as Car
from parking_garage.bus import BusSmallTrailer  # as Car
from parking_garage.bus import FWDBusSmallTrailer  # as Car
from parking_garage.bus import AWDBusSmallTrailer  # as Car
from parking_garage.bus import BusLargeTrailer  # as Car
from parking_garage.bus import FWDBusLargeTrailer  # as Car
from parking_garage.bus import AWDBusLargeTrailer  # as Car

import pyglet

pyglet.options["debug_gl"] = False
from pyglet import gl


class MetaCarRacing(CarRacing):

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.instance_set = np.array([Car, FWDRaceCar, AWDRaceCar,
                                RaceCarSmallTrailer, FWDRaceCarSmallTrailer, AWDRaceCarSmallTrailer,
                                RaceCarLargeTrailer, FWDRaceCarLargeTrailer, AWDRaceCarLargeTrailer,
                                StreetCar, FWDStreetCar, AWDStreetCar,
                                StreetCarSmallTrailer, FWDStreetCarSmallTrailer, AWDStreetCarSmallTrailer,
                                StreetCarLargeTrailer, FWDStreetCarLargeTrailer, AWDStreetCarLargeTrailer,
                                Bus, FWDBus, AWDBus,
                                BusSmallTrailer, FWDBusSmallTrailer, AWDBusSmallTrailer,
                                BusLargeTrailer, FWDBusLargeTrailer, AWDBusLargeTrailer,
                                TukTuk, TukTukSmallTrailer
                                ])
        self.index = 0
        self.instances =[0]
        self.instance_set_size = 1
        self.curr_set = self.instance_set[self.instances]
        self.test = None

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.index = (self.index +1) % len(self.instance_set)
        car = self.instance_set[self.index]
        self.car = car(self.world, *self.track[0][1:4])

        for i in range(49):   # this sets up the environment and resolves any initial vilations of geometry
            self.step(None)
        return self.step(None)[0]

    def get_instance_set(self):
        return self.instances, self.curr_set

    def get_id(self):
        return self.index

    def get_feats(self):
        return len(self.instance_set)

    def get_instance_size(self):
        return int(np.ceil(len(self.instances_set) * self.instance_set_size))

    def increase_set_size(self, kappa):
        self.instance_set_size += kappa/len(self.instance_set)

    def set_test(self):
        self.test = test

    def set_instance_set(self, indices):
        size = self.get_instance_size()
        if size == 0:
            size = 1
        self.curr_set = np.array(self.instance_set)[indices[:size]]
        self.instances = indices[:size]

    def render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    place * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h,
                    0,
                    (place + 0) * s,
                    h,
                    0,
                ]
            )

        def horiz_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    (place + 0) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    2 * h,
                    0,
                    (place + 0) * s,
                    2 * h,
                    0,
                ]
            )

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))


        # Custom render to handle different amounts of wheels
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        for i in range(len(self.car.wheels)):
            vertical_ind(7+i, 0.01 * self.car.wheels[i].omega, (0.0+i*0.1, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


if __name__ == "__main__":
    from pyglet.window import key
    import time

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.SPACE:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +1.

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = MetaCarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            time.sleep(0.025)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()

