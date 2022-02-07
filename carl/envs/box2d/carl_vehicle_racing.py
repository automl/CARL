from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pyglet
from gym.envs.box2d import CarRacing
from gym.envs.box2d.car_dynamics import Car
from pyglet import gl

from carl.envs.box2d.parking_garage.bus import AWDBus  # as Car
from carl.envs.box2d.parking_garage.bus import AWDBusLargeTrailer  # as Car
from carl.envs.box2d.parking_garage.bus import AWDBusSmallTrailer  # as Car
from carl.envs.box2d.parking_garage.bus import Bus  # as Car
from carl.envs.box2d.parking_garage.bus import BusLargeTrailer  # as Car
from carl.envs.box2d.parking_garage.bus import BusSmallTrailer  # as Car
from carl.envs.box2d.parking_garage.bus import FWDBus  # as Car
from carl.envs.box2d.parking_garage.bus import FWDBusLargeTrailer  # as Car
from carl.envs.box2d.parking_garage.bus import FWDBusSmallTrailer  # as Car
from carl.envs.box2d.parking_garage.race_car import AWDRaceCar  # as Car
from carl.envs.box2d.parking_garage.race_car import AWDRaceCarLargeTrailer  # as Car
from carl.envs.box2d.parking_garage.race_car import AWDRaceCarSmallTrailer  # as Car
from carl.envs.box2d.parking_garage.race_car import FWDRaceCar  # as Car
from carl.envs.box2d.parking_garage.race_car import FWDRaceCarLargeTrailer  # as Car
from carl.envs.box2d.parking_garage.race_car import FWDRaceCarSmallTrailer  # as Car
from carl.envs.box2d.parking_garage.race_car import RaceCarLargeTrailer  # as Car
from carl.envs.box2d.parking_garage.race_car import RaceCarSmallTrailer  # as Car
from carl.envs.box2d.parking_garage.race_car import RaceCar
from carl.envs.box2d.parking_garage.street_car import AWDStreetCar  # as Car
from carl.envs.box2d.parking_garage.street_car import AWDStreetCarLargeTrailer  # as Car
from carl.envs.box2d.parking_garage.street_car import AWDStreetCarSmallTrailer  # as Car
from carl.envs.box2d.parking_garage.street_car import FWDStreetCar  # as Car
from carl.envs.box2d.parking_garage.street_car import FWDStreetCarLargeTrailer  # as Car
from carl.envs.box2d.parking_garage.street_car import FWDStreetCarSmallTrailer  # as Car
from carl.envs.box2d.parking_garage.street_car import StreetCar  # as Car
from carl.envs.box2d.parking_garage.street_car import StreetCarLargeTrailer  # as Car
from carl.envs.box2d.parking_garage.street_car import StreetCarSmallTrailer  # as Car
from carl.envs.box2d.parking_garage.trike import TukTuk  # as Car
from carl.envs.box2d.parking_garage.trike import TukTukSmallTrailer  # as Car
from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector

PARKING_GARAGE_DICT = {
    # Racing car
    "RaceCar": RaceCar,
    "FWDRaceCar": FWDRaceCar,
    "AWDRaceCar": AWDRaceCar,
    "RaceCarSmallTrailer": RaceCarSmallTrailer,
    "FWDRaceCarSmallTrailer": FWDRaceCarSmallTrailer,
    "AWDRaceCarSmallTrailer": AWDRaceCarSmallTrailer,
    "RaceCarLargeTrailer": RaceCarLargeTrailer,
    "FWDRaceCarLargeTrailer": FWDRaceCarLargeTrailer,
    "AWDRaceCarLargeTrailer": AWDRaceCarLargeTrailer,
    # Street car
    "StreetCar": StreetCar,
    "FWDStreetCar": FWDStreetCar,
    "AWDStreetCar": AWDStreetCar,
    "StreetCarSmallTrailer": StreetCarSmallTrailer,
    "FWDStreetCarSmallTrailer": FWDStreetCarSmallTrailer,
    "AWDStreetCarSmallTrailer": AWDStreetCarSmallTrailer,
    "StreetCarLargeTrailer": StreetCarLargeTrailer,
    "FWDStreetCarLargeTrailer": FWDStreetCarLargeTrailer,
    "AWDStreetCarLargeTrailer": AWDStreetCarLargeTrailer,
    # Bus
    "Bus": Bus,
    "FWDBus": FWDBus,
    "AWDBus": AWDBus,
    "BusSmallTrailer": BusSmallTrailer,
    "FWDBusSmallTrailer": FWDBusSmallTrailer,
    "AWDBusSmallTrailer": AWDBusSmallTrailer,
    "BusLargeTrailer": BusLargeTrailer,
    "FWDBusLargeTrailer": FWDBusLargeTrailer,
    "AWDBusLargeTrailer": AWDBusLargeTrailer,
    # Tuk Tuk :)
    "TukTuk": TukTuk,
    "TukTukSmallTrailer": TukTukSmallTrailer,
}
PARKING_GARAGE = list(PARKING_GARAGE_DICT.values())
VEHICLE_NAMES = list(PARKING_GARAGE_DICT.keys())
DEFAULT_CONTEXT = {
    "VEHICLE": PARKING_GARAGE.index(RaceCar),
}

CONTEXT_BOUNDS = {
    "VEHICLE": (None, None, "categorical", np.arange(0, len(PARKING_GARAGE)))
}
CATEGORICAL_CONTEXT_FEATURES = ["VEHICLE"]


class CustomCarRacingEnv(CarRacing):
    def __init__(self, vehicle_class: Type[Car] = Car, verbose: int = 1):
        super().__init__(verbose)
        self.vehicle_class = vehicle_class

    def reset(self) -> np.ndarray:
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []  # type: List

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = self.vehicle_class(self.world, *self.track[0][1:4])

        for i in range(
            49
        ):  # this sets up the environment and resolves any initial violations of geometry
            self.step(None)
        return self.step(None)[0]

    def render_indicators(self, W: int, H: int) -> None:
        # copied from meta car racing
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place: int, val: int, color: Tuple) -> None:
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

        def horiz_ind(place: int, val: int, color: Tuple) -> None:
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
            vertical_ind(7 + i, 0.01 * self.car.wheels[i].omega, (0.0 + i * 0.1, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


class CARLVehicleRacingEnv(CARLEnv):
    def __init__(
        self,
        env: CustomCarRacingEnv = CustomCarRacingEnv(),
        contexts: Optional[Dict[Union[str, int], Dict[Any, Any]]] = None,
        hide_context: bool = True,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
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
        if not hide_context:
            raise NotImplementedError(
                "The context is already coded in the pixel state, the context cannot be hidden that easily. "
                "Due to the pixel state we cannot easily concatenate the context to the state, therefore "
                "hide_context must be True but at the same time the context is visible via the pixel state."
            )

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
            context_selector_kwargs=context_selector_kwargs,
        )
        self.whitelist_gaussian_noise = [
            k for k in DEFAULT_CONTEXT.keys() if k not in CATEGORICAL_CONTEXT_FEATURES
        ]

    def _update_context(self) -> None:
        vehicle_class_index = self.context["VEHICLE"]
        self.env.vehicle_class = PARKING_GARAGE[vehicle_class_index]


if __name__ == "__main__":
    import time

    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k: int, mod: Any) -> None:
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
            a[2] = +1.0

    def key_release(k: int, mod: Any) -> None:
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    contexts = {i: {"VEHICLE": i} for i in range(len(VEHICLE_NAMES))}
    env = CARLVehicleRacingEnv(contexts=contexts)
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
            if done or restart or not isopen:
                break
    env.close()
