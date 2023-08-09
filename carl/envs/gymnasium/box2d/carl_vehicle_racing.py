from __future__ import annotations

from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.registration import register

from carl.context.context_space import ContextFeature, UniformIntegerContextFeature
from carl.envs.gymnasium.box2d.parking_garage.bus import AWDBus  # as Car
from carl.envs.gymnasium.box2d.parking_garage.bus import AWDBusLargeTrailer  # as Car
from carl.envs.gymnasium.box2d.parking_garage.bus import AWDBusSmallTrailer  # as Car
from carl.envs.gymnasium.box2d.parking_garage.bus import Bus  # as Car
from carl.envs.gymnasium.box2d.parking_garage.bus import BusLargeTrailer  # as Car
from carl.envs.gymnasium.box2d.parking_garage.bus import BusSmallTrailer  # as Car
from carl.envs.gymnasium.box2d.parking_garage.bus import FWDBus  # as Car
from carl.envs.gymnasium.box2d.parking_garage.bus import FWDBusLargeTrailer  # as Car
from carl.envs.gymnasium.box2d.parking_garage.bus import FWDBusSmallTrailer  # as Car
from carl.envs.gymnasium.box2d.parking_garage.race_car import AWDRaceCar  # as Car
from carl.envs.gymnasium.box2d.parking_garage.race_car import FWDRaceCar  # as Car
from carl.envs.gymnasium.box2d.parking_garage.race_car import (  # as Car
    AWDRaceCarLargeTrailer,
    AWDRaceCarSmallTrailer,
    FWDRaceCarLargeTrailer,
    FWDRaceCarSmallTrailer,
    RaceCar,
    RaceCarLargeTrailer,
    RaceCarSmallTrailer,
)
from carl.envs.gymnasium.box2d.parking_garage.street_car import AWDStreetCar  # as Car
from carl.envs.gymnasium.box2d.parking_garage.street_car import FWDStreetCar  # as Car
from carl.envs.gymnasium.box2d.parking_garage.street_car import StreetCar  # as Car
from carl.envs.gymnasium.box2d.parking_garage.street_car import (  # as Car
    AWDStreetCarLargeTrailer,
    AWDStreetCarSmallTrailer,
    FWDStreetCarLargeTrailer,
    FWDStreetCarSmallTrailer,
    StreetCarLargeTrailer,
    StreetCarSmallTrailer,
)
from carl.envs.gymnasium.box2d.parking_garage.trike import TukTuk  # as Car
from carl.envs.gymnasium.box2d.parking_garage.trike import TukTukSmallTrailer  # as Car
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv
from carl.utils.types import ObsType

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


class CustomCarRacing(CarRacing):
    def __init__(
        self,
        vehicle_class: Type[Car] = Car,
        verbose: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__(verbose=verbose, render_mode=render_mode)
        self.vehicle_class = vehicle_class

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = True,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly: List[Tuple[List[float], Tuple[Any]]] = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = self.vehicle_class(self.world, *self.track[0][1:4])  # type: ignore [assignment]

        for i in range(
            49
        ):  # this sets up the environment and resolves any initial violations of geometry
            self.step(None)  # type: ignore [arg-type]

        return self.step(None)[0], {}

    def _render_indicators_BROKEN(self, W: int, H: int) -> None:
        # TODO Fix CarRacing rendering
        # copied from meta car racing
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place: int, val: int, color: Tuple) -> None:
            points = [
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
            C = [color[0], color[1], color[2], 1]  # * 4
            colors.extend(C)
            polygons.extend(points)
            self._draw_colored_polygon(
                self.surf, points, C, zoom=1, translation=[0, 0], angle=0, clip=True
            )

        def horiz_ind(place: int, val: int, color: Tuple) -> None:
            points = [
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
            C = [color[0], color[1], color[2], 1]  # * 4
            colors.extend(C)
            polygons.extend(points)
            self._draw_colored_polygon(
                self.surf, points, C, zoom=1, translation=[0, 0], angle=0, clip=True
            )

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])  # type: ignore [attr-defined]
            + np.square(self.car.hull.linearVelocity[1])  # type: ignore [attr-defined]
        )

        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))

        # Custom render to handle different amounts of wheels
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # type: ignore [attr-defined]
        for i in range(len(self.car.wheels)):  # type: ignore [attr-defined]
            vertical_ind(7 + i, 0.01 * self.car.wheels[i].omega, (0.0 + i * 0.1, 0, 1))  # type: ignore [attr-defined]
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))  # type: ignore [attr-defined]
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))  # type: ignore [attr-defined]
        # vl = pyglet.graphics.vertex_list(
        #     len(polygons) // 3, ("v3f", polygons), ("c4f", colors)  # gl.GL_QUADS,
        # )
        # vl.draw(gl.GL_QUADS)

        # shader_program = pyglet.graphics.get_default_shader()

        # mode = gl.GL_POLYGON_MODE

        # vertex_positions = polygons
        # vl = shader_program.vertex_list(
        #     count=len(polygons) // 3,
        #     mode=mode,
        #     position=('f', vertex_positions),
        #     colors=('f', colors)
        # )
        # vl.draw(mode)


register(
    id="CustomCarRacing-v2",
    entry_point="carl.envs.gymnasium.box2d.carl_vehicle_racing:CustomCarRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)


class CARLVehicleRacing(CARLGymnasiumEnv):
    env_name: str = "CustomCarRacing-v2"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "VEHICLE_ID": UniformIntegerContextFeature(
                "VEHICLE_ID", lower=0, upper=len(PARKING_GARAGE) - 1, default_value=0
            )  # RaceCar
        }

    def _update_context(self) -> None:
        self.env: CustomCarRacing
        vehicle_class_index = self.context["VEHICLE_ID"]
        self.env.vehicle_class = PARKING_GARAGE[vehicle_class_index]
        print(self.env.vehicle_class)
