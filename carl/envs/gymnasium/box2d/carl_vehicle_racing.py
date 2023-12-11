from __future__ import annotations

from typing import Optional, Type, Union

import numpy as np
import pygame
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.envs.box2d.car_racing import CarRacing, FrictionDetector
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
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = self.vehicle_class(self.world, *self.track[0][1:4])

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # Custom render to handle different amounts of wheels
        for i in range(len(self.car.wheels)):  # type: ignore [attr-defined]
            render_if_min(
                self.car.wheels[i].omega,
                vertical_ind(7 + i, 0.01 * self.car.wheels[i].omega),
                (0 + i * 10, 0, 255),
            )  # type: ignore [attr-defined]
        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )


register(
    id="CustomCarRacing-v2",
    entry_point="carl.envs.gymnasium.box2d.carl_vehicle_racing:CustomCarRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)


class CARLVehicleRacing(CARLGymnasiumEnv):
    env_name: str = "CustomCarRacing-v2"
    metadata = {"render.modes": ["human", "rgb_array"]}

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
        self.env.unwrapped.vehicle_class = PARKING_GARAGE[vehicle_class_index]
