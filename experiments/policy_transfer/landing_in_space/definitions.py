"""
POLICY TRANSFER
=======================================================================

g = 9.80665 m/s²
train on moon = 0.166g

test policy transfer:
    in distribution:
        Mars (0.377g)
        Pluto (0.071g)
    out of distribution:
        Jupiter (2.36g)
        Neptune (1.12g)

https://nssdc.gsfc.nasa.gov/planetary/factsheet/planet_table_ratio.html
"""
import numpy as np
from numpy.random import Generator
from typing import Dict, Tuple, Optional, List, Any
from carl.utils.types import Contexts
from carl.envs.box2d.carl_vehicle_racing import (
    RaceCar,
    AWDRaceCar,
    StreetCar,
    TukTuk,
    BusSmallTrailer,
    PARKING_GARAGE,
)


# Experiment 1: LunarLander
g_earth = -9.80665  # m/s², beware of coordinate systems

gravities = {
    "Jupiter": g_earth * 2.36,
    "Neptune": g_earth * 1.12,
    "Earth": g_earth * 1,
    "Mars": g_earth * 0.377,
    "Moon": g_earth * 0.166,
    "Pluto": g_earth * 0.071,
}

planets_train = ["Moon"]
planet_train = "Moon"
planets_test_in = ["Mars", "Pluto"]
planets_test_out = ["Jupiter", "Neptune"]

outdir = "results/experiments/policytransfer"

# Experiment 2: CARLVehicleRacingEnv
vehicle_train = "RaceCar"
vehicles = {
    "RaceCar": PARKING_GARAGE.index(RaceCar),
    "StreetCar": PARKING_GARAGE.index(StreetCar),
    "TukTuk": PARKING_GARAGE.index(TukTuk),
    "AWDRaceCar": PARKING_GARAGE.index(AWDRaceCar),
    "BusSmallTrailer": PARKING_GARAGE.index(BusSmallTrailer),
}


def sample_gravities_normal(
    mean: float, std: float, seed: int, n_contexts: int
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    gravities = rng.normal(loc=mean, scale=std, size=n_contexts)
    gravities[
        gravities > 0
    ] = 0  # gravity is defined in -Y direction. if it is greater 0, the spacecraft would be repelled
    return gravities


def sample_gravities_uniform(
    intervals: List[Tuple[float, float]], seed: int, n_contexts: int
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    G = []
    n_intervals = len(intervals)
    n_contexts_per_interval = n_contexts // n_intervals
    for interval in intervals:
        gravities = rng.uniform(
            low=interval[0], high=interval[1], size=n_contexts_per_interval
        )
        G.append(gravities)
    gravities = np.concatenate(G)
    gravities[gravities > 0] = 0
    return gravities


def create_test_gravity_contexts() -> Contexts:
    contexts = {planet_name: {"GRAVITY_Y": gravity} for planet_name, gravity in gravities.items()}
    return contexts
