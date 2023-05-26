# flake8: noqa: F401

# Contextenvs.s and bounds by name
import warnings
from functools import partial

import gym

from carl.envs.box2d.carl_bipedal_walker import (
    CONTEXT_BOUNDS as CARLBipedalWalkerEnv_bounds,
)
from carl.envs.box2d.carl_bipedal_walker import (
    DEFAULT_CONTEXT as CARLBipedalWalkerEnv_defaults,
)
from carl.envs.box2d.carl_bipedal_walker import CARLBipedalWalkerEnv
from carl.envs.box2d.carl_lunarlander import (
    DEFAULT_CONTEXT as CARLLunarLanderEnv_defaults,
)
from carl.envs.box2d.carl_lunarlander import CARLLunarLanderEnv
from carl.envs.box2d.carl_vehicle_racing import (
    CONTEXT_BOUNDS as CARLVehicleRacingEnv_bounds,
)
from carl.envs.box2d.carl_vehicle_racing import (
    DEFAULT_CONTEXT as CARLVehicleRacingEnv_defaults,
)
from carl.envs.box2d.carl_vehicle_racing import CARLVehicleRacingEnv

try:
    from gym.envs.registration import register

    from carl.envs.box2d.carl_bipedal_walker import CARLBipedalWalkerEnv

    def make_env(**kwargs):
        return CARLBipedalWalkerEnv(**kwargs)

    register("CARLBipedalWalkerEnv-v0", entry_point=make_env)
    register(
        "CARLBipedalWalkerHardcoreEnv-v0",
        entry_point=partial(make_env, env=gym.make("BipedalWalkerHardcore-v3")),
    )
except Exception as e:
    warnings.warn(f"Could not load CARLMarioEnv which is probably not installed ({e}).")
