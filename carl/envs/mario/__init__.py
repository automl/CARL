# flake8: noqa: F401
import warnings
from functools import partial

try:
    from gym.envs.registration import register

    from carl.envs.mario.carl_mario import CARLMarioEnv

    register("CARLMarioEnv-v0", entry_point=CARLMarioEnv)
except Exception as e:
    warnings.warn(f"Could not load CARLMarioEnv which is probably not installed ({e}).")

from carl.envs.mario.carl_mario_definitions import CONTEXT_BOUNDS as CARLMarioEnv_bounds
from carl.envs.mario.carl_mario_definitions import (
    DEFAULT_CONTEXT as CARLMarioEnv_defaults,
)
