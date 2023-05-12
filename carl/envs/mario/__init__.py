# flake8: noqa: F401
from functools import partial
import warnings

try:
    from carl.envs.mario.carl_mario import CARLMarioEnv
    from gym.envs.registration import register
    
    register("CARLMarioEnv-v0", entry_point=CARLMarioEnv)
except Exception as e:
    warnings.warn(f"Could not load CARLMarioEnv which is probably not installed ({e}).")

from carl.envs.mario.carl_mario_definitions import (
    DEFAULT_CONTEXT as CARLMarioEnv_defaults,
    CONTEXT_BOUNDS as CARLMarioEnv_bounds,
)
