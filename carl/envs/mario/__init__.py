import warnings

try:
    from carl.envs.mario.carl_mario import CARLMarioEnv
except Exception as e:
    warnings.warn(f"Could not load CARLMarioEnv which is probably not installed ({e}).")

from carl.envs.mario.carl_mario_definitions import (
    DEFAULT_CONTEXT as CARLMarioEnv_defaults,
    CONTEXT_BOUNDS as CARLMarioEnv_bounds,
)
