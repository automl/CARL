# flake8: noqa: F401
import warnings

try:
    from carl.envs.mario.carl_mario import CARLMarioEnv
except Exception as e:
    warnings.warn(f"Could not load CARLMarioEnv which is probably not installed ({e}).")

__all__ = ["CARLMarioEnv"]
