from __future__ import annotations

from carl.envs.gymnax.carl_gymnax_env import CARLGymnaxEnv
from carl.context.context_space import ContextFeature, UniformFloatContextFeature


class CARLGymnaxPendulum(CARLGymnaxEnv):
    env_name: str = "Pendulum-v1"
    module: str = "classic_control.pendulum"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "dt": UniformFloatContextFeature(
                "dt", lower=0.001, upper=10, default_value=0.05
            ),
            "g": UniformFloatContextFeature(
                "g", lower=-100, upper=100, default_value=10
            ),
            "m": UniformFloatContextFeature(
                "m", lower=1e-6, upper=100, default_value=1
            ),
            "l": UniformFloatContextFeature(
                "l", lower=1e-6, upper=100, default_value=1
            ),
            "max_speed": UniformFloatContextFeature(
                "max_speed", lower=0.08, upper=80, default_value=8
            ),
            "max_torque": UniformFloatContextFeature(
                "max_torque", lower=0.02, upper=40, default_value=2
            ),
        }
