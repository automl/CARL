from __future__ import annotations

import importlib

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.gymnax.carl_gymnax_env import CARLGymnaxEnv


class CARLGymnaxCartPole(CARLGymnaxEnv):
    env_name: str = "CartPole-v1"
    module: str = "classic_control.cartpole"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "gravity": UniformFloatContextFeature(
                "gravity", lower=0.01, upper=100, default_value=9.8
            ),
            "masscart": UniformFloatContextFeature(
                "masscart", lower=0.1, upper=10, default_value=1.0
            ),
            "masspole": UniformFloatContextFeature(
                "masspole", lower=0.01, upper=1, default_value=0.1
            ),
            "length": UniformFloatContextFeature(
                "length", lower=0.05, upper=5, default_value=0.5
            ),
            "force_mag": UniformFloatContextFeature(
                "force_mag", lower=1, upper=100, default_value=10.0
            ),
            "tau": UniformFloatContextFeature(
                "tau", lower=0.002, upper=0.2, default_value=0.02
            ),
        }

    def _update_context(self) -> None:
        content = self.env.env_params.__dict__
        content.update(self.context)
        content["total_mass"] = content["masspole"] + content["masscart"]
        content["polemass_length"] = content["masspole"] * content["length"]

        # TODO Make this faster by preloading module?
        self.env.env.env_params = getattr(
            importlib.import_module(f"gymnax.environments.{self.module}"), "EnvParams"
        )(**content)
