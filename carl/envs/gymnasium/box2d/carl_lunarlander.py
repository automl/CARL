from __future__ import annotations

from Box2D.b2 import vec2
from gymnasium.envs.box2d import lunar_lander
from gymnasium.envs.box2d.lunar_lander import LunarLander

from carl.context.context_space import (
    ContextFeature,
    UniformFloatContextFeature,
    UniformIntegerContextFeature,
)
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv


class CARLLunarLander(CARLGymnasiumEnv):
    env_name: str = "LunarLander-v2"
    metadata = {"render.modes": ["human", "rgb_array"]}

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "FPS": UniformFloatContextFeature(
                "FPS", lower=1, upper=500, default_value=50
            ),
            "SCALE": UniformFloatContextFeature(
                "SCALE", lower=1, upper=100, default_value=30.0
            ),  # affects how fast-paced the game is, forces should be adjusted as well
            # Engine powers
            "MAIN_ENGINE_POWER": UniformFloatContextFeature(
                "MAIN_ENGINE_POWER", lower=0, upper=50, default_value=13.0
            ),
            "SIDE_ENGINE_POWER": UniformFloatContextFeature(
                "SIDE_ENGINE_POWER", lower=0, upper=50, default_value=0.6
            ),
            # random force on lunar lander body on reset
            "INITIAL_RANDOM": UniformFloatContextFeature(
                "INITIAL_RANDOM", lower=0, upper=2000, default_value=1000.0
            ),  # Set 1500 to make game harder
            "GRAVITY_X": UniformFloatContextFeature(
                "GRAVITY_X", lower=-20, upper=20, default_value=0
            ),
            "GRAVITY_Y": UniformFloatContextFeature(
                "GRAVITY_Y", lower=-20, upper=0.01, default_value=-10
            ),
            # lunar lander body specification
            "LEG_AWAY": UniformFloatContextFeature(
                "LEG_AWAY", lower=0, upper=50, default_value=20
            ),
            "LEG_DOWN": UniformFloatContextFeature(
                "LEG_DOWN", lower=0, upper=50, default_value=18
            ),
            "LEG_W": UniformFloatContextFeature(
                "LEG_W", lower=1, upper=10, default_value=2
            ),
            "LEG_H": UniformFloatContextFeature(
                "LEG_H", lower=1, upper=20, default_value=8
            ),
            "LEG_SPRING_TORQUE": UniformFloatContextFeature(
                "LEG_SPRING_TORQUE", lower=0, upper=100, default_value=40
            ),
            "SIDE_ENGINE_HEIGHT": UniformFloatContextFeature(
                "SIDE_ENGINE_HEIGHT", lower=1, upper=20, default_value=14.0
            ),
            "SIDE_ENGINE_AWAY": UniformFloatContextFeature(
                "SIDE_ENGINE_AWAY", lower=1, upper=20, default_value=12.0
            ),
            # Size of worl)d
            "VIEWPORT_W": UniformIntegerContextFeature(
                "VIEWPORT_W", lower=400, upper=1000, default_value=600
            ),
            "VIEWPORT_H": UniformIntegerContextFeature(
                "VIEWPORT_H", lower=200, upper=800, default_value=400
            ),
        }

    def _update_context(self) -> None:
        self.env: LunarLander
        for key, value in self.context.items():
            if hasattr(lunar_lander, key):
                setattr(lunar_lander, key, value)

        gravity_x = self.context.get(
            "GRAVITY_X", self.get_context_features()["GRAVITY_X"].default_value
        )
        gravity_y = self.context.get(
            "GRAVITY_Y", self.get_context_features()["GRAVITY_Y"].default_value
        )

        gravity = vec2(float(gravity_x), float(gravity_y))
        self.env.unwrapped.world.gravity = gravity
