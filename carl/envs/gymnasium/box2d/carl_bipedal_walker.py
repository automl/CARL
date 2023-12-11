from __future__ import annotations

from Box2D.b2 import edgeShape, fixtureDef, polygonShape
from gymnasium.envs.box2d import bipedal_walker
from gymnasium.envs.box2d import bipedal_walker as bpw

from carl.context.context_space import (
    ContextFeature,
    UniformFloatContextFeature,
    UniformIntegerContextFeature,
)
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv


class CARLBipedalWalker(CARLGymnasiumEnv):
    env_name: str = "BipedalWalker-v3"
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
            "GRAVITY_X": UniformFloatContextFeature(
                "GRAVITY_X", lower=-20, upper=20, default_value=0
            ),
            "GRAVITY_Y": UniformFloatContextFeature(
                "GRAVITY_Y", lower=-20, upper=-0.01, default_value=-10
            ),
            # surroundings
            "FRICTION": UniformFloatContextFeature(
                "FRICTION", lower=0, upper=10, default_value=2.5
            ),
            "TERRAIN_STEP": UniformFloatContextFeature(
                "TERRAIN_STEP", lower=0.25, upper=1, default_value=14 / 30.0
            ),
            "TERRAIN_LENGTH": UniformIntegerContextFeature(
                "TERRAIN_LENGTH", lower=100, upper=500, default_value=200
            ),  # in steps
            "TERRAIN_HEIGHT": UniformFloatContextFeature(
                "TERRAIN_HEIGHT", lower=3, upper=10, default_value=600 / 30 / 4
            ),  # VIEWPORT_H/SCALE/4
            "TERRAIN_GRASS": UniformIntegerContextFeature(
                "TERRAIN_GRASS", lower=5, upper=15, default_value=10
            ),  # low long are grass spots, in step)s
            "TERRAIN_STARTPAD": UniformFloatContextFeature(
                "TERRAIN_STARTPAD", lower=10, upper=30, default_value=20
            ),  # in steps
            # walker
            "MOTORS_TORQUE": UniformFloatContextFeature(
                "MOTORS_TORQUE", lower=0.01, upper=200, default_value=80
            ),
            "SPEED_HIP": UniformFloatContextFeature(
                "SPEED_HIP", lower=0.01, upper=15, default_value=4
            ),
            "SPEED_KNEE": UniformFloatContextFeature(
                "SPEED_KNEE", lower=0.01, upper=15, default_value=6
            ),
            "LIDAR_RANGE": UniformFloatContextFeature(
                "LIDAR_RANGE", lower=0.01, upper=20, default_value=160 / 30.0
            ),
            "LEG_DOWN": UniformFloatContextFeature(
                "LEG_DOWN", lower=-2, upper=-0.25, default_value=-8 / 30.0
            ),
            "LEG_W": UniformFloatContextFeature(
                "LEG_W", lower=0.25, upper=0.5, default_value=8 / 30.0
            ),
            "LEG_H": UniformFloatContextFeature(
                "LEG_H", lower=0.25, upper=2, default_value=34 / 30.0
            ),
            # absolute value of random force applied to walker at start of episode
            "INITIAL_RANDOM": UniformFloatContextFeature(
                "INITIAL_RANDOM", lower=0, upper=50, default_value=5
            ),
            # Size of world
            "VIEWPORT_W": UniformIntegerContextFeature(
                "VIEWPORT_W", lower=400, upper=1000, default_value=600
            ),
            "VIEWPORT_H": UniformIntegerContextFeature(
                "VIEWPORT_H", lower=200, upper=800, default_value=400
            ),
        }

    def _update_context(self) -> None:
        self.env: bipedal_walker.BipedalWalker
        self.context = CARLBipedalWalker.get_context_space().insert_defaults(
            self.context
        )
        bpw.FPS = self.context["FPS"]
        bpw.SCALE = self.context["SCALE"]
        bpw.FRICTION = self.context["FRICTION"]
        bpw.TERRAIN_STEP = self.context["TERRAIN_STEP"]
        bpw.TERRAIN_LENGTH = int(
            self.context["TERRAIN_LENGTH"]
        )  # TODO do this automatically
        bpw.TERRAIN_HEIGHT = self.context["TERRAIN_HEIGHT"]
        bpw.TERRAIN_GRASS = self.context["TERRAIN_GRASS"]
        bpw.TERRAIN_STARTPAD = self.context["TERRAIN_STARTPAD"]
        bpw.MOTORS_TORQUE = self.context["MOTORS_TORQUE"]
        bpw.SPEED_HIP = self.context["SPEED_HIP"]
        bpw.SPEED_KNEE = self.context["SPEED_KNEE"]
        bpw.LIDAR_RANGE = self.context["LIDAR_RANGE"]
        bpw.LEG_DOWN = self.context["LEG_DOWN"]
        bpw.LEG_W = self.context["LEG_W"]
        bpw.LEG_H = self.context["LEG_H"]
        bpw.INITIAL_RANDOM = self.context["INITIAL_RANDOM"]
        bpw.VIEWPORT_W = self.context["VIEWPORT_W"]
        bpw.VIEWPORT_H = self.context["VIEWPORT_H"]

        gravity_x = self.context["GRAVITY_X"]
        gravity_y = self.context["GRAVITY_Y"]

        gravity = (gravity_x, gravity_y)
        self.env.unwrapped.world.gravity = gravity

        # Important for building terrain
        self.env.unwrapped.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=bipedal_walker.FRICTION,
        )
        self.env.unwrapped.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=bipedal_walker.FRICTION,
            categoryBits=0x0001,
        )

        bpw.HULL_FD = fixtureDef(
            shape=polygonShape(
                vertices=[(x / bpw.SCALE, y / bpw.SCALE) for x, y in bpw.HULL_POLY]
            ),
            density=5.0,
            friction=0.1,
            categoryBits=0x0020,
            maskBits=0x001,  # collide only with ground
            restitution=0.0,
        )  # 0.99 bouncy

        bpw.LEG_FD = fixtureDef(
            shape=polygonShape(box=(bpw.LEG_W / 2, bpw.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001,
        )

        bpw.LOWER_FD = fixtureDef(
            shape=polygonShape(box=(0.8 * bpw.LEG_W / 2, bpw.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001,
        )

        self.env.unwrapped.world.gravity = gravity
