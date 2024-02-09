from __future__ import annotations

from typing import Optional

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv


class CARLAcrobot(CARLGymnasiumEnv):
    env_name: str = "Acrobot-v1"
    metadata = {"render.modes": ["human", "rgb_array"]}

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "LINK_LENGTH_1": UniformFloatContextFeature(
                "LINK_LENGTH_1", lower=0.1, upper=10, default_value=1
            ),
            "LINK_LENGTH_2": UniformFloatContextFeature(
                "LINK_LENGTH_2", lower=0.1, upper=10, default_value=1
            ),
            "LINK_MASS_1": UniformFloatContextFeature(
                "LINK_MASS_1", lower=0.1, upper=10, default_value=1
            ),
            "LINK_MASS_2": UniformFloatContextFeature(
                "LINK_MASS_2", lower=0.1, upper=10, default_value=1
            ),
            "LINK_COM_POS_1": UniformFloatContextFeature(
                "LINK_COM_POS_1", lower=0, upper=1, default_value=0.5
            ),
            "LINK_COM_POS_2": UniformFloatContextFeature(
                "LINK_COM_POS_2", lower=0, upper=1, default_value=0.5
            ),
            "LINK_MOI": UniformFloatContextFeature(
                "LINK_MOI", lower=0.1, upper=10, default_value=1
            ),
            "MAX_VEL_1": UniformFloatContextFeature(
                "MAX_VEL_1",
                lower=0.4 * np.pi,
                upper=40 * np.pi,
                default_value=4 * np.pi,
            ),
            "MAX_VEL_2": UniformFloatContextFeature(
                "MAX_VEL_2",
                lower=0.9 * np.pi,
                upper=90 * np.pi,
                default_value=9 * np.pi,
            ),
            "torque_noise_max": UniformFloatContextFeature(
                "torque_noise_max", lower=-1, upper=1, default_value=0
            ),
            "INITIAL_ANGLE_LOWER": UniformFloatContextFeature(
                "INITIAL_ANGLE_LOWER", lower=-np.inf, upper=np.inf, default_value=-0.1
            ),
            "INITIAL_ANGLE_UPPER": UniformFloatContextFeature(
                "INITIAL_ANGLE_UPPER", lower=-np.inf, upper=np.inf, default_value=0.1
            ),
            "INITIAL_VELOCITY_LOWER": UniformFloatContextFeature(
                "INITIAL_VELOCITY_LOWER",
                lower=-np.inf,
                upper=np.inf,
                default_value=-0.1,
            ),
            "INITIAL_VELOCITY_UPPER": UniformFloatContextFeature(
                "INITIAL_VELOCITY_UPPER", lower=-np.inf, upper=np.inf, default_value=0.1
            ),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        angles = self.env.np_random.uniform(
            low=self.context.get(
                "INITIAL_ANGLE_LOWER",
                self.get_context_features()["INITIAL_ANGLE_LOWER"].default_value,
            ),
            high=self.context.get(
                "INITIAL_ANGLE_UPPER",
                self.get_context_features()["INITIAL_ANGLE_UPPER"].default_value,
            ),
            size=(2,),
        )
        velocities = self.env.np_random.uniform(
            low=self.context.get(
                "INITIAL_VELOCITY_LOWER",
                self.get_context_features()["INITIAL_VELOCITY_LOWER"].default_value,
            ),
            high=self.context.get(
                "INITIAL_VELOCITY_UPPER",
                self.get_context_features()["INITIAL_VELOCITY_UPPER"].default_value,
            ),
            size=(2,),
        )
        self.env.unwrapped.state = np.concatenate([angles, velocities])
        state = np.array(
            [
                np.cos(self.env.unwrapped.state[0]),
                np.sin(self.env.unwrapped.state[0]),
                np.cos(self.env.unwrapped.state[1]),
                np.sin(self.env.unwrapped.state[1]),
                self.env.unwrapped.state[2],
                self.env.unwrapped.state[3],
            ],
            dtype=np.float32,
        )
        info = {}
        state = self._add_context_to_state(state)
        info["context_id"] = self.context_id
        return state, info
