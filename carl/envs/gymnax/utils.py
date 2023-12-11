from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium
import gymnasium.spaces
import gymnax
from gymnasium.wrappers import EnvCompatibility
from gymnasium.wrappers.compatibility import LegacyEnv
from gymnax.environments.environment import Environment, EnvParams
from gymnax.environments.spaces import Space, gymnax_space_to_gym_space
from gymnax.wrappers.gym import GymnaxToGymWrapper
from numpy._typing import DTypeLike
from numpy.random._generator import Generator as Generator


# Although this converts to gym, the step API already is for gymnasium
class CustomGymnaxToGymnasiumWrapper(GymnaxToGymWrapper):
    def __init__(
        self, env: Environment, params: EnvParams | None = None, seed: int | None = None
    ):
        super().__init__(env, params, seed)

        self._observation_space = SpaceWrapper(
            gymnax_space_to_gym_space(self._env.observation_space(self.env_params))
        )

    @property
    def env(self) -> Environment:
        return self._env

    @env.setter
    def env(self, value: Environment) -> None:
        self._env = value

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: Space) -> None:
        self._observation_space = value


class SpaceWrapper(gymnasium.Space):
    def __init__(self, space):
        self.space = space

    def __getattr__(self, __name: str) -> Any:
        return self.space.__getattr__(__name=__name)


def make_gymnax_env(env_name: str) -> gymnasium.Env:
    # Make gymnax env
    env, env_params = gymnax.make(env_id=env_name)

    # Convert gymnax to gymnasium API
    env = CustomGymnaxToGymnasiumWrapper(env=env, params=env_params)

    return env
