from __future__ import annotations

import gymnasium
import gymnax
from gymnasium.wrappers import EnvCompatibility
from gymnax.environments.spaces import Space, gymnax_space_to_gym_space
from gymnax.wrappers.gym import GymnaxToGymWrapper


class CustomGymnaxToGymWrapper(GymnaxToGymWrapper):
    @property
    def observation_space(self) -> dict:
        return gymnax_space_to_gym_space(self._env.observation_space(self.env_params))

    @observation_space.setter
    def observation_space(self, value: Space) -> None:
        self._observation_space = value


def make_gymnax_env(env_name: str) -> gymnasium.Env:
    # Make gymnax env
    env, env_params = gymnax.make(env_id=env_name)

    # Convert gymnax to gym API
    env = CustomGymnaxToGymWrapper(env=env, params=env_params)

    # Convert gym to gymnasium API
    env = EnvCompatibility(old_env=env)

    return env
