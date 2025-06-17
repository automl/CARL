# Updated version of the brax gym wrappers to fix rendering issues.
# Hopefully we can go back to the original version with v1 of brax.
#
# The original copyright:
# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrappers to convert brax envs to gym envs."""

from typing import ClassVar, Optional

import gymnasium
import gymnasium as gym
import jax
import numpy as np
from brax.envs.base import PipelineEnv
from brax.io import image
from gymnasium import spaces, vector


class BraxGymWrapper(gym.Env):
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: PipelineEnv, seed: int = 0, backend: Optional[str] = None):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }
        self.seed(seed)
        self.backend = backend
        self._state = None

        # Set up observation space.
        obs = np.inf * np.ones(self._env.observation_size, dtype="float32")
        self.observation_space = spaces.Box(-obs, obs, dtype="float32")

        # Set up action space.
        action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
        self.action_space = spaces.Box(action[:, 0], action[:, 1], dtype="float32")

        # Modified reset function now accepts seed and options.
        def reset(key, seed=None, options=None):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        # The step function remains unchanged.
        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._state, obs, self._key = self._reset(self._key, seed=seed, options=options)
        info = {}
        return obs, info

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode="human"):
        if mode == "rgb_array":
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError("Call reset or step before rendering.")
            return image.render_array(sys, state.pipeline_state, 256, 256)
        else:
            return super().render(mode=mode)


class VectorGymWrapper(gym.vector.VectorEnv):
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: PipelineEnv, seed: int = 0, backend: Optional[str] = None):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }
        if not hasattr(self._env, "batch_size"):
            raise ValueError("The underlying env must be batched.")

        self.num_envs = self._env.batch_size
        self.seed(seed)
        self.backend = backend
        self._state = None

        # Set up batched observation space.
        obs = np.inf * np.ones(self._env.observation_size, dtype="float32")
        obs_space = spaces.Box(-obs, obs, dtype="float32")
        self.observation_space = vector.utils.batch_space(obs_space, self.num_envs)

        # Set up batched action space.
        action = jax.tree_map(np.array, self._env.sys.actuator.ctrl_range)
        action_space = spaces.Box(action[:, 0], action[:, 1], dtype="float32")
        self.action_space = vector.utils.batch_space(action_space, self.num_envs)

        # Modified reset function to accept extra kwargs.
        def reset(key, seed=None, options=None):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        # Step function remains as before.
        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._state, obs, self._key = self._reset(self._key, seed=seed, options=options)
        info = {}
        return obs, info

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode="human"):
        if mode == "rgb_array":
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError("Call reset or step before rendering.")
            # Render the first instance in the batch.
            return image.render_array(sys, state.pipeline_state.take(0), 256, 256)
        else:
            return super().render(mode=mode)


class GymWrapper(gymnasium.Env):
    """Wrapper that converts Brax env to be Gymnasium-compatible"""

    def __init__(self, env):
        self._env = BraxGymWrapper(env)

        # Convert spaces to gymnasium spaces
        self.observation_space = spaces.Box(
            low=self._env.observation_space.low,
            high=self._env.observation_space.high,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=self._env.action_space.low,
            high=self._env.action_space.high,
            dtype=np.float32,
        )

        # Forward attributes from wrapped env
        # import pdb; pdb.set_trace()
        # self.unwrapped = self._env
        self.metadata = self._env.metadata if hasattr(self._env, "metadata") else {}

    @property
    def unwrapped(self):
        """Return the base environment without any wrappers."""
        return self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env

    def reset(self, *args, **kwargs):
        # import pdb; pdb.set_trace()
        return self._env.reset(*args, **kwargs)  #

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    @property
    def context(self):
        return self._env.context if hasattr(self._env, "context") else None

    @context.setter
    def context(self, context):
        self._env.context = context
