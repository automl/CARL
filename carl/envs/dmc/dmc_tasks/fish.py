# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Fish Domain."""

from typing import Dict, Optional, Tuple, Union

import collections

import dm_env  # type: ignore
import numpy as np
from dm_control import mujoco  # type: ignore
from dm_control.rl import control  # type: ignore
from dm_control.suite import base, common  # type: ignore
from dm_control.utils import containers, rewards  # type: ignore

from carl.envs.dmc.dmc_tasks.utils import adapt_context  # type: ignore
from carl.utils.types import Context

_DEFAULT_TIME_LIMIT = 40
_CONTROL_TIMESTEP = 0.04
STEP_LIMIT = 1000
_JOINTS = [
    "tail1",
    "tail_twist",
    "tail2",
    "finright_roll",
    "finright_pitch",
    "finleft_roll",
    "finleft_pitch",
]
SUITE = containers.TaggedTasks()


def get_model_and_assets() -> Tuple[bytes, Dict]:
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("fish.xml"), common.ASSETS


@SUITE.add("benchmarking")  # type: ignore
def upright_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the Fish Upright task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Upright(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        control_timestep=_CONTROL_TIMESTEP,
        time_limit=time_limit,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")  # type: ignore
def swim_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the Fish Swim task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Swim(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        control_timestep=_CONTROL_TIMESTEP,
        time_limit=time_limit,
        **environment_kwargs,
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Fish domain."""

    def upright(self) -> np.float64:
        """Returns projection from z-axes of torso to the z-axes of worldbody."""
        return self.named.data.xmat["torso", "zz"]

    def torso_velocity(self) -> np.ndarray:
        """Returns velocities and angular velocities of the torso."""
        return self.data.sensordata

    def joint_velocities(self) -> np.ndarray:
        """Returns the joint velocities."""
        return self.named.data.qvel[_JOINTS]

    def joint_angles(self) -> np.ndarray:
        """Returns the joint positions."""
        return self.named.data.qpos[_JOINTS]

    def mouth_to_target(self) -> np.ndarray:
        """Returns a vector, from mouth to target in local coordinate of mouth."""
        data = self.named.data
        mouth_to_target_global = data.geom_xpos["target"] - data.geom_xpos["mouth"]
        return mouth_to_target_global.dot(data.geom_xmat["mouth"].reshape(3, 3))


class Upright(base.Task):
    """A Fish `Task` for getting the torso upright with smooth reward."""

    def __init__(self, random: Union[np.random.RandomState, int, None] = None) -> None:
        """Initializes an instance of `Upright`.
        Args:
          random: Either an existing `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically.
        """
        super().__init__(random=random)

    def initialize_episode(self, physics: Physics) -> None:
        """Randomizes the tail and fin angles and the orientation of the Fish."""
        quat = self.random.randn(4)
        physics.named.data.qpos["root"][3:7] = quat / np.linalg.norm(quat)
        for joint in _JOINTS:
            physics.named.data.qpos[joint] = self.random.uniform(-0.2, 0.2)
        # Hide the target. It's irrelevant for this task.
        physics.named.model.geom_rgba["target", 3] = 0
        super().initialize_episode(physics)

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """Returns an observation of joint angles, velocities and uprightness."""
        obs = collections.OrderedDict()
        obs["joint_angles"] = physics.joint_angles()
        obs["upright"] = physics.upright()  # type: ignore
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics: Physics) -> float:
        """Returns a smooth reward."""
        return rewards.tolerance(physics.upright(), bounds=(1, 1), margin=1)


class Swim(base.Task):
    """A Fish `Task` for swimming with smooth reward."""

    def __init__(self, random: Union[np.random.RandomState, int, None] = None) -> None:
        """Initializes an instance of `Swim`.
        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(random=random)

    def initialize_episode(self, physics: Physics) -> None:
        """Sets the state of the environment at the start of each episode."""

        quat = self.random.randn(4)
        physics.named.data.qpos["root"][3:7] = quat / np.linalg.norm(quat)
        for joint in _JOINTS:
            physics.named.data.qpos[joint] = self.random.uniform(-0.2, 0.2)
        # Randomize target position.
        physics.named.model.geom_pos["target", "x"] = self.random.uniform(-0.4, 0.4)
        physics.named.model.geom_pos["target", "y"] = self.random.uniform(-0.4, 0.4)
        physics.named.model.geom_pos["target", "z"] = self.random.uniform(0.1, 0.3)
        super().initialize_episode(physics)

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """Returns an observation of joints, target direction and velocities."""
        obs = collections.OrderedDict()
        obs["joint_angles"] = physics.joint_angles()
        obs["upright"] = physics.upright()  # type: ignore
        obs["target"] = physics.mouth_to_target()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics: Physics) -> np.float64:
        """Returns a smooth reward."""
        radii = physics.named.model.geom_size[["mouth", "target"], 0].sum()
        in_target = rewards.tolerance(
            np.linalg.norm(physics.mouth_to_target()),
            bounds=(0, radii),
            margin=2 * radii,
        )
        is_upright = 0.5 * (physics.upright() + 1)
        return (7 * in_target + is_upright) / 8
