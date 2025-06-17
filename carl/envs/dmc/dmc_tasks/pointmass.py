# flake8: noqa: E501
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

"""Finger Domain."""

from __future__ import annotations

from typing import Any

import numpy as np
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control  # type: ignore
from dm_control.suite.point_mass import (  # type: ignore
    _DEFAULT_TIME_LIMIT,
    SUITE,
    Physics,
    PointMass,
    get_model_and_assets,
)

from carl.envs.dmc.dmc_tasks.utils import adapt_context  # type: ignore
from carl.utils.types import Context


def check_constraints(
    mass,
    starting_x,
    starting_y,
    target_x,
    target_y,
    area_size,
) -> None:
    if (
        starting_x >= area_size / 4
        or starting_y >= area_size / 4
        or starting_x <= -area_size / 4
        or starting_y <= -area_size / 4
    ):
        raise ValueError(
            f"The starting points are located outside of the grid. Choose a value lower than {area_size / 4}."
        )

    if (
        target_x >= area_size / 4
        or target_y >= area_size / 4
        or target_x <= -area_size / 4
        or target_y <= -area_size / 4
    ):
        raise ValueError(
            f"The target points are located outside of the grid. Choose a value lower than {area_size / 4}."
        )


def make_model(
    mass: float = 0.3,
    starting_x: float = 0.0,
    starting_y: float = 0.0,
    target_x: float = 0.0,
    target_y: float = 0.0,
    area_size: float = 0.6,
    **kwargs: Any,
) -> bytes:
    check_constraints(
        mass=mass,
        starting_x=starting_x,
        starting_y=starting_y,
        target_x=target_x,
        target_y=target_y,
        area_size=area_size,
    )

    xml_string = f"""
     <mujoco model="planar point mass">
    <include file="./common/skybox.xml"/>
    <include file="./common/visual.xml"/>
    <include file="./common/materials.xml"/>

    <option timestep="0.02">
        <flag contact="disable"/>
    </option>

    <default>
        <joint type="hinge" axis="0 0 1" limited="true" range="-.29 .29" damping="1"/>
        <motor gear=".1" ctrlrange="-1 1" ctrllimited="true"/>
    </default>

    <worldbody>
        <light name="light" pos="0 0 1"/>
        <camera name="fixed" pos="0 0 .75" quat="1 0 0 0"/>
        <geom name="ground" type="plane" pos="0 0 0" size="{area_size / 2} {area_size / 2} .1" material="grid"/>
        <geom name="wall_x" type="plane" pos="{-area_size / 2} 0 .02" zaxis="1 0 0"  size=".02 .3 .02" material="decoration"/>
        <geom name="wall_y" type="plane" pos="0 {-area_size / 2} .02" zaxis="0 1 0"  size=".3 .02 .02" material="decoration"/>
        <geom name="wall_neg_x" type="plane" pos="{area_size / 2} 0 .02" zaxis="-1 0 0"  size=".02 .3 .02" material="decoration"/>
        <geom name="wall_neg_y" type="plane" pos="0 {area_size / 2} .02" zaxis="0 -1 0"  size=".3 .02 .02" material="decoration"/>

        <body name="pointmass" pos="{starting_x} {starting_y} .01">
        <camera name="cam0" pos="0 -0.3 0.3" xyaxes="1 0 0 0 0.7 0.7"/>
        <joint name="root_x" type="slide"  pos="{starting_x} {starting_y} 0" axis="1 0 0" />
        <joint name="root_y" type="slide"  pos="{starting_x} {starting_y} 0" axis="0 1 0" />
        <geom name="pointmass" type="sphere" size=".01" material="self" mass="{mass}"/>
        </body>

        <geom name="target" pos="{target_x} {target_y} .01" material="target" type="sphere" size=".015"/>
    </worldbody>

    <tendon>
        <fixed name="t1">
        <joint joint="root_x" coef="1"/>
        <joint joint="root_y" coef="0"/>
        </fixed>
        <fixed name="t2">
        <joint joint="root_x" coef="0"/>
        <joint joint="root_y" coef="1"/>
        </fixed>
    </tendon>

    <actuator>
        <motor name="t1" tendon="t1"/>
        <motor name="t2" tendon="t2"/>
    </actuator>
    </mujoco>
    """
    xml_string_bytes = xml_string.encode()
    return xml_string_bytes


def random_limited_quaternion(random, limit):
    """Generates a random quaternion limited to the specified rotations."""
    axis = random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = random.rand() * limit

    quaternion = np.zeros(4)
    mjbindings.mjlib.mju_axisAngle2Quat(quaternion, axis, angle)

    return quaternion


class ContextualPointMass(PointMass):
    starting_x: float = 0.2
    starting_y: float = 0.2

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        If _randomize_gains is True, the relationship between the controls and
        the joints is randomized, so that each control actuates a random linear
        combination of joints.

        Args:
        physics: An instance of `mujoco.Physics`.
        """
        if self._randomize_gains:
            dir1 = self.random.randn(2)
            dir1 /= np.linalg.norm(dir1)
            # Find another actuation direction that is not 'too parallel' to dir1.
            parallel = True
            while parallel:
                dir2 = self.random.randn(2)
                dir2 /= np.linalg.norm(dir2)
                parallel = abs(np.dot(dir1, dir2)) > 0.9
            physics.model.wrap_prm[[0, 1]] = dir1
            physics.model.wrap_prm[[2, 3]] = dir2
        super().initialize_episode(physics)
        self.randomize_limited_and_rotational_joints(physics, self.random)

    def randomize_limited_and_rotational_joints(self, physics, random=None):
        random = random or np.random

        hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
        slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
        ball = mjbindings.enums.mjtJoint.mjJNT_BALL
        free = mjbindings.enums.mjtJoint.mjJNT_FREE

        qpos = physics.named.data.qpos

        for joint_id in range(physics.model.njnt):
            joint_name = physics.model.id2name(joint_id, "joint")
            joint_type = physics.model.jnt_type[joint_id]
            is_limited = physics.model.jnt_limited[joint_id]
            range_min, range_max = physics.model.jnt_range[joint_id]

            if is_limited:
                if joint_type == hinge or joint_type == slide:
                    if "root_x" in joint_name:
                        qpos[joint_name] = self.starting_x
                    elif "root_y" in joint_name:
                        qpos[joint_name] = self.starting_y
                    else:
                        qpos[joint_name] = random.uniform(range_min, range_max)

                elif joint_type == ball:
                    qpos[joint_name] = random_limited_quaternion(random, range_max)

            else:
                if joint_type == hinge:
                    qpos[joint_name] = random.uniform(-np.pi, np.pi)

                elif joint_type == ball:
                    quat = random.randn(4)
                    quat /= np.linalg.norm(quat)
                    qpos[joint_name] = quat

                elif joint_type == free:
                    # this should be random.randn, but changing it now could significantly
                    # affect benchmark results.
                    quat = random.rand(4)
                    quat /= np.linalg.norm(quat)
                    qpos[joint_name][3:] = quat


@SUITE.add("benchmarking")  # type: ignore[misc]
def easy_pointmass(
    context: Context = {},
    context_mask: list = [],
    time_limit: float = _DEFAULT_TIME_LIMIT,
    random: np.random.RandomState | int | None = None,
    environment_kwargs: dict | None = None,
) -> control.Environment:
    """No randomization."""
    xml_string, assets = get_model_and_assets()
    xml_string = make_model(**context)
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = ContextualPointMass(randomize_gains=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")  # type: ignore[misc]
def hard_pointmass(
    context: Context = {},
    context_mask: list = [],
    time_limit: float = _DEFAULT_TIME_LIMIT,
    random: np.random.RandomState | int | None = None,
    environment_kwargs: dict | None = None,
) -> control.Environment:
    """Randomized initializations."""
    xml_string, assets = get_model_and_assets()
    xml_string = make_model(**context)
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = ContextualPointMass(randomize_gains=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        **environment_kwargs,
    )
