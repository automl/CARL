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
from dm_control.rl import control  # type: ignore
from dm_control.suite.finger import (  # type: ignore
    _CONTROL_TIMESTEP,
    _DEFAULT_TIME_LIMIT,
    _EASY_TARGET_SIZE,
    _HARD_TARGET_SIZE,
    SUITE,
    Physics,
    Spin,
    Turn,
    get_model_and_assets,
)

from carl.envs.dmc.dmc_tasks.utils import adapt_context  # type: ignore
from carl.utils.types import Context


def check_constraints(
    spinner_length: float,
    limb_length_0: float,
    limb_length_1: float,
    x_spinner: float = 0.2,
    x_finger: float = -0.2,
    raise_error: bool = False,
    **kwargs: Any,
) -> bool:
    is_okay = True
    spinner_half_length = spinner_length / 2
    # Check if spinner collides with finger hinge
    distance_spinner_to_fingerhinge = (x_spinner - x_finger) - spinner_half_length
    if distance_spinner_to_fingerhinge < 0:
        is_okay = False
        if raise_error:
            raise ValueError(
                f"Distance finger to spinner ({distance_spinner_to_fingerhinge}) not big enough, "
                f"spinner can't spin. Decrease spinner_length ({spinner_length})."
            )
        is_okay = False
        if raise_error:
            raise ValueError(
                f"Distance finger to spinner ({distance_spinner_to_fingerhinge}) not big enough, "
                f"spinner can't spin. Decrease spinner_length ({spinner_length})."
            )

    # Check if finger can reach spinner (distance should be negative)
    distance_fingertip_to_spinner = (x_spinner - spinner_half_length) - (
        x_finger + limb_length_0 + limb_length_1
    )
    if distance_fingertip_to_spinner > 0:
        is_okay = False
        if raise_error:
            raise ValueError(
                f"Finger cannot reach spinner ({distance_fingertip_to_spinner}). Increase either "
                f"limb_length_0, limb_length_1 or spinner_length."
            )
        is_okay = False
        if raise_error:
            raise ValueError(
                f"Finger cannot reach spinner ({distance_fingertip_to_spinner}). Increase either "
                f"limb_length_0, limb_length_1 or spinner_length."
            )

    return is_okay

    return is_okay


def get_finger_xml_string(
    limb_length_0: float = 0.17,
    limb_length_1: float = 0.16,
    spinner_radius: float = 0.04,
    spinner_length: float = 0.18,
    **kwargs: Any,
) -> bytes:
    # Finger position
    x_finger = -0.2
    y_finger = 0.4

    # Spinner position
    x_spinner = 0.2
    y_spinner = 0.4

    # Target position
    y_target = 0.4

    # Spinner geometry
    spinner_half_length = spinner_length / 2
    spinner_tip_radius = 0.02
    distance_spinner_tip_to_captop = 0.06
    y_spinner_tip = (
        spinner_half_length + distance_spinner_tip_to_captop - spinner_tip_radius
    )  # originally 0.13

    check_constraints(
        limb_length_0=limb_length_0,
        limb_length_1=limb_length_1,
        x_spinner=x_spinner,
        x_finger=x_finger,
        spinner_length=spinner_length,
        raise_error=True,
    )

    proximal_to = -limb_length_0
    xml_string = f"""
     <mujoco model="finger">
      <include file="./common/visual.xml"/>
      <include file="./common/skybox.xml"/>
      <include file="./common/materials.xml"/>

      <option timestep="0.01" cone="elliptic" iterations="200">
        <flag gravity="disable"/>
      </option>

      <default>
        <geom solimp="0 0.9 0.01" solref=".02 1"/>
        <joint type="hinge" axis="0 -1 0"/>
        <motor ctrllimited="true" ctrlrange="-1 1"/>
        <default class="finger">
          <joint damping="2.5" limited="true"/>
          <site type="ellipsoid" size=".025 .03 .025" material="site" group="3"/>
        </default>
      </default>

      <worldbody>
        <light name="light" directional="true" diffuse=".6 .6 .6" pos="0 0 2" specular=".3 .3 .3"/>
        <geom name="ground" type="plane" pos="0 0 0" size=".6 .2 10" material="grid"/>
        <camera name="cam0" pos="0 -1 .8" xyaxes="1 0 0 0 1 2"/>
        <camera name="cam1" pos="0 -1 .4" xyaxes="1 0 0 0 0 1" />

        <body name="proximal" pos="{x_finger} 0 {y_finger}" childclass="finger">
          <geom name="proximal_decoration" type="cylinder" fromto="0 -.033 0 0 .033 0" size=".034" material="decoration"/>
          <joint name="proximal" range="-110 110" ref="-90"/>
          <geom name="proximal" type="capsule" material="self" size=".03" fromto="0 0 0 0 0 {proximal_to}"/>
          <body name="distal" pos="0 0 {proximal_to - 0.01}" childclass="finger">
            <joint name="distal" range="-110 110"/>
            <geom name="distal" type="capsule" size=".028" material="self" fromto="0 0 0 0 0 {-limb_length_1}" contype="0" conaffinity="0"/>
            <geom name="fingertip" type="capsule" size=".03" material="effector" fromto="0 0 {-limb_length_1 - 0.03} 0 0 {-limb_length_1 - 0.001}"/>
            <site name="touchtop" pos=".01 0 -.17"/>
            <site name="touchbottom" pos="-.01 0 -.17"/>
          </body>
        </body>

        <body name="spinner" pos="{x_spinner} 0 {y_spinner}">
          <joint name="hinge" frictionloss=".1" damping=".5"/>
          <geom name="cap1" type="capsule" size="{spinner_radius}" fromto="{spinner_radius / 2} 0 {-spinner_half_length} {spinner_radius} 0 {spinner_half_length}" material="self"/>
          <geom name="cap2" type="capsule" size="{spinner_radius}" fromto="{-spinner_radius / 2} 0 {-spinner_half_length} 0 0 {spinner_half_length}" material="self"/>
          <site name="tip" type="sphere"  size="{spinner_tip_radius}" pos="0 0 {y_spinner_tip}" material="target"/>
          <geom name="spinner_decoration" type="cylinder" fromto="0 -.045 0 0 .045 0" size="{spinner_radius / 2}" material="decoration"/>
        </body>

        <site name="target" type="sphere" size=".03" pos="0 0 {y_target}" material="target"/>
      </worldbody>

      <actuator>
        <motor name="proximal" joint="proximal" gear="30"/>
        <motor name="distal" joint="distal" gear="15"/>
      </actuator>

      <!-- All finger observations are functions of sensors. This is useful for finite-differencing. -->
      <sensor>
        <jointpos name="proximal" joint="proximal"/>
        <jointpos name="distal" joint="distal"/>
        <jointvel name="proximal_velocity" joint="proximal"/>
        <jointvel name="distal_velocity" joint="distal"/>
        <jointvel name="hinge_velocity" joint="hinge"/>
        <framepos name="tip" objtype="site" objname="tip"/>
        <framepos name="target" objtype="site" objname="target"/>
        <framepos name="spinner" objtype="xbody" objname="spinner"/>
        <touch name="touchtop" site="touchtop"/>
        <touch name="touchbottom" site="touchbottom"/>
        <framepos name="touchtop_pos" objtype="site" objname="touchtop"/>
        <framepos name="touchbottom_pos" objtype="site" objname="touchbottom"/>
      </sensor>

    </mujoco>
    """
    xml_string_bytes = xml_string.encode()
    return xml_string_bytes


@SUITE.add("benchmarking")  # type: ignore[misc]
def spin_context(
    context: Context = {},
    time_limit: float = _DEFAULT_TIME_LIMIT,
    random: np.random.RandomState | int | None = None,
    environment_kwargs: dict | None = None,
) -> control.Environment:
    """Returns the Spin task."""
    xml_string, assets = get_model_and_assets()
    xml_string = get_finger_xml_string(**context)
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Spin(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")  # type: ignore[misc]
def turn_easy_context(
    context: Context = {},
    time_limit: float = _DEFAULT_TIME_LIMIT,
    random: np.random.RandomState | int | None = None,
    environment_kwargs: dict | None = None,
) -> control.Environment:
    """Returns the easy Turn task."""
    xml_string, assets = get_model_and_assets()
    xml_string = get_finger_xml_string(**context)
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Turn(target_radius=_EASY_TARGET_SIZE, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")  # type: ignore[misc]
def turn_hard_context(
    context: Context = {},
    time_limit: float = _DEFAULT_TIME_LIMIT,
    random: np.random.RandomState | int | None = None,
    environment_kwargs: dict | None = None,
) -> control.Environment:
    """Returns the hard Turn task."""
    xml_string, assets = get_model_and_assets()
    xml_string = get_finger_xml_string(**context)
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Turn(target_radius=_HARD_TARGET_SIZE, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )
