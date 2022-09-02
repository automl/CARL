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


@SUITE.add("benchmarking")  # type: ignore[misc]
def spin_context(
    context: Context = {},
    context_mask: list = [],
    time_limit: float = _DEFAULT_TIME_LIMIT,
    random: np.random.RandomState | int | None = None,
    environment_kwargs: dict | None = None,
) -> control.Environment:
    """Returns the Spin task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(
            xml_string=xml_string, context=context, context_mask=context_mask
        )
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
    context_mask: list = [],
    time_limit: float = _DEFAULT_TIME_LIMIT,
    random: np.random.RandomState | int | None = None,
    environment_kwargs: dict | None = None,
) -> control.Environment:
    """Returns the easy Turn task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(
            xml_string=xml_string, context=context, context_mask=context_mask
        )
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
    context_mask: list = [],
    time_limit: float = _DEFAULT_TIME_LIMIT,
    random: np.random.RandomState | int | None = None,
    environment_kwargs: dict | None = None,
) -> control.Environment:
    """Returns the hard Turn task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(
            xml_string=xml_string, context=context, context_mask=context_mask
        )
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
