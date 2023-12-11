# Copyright 2019 The dm_control Authors.
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

"""Quadruped Domain."""

from typing import Any, Dict, List, Optional, Union

import collections
from collections import OrderedDict

import dm_env  # type: ignore
import numpy as np
from dm_control import mujoco  # type: ignore
from dm_control.mujoco.wrapper import mjbindings  # type: ignore
from dm_control.mujoco.wrapper.core import MjData  # type: ignore
from dm_control.rl import control  # type: ignore
from dm_control.suite import base, common  # type: ignore
from dm_control.utils import containers, rewards, xml_tools  # type: ignore
from lxml import etree  # type: ignore
from scipy import ndimage

from carl.envs.dmc.dmc_tasks.utils import adapt_context  # type: ignore
from carl.utils.types import Context

enums = mjbindings.enums
mjlib = mjbindings.mjlib


_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = 0.02
STEP_LIMIT = 1000

# Horizontal speeds above which the move reward is 1.
_RUN_SPEED = 5
_WALK_SPEED = 0.5

# Constants related to terrain generation.
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).

# Named model elements.
_TOES = ["toe_front_left", "toe_back_left", "toe_back_right", "toe_front_right"]
_WALLS = ["wall_px", "wall_py", "wall_nx", "wall_ny"]

SUITE = containers.TaggedTasks()


def make_model(
    floor_size: Optional[Any] = None,
    terrain: bool = False,
    rangefinders: bool = False,
    walls_and_ball: bool = False,
) -> bytes:
    """Returns the model XML string."""
    xml_string = common.read_model("quadruped.xml")
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    # Set floor size.
    if floor_size is not None:
        floor_geom = mjcf.find(".//geom[@name='floor']")
        floor_geom.attrib["size"] = f"{floor_size} {floor_size} .5"

    # Remove walls, ball and target.
    if not walls_and_ball:
        for wall in _WALLS:
            wall_geom = xml_tools.find_element(mjcf, "geom", wall)
            wall_geom.getparent().remove(wall_geom)

        # Remove ball.
        ball_body = xml_tools.find_element(mjcf, "body", "ball")
        ball_body.getparent().remove(ball_body)

        # Remove target.
        target_site = xml_tools.find_element(mjcf, "site", "target")
        target_site.getparent().remove(target_site)

    # Remove terrain.
    if not terrain:
        terrain_geom = xml_tools.find_element(mjcf, "geom", "terrain")
        terrain_geom.getparent().remove(terrain_geom)

    # Remove rangefinders if they're not used, as range computations can be
    # expensive, especially in a scene with heightfields.
    if not rangefinders:
        rangefinder_sensors = mjcf.findall(".//rangefinder")
        for rf in rangefinder_sensors:
            rf.getparent().remove(rf)

    return etree.tostring(mjcf, pretty_print=True)


@SUITE.add()  # type: ignore
def walk_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the Walk task with the adapted context."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Move(desired_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add()  # type: ignore
def run_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the Run task with the adapted context."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _RUN_SPEED)
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Move(desired_speed=_RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add()  # type: ignore
def escape_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the Escape task with the adapted context."""
    xml_string = make_model(floor_size=40, terrain=True, rangefinders=True)
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Escape(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add()  # type: ignore
def fetch_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the Fetch task with the adapted context."""
    xml_string = make_model(walls_and_ball=True)
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Fetch(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Quadruped domain."""

    def _reload_from_data(self, data: MjData) -> None:
        super()._reload_from_data(data)
        # Clear cached sensor names when the physics is reloaded.
        self._sensor_types_to_names: Dict = {}
        self._hinge_names: List = []

    def _get_sensor_names(self, *sensor_types: List[int]) -> List[str]:
        try:
            sensor_names = self._sensor_types_to_names[sensor_types]
        except KeyError:
            [sensor_ids] = np.where(np.in1d(self.model.sensor_type, sensor_types))
            sensor_names = [self.model.id2name(s_id, "sensor") for s_id in sensor_ids]
            self._sensor_types_to_names[sensor_types] = sensor_names
        return sensor_names

    def torso_upright(self) -> np.ndarray:
        """Returns the dot-product of the torso z-axis and the global z-axis."""
        return np.asarray(self.named.data.xmat["torso", "zz"])

    def torso_velocity(self) -> np.ndarray:
        """Returns the velocity of the torso, in the local frame."""
        return self.named.data.sensordata["velocimeter"].copy()

    def egocentric_state(self) -> np.ndarray:
        """Returns the state without global orientation or position."""
        if not self._hinge_names:
            [hinge_ids] = np.nonzero(self.model.jnt_type == enums.mjtJoint.mjJNT_HINGE)
            self._hinge_names = [
                self.model.id2name(j_id, "joint") for j_id in hinge_ids
            ]
        return np.hstack(
            (
                self.named.data.qpos[self._hinge_names],
                self.named.data.qvel[self._hinge_names],
                self.data.act,
            )
        )

    def toe_positions(self) -> np.ndarray:
        """Returns toe positions in egocentric frame."""
        torso_frame = self.named.data.xmat["torso"].reshape(3, 3)
        torso_pos = self.named.data.xpos["torso"]
        torso_to_toe = self.named.data.xpos[_TOES] - torso_pos
        return torso_to_toe.dot(torso_frame)

    def force_torque(self) -> np.ndarray:
        """Returns scaled force/torque sensor readings at the toes."""
        force_torque_sensors = self._get_sensor_names(
            enums.mjtSensor.mjSENS_FORCE, enums.mjtSensor.mjSENS_TORQUE
        )
        return np.arcsinh(self.named.data.sensordata[force_torque_sensors])

    def imu(self) -> np.ndarray:
        """Returns IMU-like sensor readings."""
        imu_sensors = self._get_sensor_names(
            enums.mjtSensor.mjSENS_GYRO, enums.mjtSensor.mjSENS_ACCELEROMETER
        )
        return self.named.data.sensordata[imu_sensors]

    def rangefinder(self) -> np.ndarray:
        """Returns scaled rangefinder sensor readings."""
        rf_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_RANGEFINDER)
        rf_readings = self.named.data.sensordata[rf_sensors]
        no_intersection = -1.0
        return np.where(rf_readings == no_intersection, 1.0, np.tanh(rf_readings))

    def origin_distance(self) -> np.ndarray:
        """Returns the distance from the origin to the workspace."""
        return np.asarray(np.linalg.norm(self.named.data.site_xpos["workspace"]))

    def origin(self) -> np.ndarray:
        """Returns origin position in the torso frame."""
        torso_frame = self.named.data.xmat["torso"].reshape(3, 3)
        torso_pos = self.named.data.xpos["torso"]
        return -torso_pos.dot(torso_frame)

    def ball_state(self) -> np.ndarray:
        """Returns ball position and velocity relative to the torso frame."""
        data = self.named.data
        torso_frame = data.xmat["torso"].reshape(3, 3)
        ball_rel_pos = data.xpos["ball"] - data.xpos["torso"]
        ball_rel_vel = data.qvel["ball_root"][:3] - data.qvel["root"][:3]
        ball_rot_vel = data.qvel["ball_root"][3:]
        ball_state = np.vstack((ball_rel_pos, ball_rel_vel, ball_rot_vel))
        return ball_state.dot(torso_frame).ravel()

    def target_position(self) -> np.ndarray:
        """Returns target position in torso frame."""
        torso_frame = self.named.data.xmat["torso"].reshape(3, 3)
        torso_pos = self.named.data.xpos["torso"]
        torso_to_target = self.named.data.site_xpos["target"] - torso_pos
        return torso_to_target.dot(torso_frame)

    def ball_to_target_distance(self) -> np.float64:
        """Returns horizontal distance from the ball to the target."""
        ball_to_target = (
            self.named.data.site_xpos["target"] - self.named.data.xpos["ball"]
        )
        return np.linalg.norm(ball_to_target[:2])

    def self_to_ball_distance(self) -> np.float64:
        """Returns horizontal distance from the quadruped workspace to the ball."""
        self_to_ball = (
            self.named.data.site_xpos["workspace"] - self.named.data.xpos["ball"]
        )
        return np.linalg.norm(self_to_ball[:2])


def _find_non_contacting_height(
    physics: Physics,
    orientation: np.ndarray,
    x_pos: float = 0.0,
    y_pos: float = 0.0,
) -> None:
    """Find a height with no contacts given a body orientation.
    Args:
      physics: An instance of `Physics`.
      orientation: A quaternion.
      x_pos: A float. Position along global x-axis.
      y_pos: A float. Position along global y-axis.
    Raises:
      RuntimeError: If a non-contacting configuration has not been found after
      10,000 attempts.
    """
    z_pos = 0.0  # Start embedded in the floor.
    num_contacts = 1
    num_attempts = 0
    # Move up in 1cm increments until no contacts.
    while num_contacts > 0:
        try:
            with physics.reset_context():
                physics.named.data.qpos["root"][:3] = x_pos, y_pos, z_pos
                physics.named.data.qpos["root"][3:] = orientation
        except control.PhysicsError:
            # We may encounter a PhysicsError here due to filling the contact
            # buffer, in which case we simply increment the height and continue.
            pass
        num_contacts = physics.data.ncon
        z_pos += 0.01
        num_attempts += 1
        if num_attempts > 10000:
            raise RuntimeError("Failed to find a non-contacting configuration.")


def _common_observations(physics: Physics) -> collections.OrderedDict:
    """Returns the observations common to all tasks."""
    obs = collections.OrderedDict()
    obs["egocentric_state"] = physics.egocentric_state()
    obs["torso_velocity"] = physics.torso_velocity()
    obs["torso_upright"] = physics.torso_upright()
    obs["imu"] = physics.imu()
    obs["force_torque"] = physics.force_torque()
    return obs


def _upright_reward(physics: Physics, deviation_angle: float = 0) -> np.float64:
    """Returns a reward proportional to how upright the torso is.
    Args:
      physics: an instance of `Physics`.
      deviation_angle: A float, in degrees. The reward is 0 when the torso is
        exactly upside-down and 1 when the torso's z-axis is less than
        `deviation_angle` away from the global z-axis.
    """
    deviation = np.cos(np.deg2rad(deviation_angle))
    return rewards.tolerance(
        physics.torso_upright(),
        bounds=(deviation, float("inf")),
        sigmoid="linear",
        margin=1 + deviation,
        value_at_margin=0,
    )


class Move(base.Task):
    """A quadruped task solved by moving forward at a designated speed."""

    def __init__(
        self,
        desired_speed: float,
        random: Union[np.random.RandomState, int, None] = None,
    ) -> None:
        """Initializes an instance of `Move`.
        Args:
          desired_speed: A float. If this value is zero, reward is given simply
            for standing upright. Otherwise this specifies the horizontal velocity
            at which the velocity-dependent reward component is maximized.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._desired_speed = desired_speed
        super().__init__(random=random)

    def initialize_episode(self, physics: Physics) -> None:
        """Sets the state of the environment at the start of each episode.
        Args:
          physics: An instance of `Physics`.
        """
        # Initial configuration.
        orientation = self.random.randn(4)
        orientation /= np.linalg.norm(orientation)
        _find_non_contacting_height(physics, orientation)
        super().initialize_episode(physics)

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """Returns an observation to the agent."""
        return _common_observations(physics)

    def get_reward(self, physics: Physics) -> np.float64:
        """Returns a reward to the agent."""

        # Move reward term.
        move_reward = rewards.tolerance(
            physics.torso_velocity()[0],
            bounds=(self._desired_speed, float("inf")),
            margin=self._desired_speed,
            value_at_margin=0.5,
            sigmoid="linear",
        )

        return _upright_reward(physics) * move_reward


class Escape(base.Task):
    """A quadruped task solved by escaping a bowl-shaped terrain."""

    def initialize_episode(self, physics: Physics) -> None:
        """Sets the state of the environment at the start of each episode.
        Args:
          physics: An instance of `Physics`.
        """
        # Get heightfield resolution, assert that it is square.
        res = physics.model.hfield_nrow[_HEIGHTFIELD_ID]
        assert res == physics.model.hfield_ncol[_HEIGHTFIELD_ID]
        # Sinusoidal bowl shape.
        row_grid, col_grid = np.ogrid[-1 : 1 : res * 1j, -1 : 1 : res * 1j]
        radius = np.clip(np.sqrt(col_grid**2 + row_grid**2), 0.04, 1)
        bowl_shape = 0.5 - np.cos(2 * np.pi * radius) / 2
        # Random smooth bumps.
        terrain_size = 2 * physics.model.hfield_size[_HEIGHTFIELD_ID, 0]
        bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)
        bumps = self.random.uniform(_TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))
        smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
        # Terrain is elementwise product.
        terrain = bowl_shape * smooth_bumps
        start_idx = physics.model.hfield_adr[_HEIGHTFIELD_ID]
        physics.model.hfield_data[start_idx : start_idx + res**2] = terrain.ravel()
        super().initialize_episode(physics)

        # If we have a rendering context, we need to re-upload the modified
        # heightfield data.
        if physics.contexts:
            with physics.contexts.gl.make_current() as ctx:
                ctx.call(
                    mjlib.mjr_uploadHField,
                    physics.model.ptr,
                    physics.contexts.mujoco.ptr,
                    _HEIGHTFIELD_ID,
                )

        # Initial configuration.
        orientation = self.random.randn(4)
        orientation /= np.linalg.norm(orientation)
        _find_non_contacting_height(physics, orientation)

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """Returns an observation to the agent."""
        obs = _common_observations(physics)
        obs["origin"] = physics.origin()
        obs["rangefinder"] = physics.rangefinder()
        return obs

    def get_reward(self, physics: Physics) -> np.float64:
        """Returns a reward to the agent."""

        # Escape reward term.
        terrain_size = physics.model.hfield_size[_HEIGHTFIELD_ID, 0]
        escape_reward = rewards.tolerance(
            physics.origin_distance(),
            bounds=(terrain_size, float("inf")),
            margin=terrain_size,
            value_at_margin=0,
            sigmoid="linear",
        )

        return _upright_reward(physics, deviation_angle=20) * escape_reward


class Fetch(base.Task):
    """A quadruped task solved by bringing a ball to the origin."""

    def initialize_episode(self, physics: Physics) -> None:
        """Sets the state of the environment at the start of each episode.
        Args:
          physics: An instance of `Physics`.
        """
        # Initial configuration, random azimuth and horizontal position.
        azimuth = self.random.uniform(0, 2 * np.pi)
        orientation = np.array((np.cos(azimuth / 2), 0, 0, np.sin(azimuth / 2)))
        spawn_radius = 0.9 * physics.named.model.geom_size["floor", 0]
        x_pos, y_pos = self.random.uniform(-spawn_radius, spawn_radius, size=(2,))
        _find_non_contacting_height(physics, orientation, x_pos, y_pos)

        # Initial ball state.
        physics.named.data.qpos["ball_root"][:2] = self.random.uniform(
            -spawn_radius, spawn_radius, size=(2,)
        )
        physics.named.data.qpos["ball_root"][2] = 2
        physics.named.data.qvel["ball_root"][:2] = 5 * self.random.randn(2)
        super().initialize_episode(physics)

    def get_observation(self, physics: Physics) -> OrderedDict:
        """Returns an observation to the agent."""
        obs = _common_observations(physics)
        obs["ball_state"] = physics.ball_state()
        obs["target_position"] = physics.target_position()
        return obs

    def get_reward(self, physics: Physics) -> np.float64:
        """Returns a reward to the agent."""

        # Reward for moving close to the ball.
        arena_radius = physics.named.model.geom_size["floor", 0] * np.sqrt(2)
        workspace_radius = physics.named.model.site_size["workspace", 0]
        ball_radius = physics.named.model.geom_size["ball", 0]
        reach_reward = rewards.tolerance(
            physics.self_to_ball_distance(),
            bounds=(0, workspace_radius + ball_radius),
            sigmoid="linear",
            margin=arena_radius,
            value_at_margin=0,
        )

        # Reward for bringing the ball to the target.
        target_radius = physics.named.model.site_size["target", 0]
        fetch_reward = rewards.tolerance(
            physics.ball_to_target_distance(),
            bounds=(0, target_radius),
            sigmoid="linear",
            margin=arena_radius,
            value_at_margin=0,
        )

        reach_then_fetch = reach_reward * (0.5 + 0.5 * fetch_reward)

        return _upright_reward(physics) * reach_then_fetch
