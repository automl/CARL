from typing import Any, Dict, List, Literal, cast

import os
import random
import socket
from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from py4j.java_gateway import GatewayParameters, JavaGateway

from carl.envs.mario.level_image_gen import LevelImageGen

from .mario_game import MarioGame
from .utils import get_port, load_level


class MarioEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        levels: List[str],
        timer=100,
        visual=False,
        sticky_action_probability=0.1,
        frame_skip=2,
        frame_stack=4,
        frame_dim=64,
        hide_points_banner=False,
        sparse_rewards=False,
        grayscale=False,
        seed=0,
    ):
        self.seed(seed)
        self.level_names = levels
        self.levels = [load_level(name) for name in levels]
        self.timer = timer
        self.visual = visual
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.sticky_action_probability = sticky_action_probability
        self.hide_points_banner = hide_points_banner
        self.sparse_rewards = sparse_rewards
        self.points_banner_height = 4
        self.grayscale = grayscale
        self.last_action = None
        self.width = self.height = frame_dim
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=[self.frame_stack if grayscale else 3, self.height, self.width],
            dtype=np.uint8,
        )
        self.original_obs = deque(maxlen=self.frame_skip)
        self.actions = [
            [False, False, False, False, False],  # noop
            [False, False, True, False, False],  # down
            [False, True, False, False, False],  # right
            [False, True, False, True, False],  # right speed
            [False, True, False, False, True],  # right jump
            [False, True, False, True, True],  # right speed jump
            [True, False, False, False, False],  # left
            [True, False, False, False, True],  # left jump
            [True, False, False, True, True],  # left speed jump
            [False, False, False, False, True],  # jump
        ]
        self.action_space = spaces.Discrete(n=len(self.actions))
        self._obs = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        self.current_level_idx = 0
        self.frame_size = -1
        self.port = get_port()
        self.mario_state: Literal[0, 1, 2] = 0  # normal, large, fire
        self.mario_inertia = 0.89
        self._init_game()

    def reset(self):
        self._reset_obs()
        if self.game is None:
            self.game = self._init_game()
        self.current_level_idx = (self.current_level_idx + 1) % len(self.levels)
        level = self.levels[self.current_level_idx]
        self.game.resetGame(level, self.timer, self.mario_state, self.mario_inertia)
        self.game.computeObservationRGB()
        buffer = self._receive()
        frame = self._read_frame(buffer)
        self._update_obs(frame)
        return self._obs.copy()

    def step(self, action):
        if self.sticky_action_probability != 0.0:
            if (
                self.last_action is not None
                and random.random() < self.sticky_action_probability
            ):
                a = self.actions[self.last_action]
            else:
                a = self.actions[action]
                self.last_action = action
        else:
            a = self.actions[action]

        assert self.game
        frame = None
        for i in range(self.frame_skip):
            self.game.stepGame(*a)
            if self.visual or i == self.frame_skip - 1:
                self.game.computeObservationRGB()
                buffer = self._receive()
                frame = self._read_frame(buffer)
        self._update_obs(frame)

        reward, done, completionPercentage = (
            self.game.computeReward(),
            self.game.computeDone(),
            self.game.getCompletionPercentage(),
        )

        info: Dict[str, Any] = {"completed": completionPercentage}
        if self.visual:
            info["original_obs"] = self.original_obs
        return (
            self._obs.copy(),
            reward if not self.sparse_rewards else int(completionPercentage == 1.0),
            done,
            info,
        )

    def render(self, *args, **kwargs):
        return self.original_obs[0]

    def __getstate__(self):
        assert self.gateway
        self.gateway.close()
        self.gateway = None
        self.game = None
        self.socket.shutdown(1)
        self.socket.close()
        return self.__dict__

    def _reset_obs(self):
        self._obs[:] = 0
        self.original_obs.clear()

    def _read_frame(self, buffer):
        frame = (
            np.frombuffer(buffer, dtype=np.int32).reshape(256, 256, 3).astype(np.uint8)
        )
        self.original_obs.append(frame)
        return frame

    def _update_obs(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (self.width, self.height), cv2.INTER_NEAREST)
        if self.hide_points_banner:
            frame[: self.points_banner_height, :] = 0
        if self.grayscale:
            self._obs = np.concatenate([self._obs[1:], frame[np.newaxis]])
        else:
            self._obs = np.transpose(frame, axes=(2, 0, 1))

    def _init_game(self):
        self.gateway = JavaGateway(
            gateway_parameters=GatewayParameters(
                port=self.port,
                eager_load=True,
            )
        )
        self.game = cast(MarioGame, cast(Any, self.gateway.jvm).engine.core.MarioGame())
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(("localhost", self.game.getPort()))
        self.game.initGame()
        self.frame_size = self.game.getFrameSize()
        return self.game

    def _receive(self):
        frameBuffer = b""
        while len(frameBuffer) != self.frame_size:
            frameBuffer += self.socket.recv(self.frame_size)
        return frameBuffer

    def get_action_meanings(self):
        return ACTION_MEANING

    def render_current_level(self):
        img_gen = LevelImageGen(
            sprite_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "sprites")
            )
        )
        return img_gen.render(self.levels[self.current_level_idx].split("\n"))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


ACTION_MEANING = [
    "NOOP",
    "DOWN",
    "RIGHT",
    "RIGHTSPEED",
    "RIGHTJUMP",
    "RIGHTSPEEDJUMP",
    "LEFT",
    "LEFTJUMP",
    "LEFTSPEEDJUMP",
    "JUMP",
]
