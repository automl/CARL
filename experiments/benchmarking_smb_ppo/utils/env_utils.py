import logging
import os

import gym
import gym.spaces
from gym.vector import AsyncVectorEnv
from gym.wrappers.autoreset import AutoResetWrapper
from gym.wrappers.normalize import NormalizeReward
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.record_video import RecordVideo

from carl.envs.mario.carl_mario import CARLMarioEnv
from carl.envs.mario.pcg_smb_env.mario_env import MarioEnv  # noqa: F401

logger = logging.getLogger(__name__)


def make_env(
    id: str,
    num_envs: int = 1,
    capture_video: bool = False,
    capture_all_episodes: bool = False,
    contexts: dict = {},
    **kwargs,
):
    def env_fn():
        return AutoResetWrapper(CARLMarioEnv(env=MarioEnv(levels=[], **kwargs), contexts=contexts))
    if num_envs > 1:
        envs = AsyncVectorEnv([env_fn for _ in range(num_envs)], shared_memory=False, copy=False)
    else:
        envs = env_fn()
    envs = NormalizeReward(envs)
    if capture_video:
        assert num_envs == 1
        envs = RecordVideo(
            envs,
            os.path.join(os.getcwd(), "videos"),
            episode_trigger=(lambda episode: True) if capture_all_episodes else None,
        )
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(
        envs.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)
    ), "only discrete action space is supported"
    logging.info(f"Created envionment: {envs}")
    return envs
