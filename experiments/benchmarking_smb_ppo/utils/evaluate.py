import logging
import os
from typing import Union

import torch
import wandb
from gym.wrappers.autoreset import AutoResetWrapper
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.record_video import RecordVideo
from modules.ppo import PPOAgent
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate(
    agent: PPOAgent,
    env: Union[RecordVideo, RecordEpisodeStatistics, AutoResetWrapper],
    cfg: DictConfig,
):
    avg_reward = 0.0
    avg_completed = 0.0
    for i in tqdm(range(cfg.eval_env.episodes)):
        obs = torch.tensor(
            env.reset(
                options=dict(current_level_idx=i)
                if cfg.eval_env.capture_all_episodes
                else None
            )
        ).to(cfg.device)
        recording = env.recording
        done = False
        info = None
        while not done:
            with torch.no_grad():
                action, *_ = agent.get_action_and_value(obs[None])
            next_obs, reward, done, info = env.step(
                int(action.squeeze().cpu().numpy())
            )
            obs = torch.tensor(next_obs).to(cfg.device)
            avg_reward += reward
        if recording:
            video_path = os.path.join(
                env.video_folder,
                f"{env.name_prefix}-episode-{env.episode_count - 1}.mp4",
            )
            postfix = f"_{i}" if cfg.eval_env.capture_all_episodes else ""
            wandb.log(
                {
                    f"eval/video_{postfix}": wandb.Video(
                        video_path, fps=30, format="mp4"
                    )
                },
                commit=False,
            )
        if info is not None and "final_info" in info:
            avg_completed += info["final_info"]["completed"]
    avg_reward /= cfg.eval_env.episodes
    avg_completed /= cfg.eval_env.episodes
    return avg_reward, avg_completed
