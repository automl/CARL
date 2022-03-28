# import os
# from pathlib import Path
# import random
# import sys
from functools import partial
from typing import Optional, cast, Dict, Any, Union
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# import coax
import gym
import hydra
# import numpy as onp
# import wandb
from omegaconf import DictConfig, OmegaConf


import carl.envs


def make_env(cfg: DictConfig, contexts: Dict[str, Dict] = None, name: Optional[str] = None, tensorboard_dir: Optional[str] = None, num_envs: int = 1):
    if num_envs > 1:
        EnvCls = partial(make_carl_env, cfg=cfg, contexts=contexts)
        vec_env_cls = SubprocVecEnv
        env = make_vec_env(EnvCls, n_envs=num_envs, vec_env_cls=vec_env_cls)
    else:
        env = make_carl_env(cfg)
    return env


def make_carl_env(cfg: DictConfig, contexts: Dict[str, Dict] = None, log_wandb: bool = False):
    env = getattr(carl.envs, cfg.env)(contexts=contexts, **cfg.carl.env_kwargs)
    env.seed(cfg.seed)
    # env.spec = gym.envs.registration.EnvSpec(cfg.env + "-v0")
    for wrapper in cfg.env_wrappers:
        env = hydra.utils.instantiate(wrapper, env)

    # env = coax.wrappers.TrainMonitor(
    #     env, name=name or cfg.algo, tensorboard_dir=tensorboard_dir)

    return env