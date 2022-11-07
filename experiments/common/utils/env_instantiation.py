from __future__ import annotations

from typing import Dict

import hydra
from omegaconf import DictConfig

import carl.envs


def make_carl_env(
    cfg: DictConfig, contexts: Dict[str, Dict] | None = None, log_wandb: bool = False
):
    env = getattr(carl.envs, cfg.env)(contexts=contexts, **cfg.carl.env_kwargs)
    env.seed(cfg.seed)
    # env.spec = gym.envs.registration.EnvSpec(cfg.env + "-v0")
    if "env_wrappers" in cfg:
        for wrapper in cfg.env_wrappers:
            env = hydra.utils.instantiate(wrapper, env)

    # env = coax.wrappers.TrainMonitor(
    #     env, name=name or cfg.algo, tensorboard_dir=tensorboard_dir)

    return env
