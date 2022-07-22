from functools import partial
from typing import Optional, Dict
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from omegaconf import DictConfig
from common.utils.env_instantiation import make_carl_env


def make_env(
    cfg: DictConfig,
    contexts: Dict[str, Dict] = None,
    name: Optional[str] = None,
    tensorboard_dir: Optional[str] = None,
    num_envs: int = 1,
):
    if num_envs > 1:
        EnvCls = partial(make_carl_env, cfg=cfg, contexts=contexts)
        vec_env_cls = SubprocVecEnv
        env = make_vec_env(EnvCls, n_envs=num_envs, vec_env_cls=vec_env_cls)
    else:
        env = make_carl_env(cfg)
    return env
