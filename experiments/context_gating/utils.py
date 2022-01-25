from pathlib import Path
import random
from typing import Any, Dict, List

import coax
import numpy as onp
import pandas as pd
import wandb


def dump_func_dict(locals_dict: Dict[str, Any]):
    func_dict = {}
    for key, val in locals_dict.items():
        if key in ["q", "q1", "q2", "pi", "pi_targ", "q_targ", "q1_targ", "q2_targ"]:
            func_dict[key] = val
    path = Path(wandb.run.dir) / "func_dict.pkl.lz4"
    coax.utils.dump(func_dict, path)
    return path


def evaluate(pi, env, num_episodes):
    returns = []
    for _ in range(num_episodes):
        ret = 0
        s = env.reset()

        for t in range(env.cutoff):
            a = pi.mean(s)  # use mean for exploitation
            s_next, r, done, info = env.step(a)
            ret += r
            if done:
                break
            s = s_next
        returns.append(ret)
    return returns


def set_seed_everywhere(seed: int):
    onp.random.seed(seed)
    random.seed(seed)


def log_wandb(train_monitor_env: coax.wrappers.TrainMonitor):
    metrics = {
        "train/episode": train_monitor_env.ep,
        "train/avg_reward": train_monitor_env.avg_r,
        "train/return": train_monitor_env.G,
        "train/steps": train_monitor_env.t,
        "train/avg_step_duration_ms": train_monitor_env.dt_ms,
    }

    if train_monitor_env._ep_actions:
        metrics["train/actions"] = wandb.Histogram(
            train_monitor_env._ep_actions.values)
    if train_monitor_env._ep_metrics and not train_monitor_env.tensorboard_write_all:
        for k, (x, n) in train_monitor_env._ep_metrics.items():
            metrics[str(k)] = float(x) / n
    wandb.log(metrics, step=train_monitor_env.T)


def check_wandb_exists(cfg, unique_fields: List[str]):

    flat_cfg = list(pd.json_normalize(cfg).T.to_dict().values())[0]
    query_config = {}
    for key, value in flat_cfg.items():
        if key not in unique_fields:
            continue
        query_config[key] = value

    query_config_wandb = {"config.{}".format(
        key): value for key, value in query_config.items()}

    query_wandb = {
        'state': 'finished',
        **query_config_wandb
    }
    print(query_wandb)

    api = wandb.Api()
    runs = api.runs("tnt/carl", query_wandb)

    found_run = False
    for run in runs:
        if cfg["env"] == "CARLPendulumEnv":
            episode = run.summary['train/episode'] if 'train/episode' in run.summary else -1
            if episode != 2488:
                # run not completed
                continue
        elif cfg["env"] == "CARLAnt":
            episode = run.summary['train/episode'] if 'train/episode' in run.summary else -1
            if episode < 500:
                # run not completed
                continue
        found_run = True

    return found_run
