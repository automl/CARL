from typing import Optional, Union
from pathlib import Path

import coax
from experiments.context_gating.networks.sac import pi_func
from experiments.evaluation.loading import load_policy


def make_sac_policy(cfg, env, path: Optional[Union[str, Path]] = None):
    if path is None:
        func_pi = pi_func(cfg, env)
        # main function approximators
        pi = coax.Policy(func_pi, env, random_seed=cfg.seed)
    else:
        pi = load_policy(cfg=cfg, weights_path=path)

    return pi


def make_agent(cfg, env):
    agent_name = cfg.agent
    path = cfg.agent_checkpoint_path
    if agent_name == "sac":
        agent = make_sac_policy(cfg=cfg, env=env, path=path)
    else:
        raise ValueError(f"Unknown algorithm {agent_name}.")
    return agent
