import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gym
from functools import partial
from pathlib import Path
import string
import random
import carl.envs as envs
import coax
import hydra
import jax
import numpy as onp
import wandb
import torch as th
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from rich import print
from hydra.utils import instantiate
from typing import Tuple

from carl.context.sampling import sample_contexts
from carl.utils.types import Contexts
from experiments.context_gating.algorithms.td3 import td3
from experiments.context_gating.algorithms.sac import sac
from experiments.context_gating.algorithms.c51 import c51
from experiments.context_gating.utils import check_wandb_exists, set_seed_everywhere

from experiments.carlbench.context_logging import (
    log_contexts_wandb_traineval,
    log_contexts_json,
)
from experiments.carlbench.context_sampling import ContextSampler
from experiments.common.utils.json_utils import lazy_json_load
from experiments.evaluation_protocol.evaluation_protocol import EvaluationProtocol


base_dir = os.getcwd()


class ActionLimitingWrapper(gym.Wrapper):
    def __init__(self, env, lower, upper):
        super().__init__(env)
        action_dim = self.env.action_space.low.shape
        self.action_space = gym.spaces.Box(low=onp.ones(action_dim)*lower, high=onp.ones(action_dim)*upper)
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class StateNormalizingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def normalize_state(self, state):
        mean = onp.mean(state)
        var = onp.var(state)
        normalized = (state - mean)/var
        return normalized

    def reset(self):
        state = self.env.reset()
        return self.normalize_state(state)

    def step(self, action):
        s, a, r, d = self.env.step(action)
        return self.normalize_state(s), a, r, d

    def __getattr__(self, name):
        return getattr(self.env, name)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def check_config_valid(cfg):
    valid = True
    if cfg.carl.hide_context and cfg.carl.state_context_features:
        valid = False
    if (
        not cfg.context_sampler.context_feature_names
        and cfg.carl.state_context_features is not None
    ):
        valid = False
    return valid


def get_contexts_evaluation_protocol(cfg: DictConfig) -> Contexts:
    sample_function_attrs = {
        "train": "create_train_contexts",
        "test_interpolation": "create_contexts_interpolation",
        "test_interpolation_combinatorial": "create_contexts_interpolation_combinatorial",
        "test_extrapolation_single": "create_contexts_extrapolation_single",
        "test_extrapolation_all": "create_contexts_extrapolation_all",
    }

    kwargs = OmegaConf.to_container(
        cfg.kirk_evaluation_protocol, resolve=True, enum_to_str=True
    )
    del kwargs["follow"]
    if "distribution_type" in kwargs:
        distribution_type = kwargs["distribution_type"]
        del kwargs["distribution_type"]
    else:
        distribution_type = "train"
    cfs = kwargs["context_features"]
    kwargs["context_features"] = [instantiate(config=cf) for cf in cfs]
    ep = EvaluationProtocol(**kwargs)
    if distribution_type in sample_function_attrs:
        sample_function = getattr(ep, sample_function_attrs[distribution_type])
    else:
        raise ValueError(f"Distribution type {distribution_type} unknown.")
    contexts = sample_function(n=cfg.context_sampler.n_samples)
    contexts = contexts.to_dict(orient="index")
    return contexts


def get_contexts_landing_in_space(cfg: DictConfig) -> Contexts:
    gravities = instantiate(cfg.landing_in_space.sample_function)
    key = cfg.landing_in_space.context_feature_key
    contexts = {i: {key: gravities[i]} for i, g in enumerate(gravities)}
    return contexts


def get_traineval_contexts(cfg: DictConfig) -> Tuple[Contexts, Contexts]:
    if cfg.contexts_train_path is not None:
        contexts = lazy_json_load(cfg.contexts_train_path)
    else:
        contexts = get_contexts(cfg=cfg)

    if cfg.eval_on_train_context:
        eval_contexts = contexts
    else:
        if cfg.contexts_eval_path is not None:
            eval_contexts = lazy_json_load(cfg.contexts_eval_path)
        else:
            eval_contexts = get_contexts(cfg=cfg)
    return contexts, eval_contexts


def get_contexts(cfg: DictConfig) -> Contexts:
    if cfg.kirk_evaluation_protocol.follow:
        contexts = get_contexts_evaluation_protocol(cfg)
    elif cfg.landing_in_space.follow:
        contexts = get_contexts_landing_in_space(cfg=cfg)
    else:
        contexts = ContextSampler(**cfg.context_sampler).sample_contexts()
    return contexts


@hydra.main("./configs", "base")
def train(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    if (
        not check_config_valid(cfg)
        or check_wandb_exists(
            dict_cfg,
            unique_fields=[
                "env",
                "seed",
                "experiment",
                # "group",
                "context_sampler.context_feature_names",
                "context_sampler.sigma_rel",
                "carl.state_context_features",
                "carl.hide_context",
                "carl.dict_observation_space",
                "carl.gating_type",
            ],
        )
    ) and not cfg.wandb.debug:
        print(f"Skipping run with cfg {dict_cfg}")
    print(cfg)

    hydra_job = (
        os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
        + "_"
        + os.path.basename(HydraConfig.get().run.dir)
    )
    cfg.wandb.id = hydra_job + "_" + id_generator()

    run = wandb.init(
        id=cfg.wandb.id,
        resume="allow",
        mode="offline" if cfg.wandb.debug else None,
        project=cfg.wandb.project,
        job_type=cfg.wandb.job_type,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        dir=os.getcwd(),
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
    )
    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None
    wandb.config.update({"command": command, "slurm_id": slurm_id})
    set_seed_everywhere(cfg.seed)

    # ----------------------------------------------------------------------
    # Sample contexts
    # ----------------------------------------------------------------------
    contexts, eval_contexts = get_traineval_contexts(cfg=cfg)
    if contexts:
        log_contexts_wandb_traineval(
            train_contexts=contexts, eval_contexts=eval_contexts
        )

    # ----------------------------------------------------------------------
    # Instantiate environments
    # ----------------------------------------------------------------------
    EnvCls = partial(getattr(envs, cfg.env), **cfg.carl)
    env = EnvCls(contexts=contexts)
    eval_env = EnvCls(contexts=eval_contexts)
    env = coax.wrappers.TrainMonitor(env, name=cfg.algorithm)
    key = jax.random.PRNGKey(cfg.seed)
    if cfg.state_context and cfg.carl.dict_observation_space:
        key, subkey = jax.random.split(key)
        context_state_indices = jax.random.choice(
            subkey,
            onp.prod(env.observation_space.spaces["state"].low.shape),
            shape=env.observation_space.spaces["context"].shape,
            replace=True,
        )
        print(f"Using state features {context_state_indices} as context")
    else:
        context_state_indices = None
    cfg.context_state_indices = context_state_indices

    # Normalization and action scaling for dmc envs
    if cfg.env.startswith("CARLDmc"):
        env = ActionLimitingWrapper(env, lower=-1 + 1e-6, upper=1 - 1e-6)
        env = StateNormalizingWrapper(env)
        eval_env = ActionLimitingWrapper(eval_env, lower=-1 + 1e-6, upper=1 - 1e-6)
        eval_env = StateNormalizingWrapper(env)

    # ----------------------------------------------------------------------
    # Log experiment
    # ----------------------------------------------------------------------
    print(OmegaConf.to_yaml(cfg))
    print(env)
    print(f"Observation Space: ", env.observation_space)
    print(f"Action Space: ", env.action_space)
    output_dir = os.getcwd()
    print("Output directory:", output_dir)

    # ----------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------
    if cfg.algorithm == "sac":
        algorithm = sac
    elif cfg.algorithm == "td3":
        algorithm = td3
    elif cfg.algorithm == "c51":
        algorithm = c51
    else:
        raise ValueError(f"Unknown algorithm {cfg.algorithm}")

    avg_return = algorithm(cfg, env, eval_env)

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    run.finish()

    return avg_return


if __name__ == "__main__":
    train()
