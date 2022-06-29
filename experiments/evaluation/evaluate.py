import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import os
from functools import partial
from pathlib import Path
import string
import random
import carl.envs as envs
import coax
import glob
import hydra
import jax
import numpy as onp
import wandb
import torch as th
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from rich import print

from carl.context_encoders import ContextEncoder, ContextAE, ContextVAE, ContextBVAE
from carl.context.sampling import sample_contexts
from experiments.context_gating.algorithms.td3 import td3
from experiments.context_gating.algorithms.sac import sac
from experiments.context_gating.algorithms.c51 import c51
from experiments.context_gating.utils import (
    check_wandb_exists,
    set_seed_everywhere,
    evaluate,
)

from experiments.carlbench.context_logging import (
    log_contexts_wandb_traineval,
    log_contexts_json,
    log_contexts_wandb,
    load_wandb_contexts,
)
from experiments.carlbench.context_sampling import ContextSampler
from experiments.benchmarking.training import get_encoder, id_generator
from experiments.common.utils.json_utils import lazy_json_load
from experiments.evaluation.loading import load_policy


base_dir = os.getcwd()


@hydra.main("./configs", "base")
def train(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    print("Eval Cfg", cfg)

    hydra_job = (
        os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
        + "_"
        + os.path.basename(HydraConfig.get().run.dir)
    )
    cfg.wandb.id = hydra_job + "_" + id_generator()

    traincfg = OmegaConf.load(str(Path(cfg.results_path) / ".hydra" / "config.yaml"))
    print("Train Cfg", traincfg)

    wandbdir = Path(cfg.results_path) / "wandb"

    # run = wandb.init(
    #     id=cfg.wandb.id,
    #     resume="allow",
    #     mode="offline" if cfg.wandb.debug else None,
    #     project=cfg.wandb.project,
    #     job_type=cfg.wandb.job_type,
    #     entity=cfg.wandb.entity,
    #     group=cfg.wandb.group,
    #     dir=os.getcwd(),
    #     config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
    #     sync_tensorboard=True,
    #     monitor_gym=True,
    #     save_code=True,
    #     tags=cfg.wandb.tags,
    #     notes=cfg.wandb.notes,
    # )
    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None
    # wandb.config.update({"command": command, "slurm_id": slurm_id})
    set_seed_everywhere(traincfg.seed)

    # ----------------------------------------------------------------------
    # Sample contexts
    # ----------------------------------------------------------------------
    contexts = None
    if not cfg.contexts_path:
        if cfg.sample_contexts:
            contexts = ContextSampler(**traincfg.context_sampler).sample_contexts()
        else:
            # use train contexts
            dir = "eval"
            if cfg.eval_on_train_contexts:
                dir = "train"

            # load contexts from traindir
            contexts_path = wandbdir / "latest-run" / "files" / "media" / "table" / dir
            contexts_path = glob.glob(
                os.path.join(str(contexts_path), "contexts_*.json")
            )[0]
            print(contexts_path)
            contexts = load_wandb_contexts(contexts_path)
    else:
        contexts = lazy_json_load(cfg.contexts_path)
    if contexts is not None:
        # log_contexts_wandb(contexts=contexts, wandb_key="evalpost/contexts")
        cfg.n_eval_episodes = cfg.n_eval_episodes_per_context * len(contexts)

    # ----------------------------------------------------------------------
    # Instantiate environments
    # ----------------------------------------------------------------------
    EnvCls = partial(getattr(envs, traincfg.env), **traincfg.carl)
    env = EnvCls(contexts=contexts, context_encoder=get_encoder(traincfg))
    env = coax.wrappers.TrainMonitor(env, name=traincfg.algorithm)
    key = jax.random.PRNGKey(traincfg.seed)
    if traincfg.state_context and traincfg.carl.dict_observation_space:
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
    traincfg.context_state_indices = context_state_indices

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
    # Evaluate
    # ----------------------------------------------------------------------
    weights_path = wandbdir / "latest-run" / "files" / "func_dict.pkl.lz4"
    policy = load_policy(traincfg, weights_path=weights_path)
    returns = evaluate(pi=policy, env=env, num_episodes=cfg.n_eval_episodes)
    avg_return = onp.mean(returns)
    print(avg_return)

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    # run.finish()

    return avg_return


if __name__ == "__main__":
    train()
