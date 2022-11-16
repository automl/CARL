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
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from rich import print
from tqdm import tqdm
import pandas as pd

from carl.context.sampling import sample_contexts
from experiments.context_gating.algorithms.td3 import td3
from experiments.context_gating.algorithms.sac import sac
from experiments.context_gating.algorithms.c51 import c51
from experiments.context_gating.utils import (
    check_wandb_exists,
    set_seed_everywhere,
)

from experiments.carlbench.context_logging import (
    log_contexts_wandb_traineval,
    log_contexts_json,
    log_contexts_wandb,
    load_wandb_contexts,
)
from experiments.carlbench.context_sampling import ContextSampler
from experiments.benchmarking.training import (
    id_generator,
    get_contexts_evaluation_protocol,
)
from experiments.common.utils.json_utils import lazy_json_load
from experiments.evaluation.loading import load_policy


base_dir = os.getcwd()


def evaluate(pi, env, num_episodes):
    returns = []
    context_ids = []
    for _ in tqdm(range(num_episodes)):
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
        context_ids.append(env.context_selector.context_id)
    return returns, context_ids


def check_evaluation_protocol_combo(mode: str, distribution_type: str) -> bool:
    is_valid = True
    invalids = [
        "A" + "test_interpolation_combinatorial",
        "C" + "test_interpolation",
    ]
    if mode + distribution_type in invalids:
        is_valid = False
    return is_valid


@hydra.main("./configs", "base")
def evaluate_policy(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    hydra_job = (
        os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
        + "_"
        + os.path.basename(HydraConfig.get().run.dir)
    )
    cfg.wandb.id = hydra_job + "_" + id_generator()

    traincfg = OmegaConf.load(str(Path(cfg.results_path) / ".hydra" / "config.yaml"))

    wandbdir = Path(cfg.results_path) / "wandb"

    if cfg.kirk_evaluation_protocol.follow:
        is_valid = check_evaluation_protocol_combo(
            mode=traincfg.kirk_evaluation_protocol.mode,
            distribution_type=cfg.kirk_evaluation_protocol.distribution_type,
        )
        if not is_valid:
            print(
                "Skipping run because combination of evaluation protocol mode and distribution type "
                "is invalid."
            )
            return None
        cfg.kirk_evaluation_protocol.mode = traincfg.kirk_evaluation_protocol.mode

    run = wandb.init(
        id=cfg.wandb.id,
        resume="allow",
        mode="offline" if cfg.wandb.debug else None,
        project=cfg.wandb.project,
        job_type=cfg.wandb.job_type,
        entity=cfg.wandb.entity,
        group=traincfg.wandb.group,
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
    wandb.config.update({"command": command, "slurm_id": slurm_id, "rundir": HydraConfig.get().run.dir})
    set_seed_everywhere(traincfg.seed)

    # ----------------------------------------------------------------------
    # Update config
    # ----------------------------------------------------------------------
    if "carl" in cfg and "context_mask" in cfg.carl:
        with open_dict(traincfg):
            traincfg.carl.context_mask = cfg.carl.context_mask
            # traincfg["carl"]["context_mask"] = cfg.carl.context_mask

    print("Eval Cfg", cfg)
    print("Train Cfg", traincfg)

    # ----------------------------------------------------------------------
    # Sample contexts
    # ----------------------------------------------------------------------
    contexts = None
    if not cfg.contexts_path:
        if cfg.sample_contexts:
            contexts = ContextSampler(**traincfg.context_sampler).sample_contexts()
        elif cfg.kirk_evaluation_protocol.follow:
            # override train config for evaluation protocol with eval config
            traincfg.kirk_evaluation_protocol.distribution_type = (
                cfg.kirk_evaluation_protocol.distribution_type
            )
            contexts = get_contexts_evaluation_protocol(traincfg)
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
    if contexts is not None and len(contexts) > 0:
        log_contexts_wandb(contexts=contexts, wandb_key="evalpost/contexts")
        cfg.n_eval_episodes = cfg.n_eval_episodes_per_context * len(contexts)

    skip_eval = False
    if contexts == {} and cfg.kirk_evaluation_protocol.follow:
        skip_eval = True

    wandb.Table.MAX_ROWS = 50000
    table_entries = cfg.n_eval_episodes 
    if table_entries > wandb.Table.MAX_ROWS:
        raise ValueError(f"Wandb table too small ({wandb.Table.MAX_ROWS}) for {table_entries}. Increase wandb.Table.MAX_ROWS.")

    # ----------------------------------------------------------------------
    # Instantiate environments
    # ----------------------------------------------------------------------
    env_kwargs = OmegaConf.to_container(traincfg.carl, resolve=True)
    # if cfg.landing_in_space.follow:
    #     env_kwargs["high_gameover_penalty"] = True
    EnvCls = partial(getattr(envs, traincfg.env), **env_kwargs)
    env = EnvCls(
        contexts=contexts,
    )
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
    wandb.config.update({"traincfg": traincfg})
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
    policy.observation_preprocessor = coax.utils.default_preprocessor(env.observation_space)
    returns, context_ids = evaluate(
        pi=policy, env=env, num_episodes=cfg.n_eval_episodes
    )
    avg_return = onp.mean(returns)
    df = pd.DataFrame(
        data=onp.vstack((context_ids, returns)).T, columns=["context_id", "return"]
    )
    return_per_context_table = wandb.Table(dataframe=df)
    wandb.log(
        {
            "return_per_context": wandb.plot.bar(
                return_per_context_table,
                "context_id",
                "return",
                title="Return per Context",
            ),
            "average_return": avg_return,  # TODO log as scalar
        }
    )
    print(avg_return)

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    run.finish()

    return avg_return


if __name__ == "__main__":
    evaluate_policy()
