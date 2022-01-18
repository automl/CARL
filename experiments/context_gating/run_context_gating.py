import os
from functools import partial
from pathlib import Path

import carl.envs as envs
import coax
import hydra
import jax
import numpy as onp
import wandb
from carl.context.sampling import sample_contexts
from experiments.context_gating.algorithms.ddpg import ddpg
from experiments.context_gating.algorithms.sac import sac
from experiments.context_gating.utils import set_seed_everywhere
from omegaconf import DictConfig, OmegaConf

from carl.context_encoders import *
import torch as th

base_dir = os.getcwd()


def get_encoder(cfg) -> ContextEncoder:
    """
    Loads the state dict of an already trained autoencoder.
    """
    model = None

    if cfg.encoder.weights is not None:
        encoder_cls = eval(cfg.encoder.model)
        model = encoder_cls(
            cfg.encoder.input_dim,
            cfg.encoder.latent_dim,
            cfg.encoder.hidden_dim,
        )

        # Load the weights of a pre-trained AE
        model.load_state_dict(th.load(os.path.join(base_dir, cfg.encoder.weights)))

    return model


@hydra.main("./configs", "base")
def train(cfg: DictConfig):
    wandb.init(
        mode="offline" if cfg.debug else None,
        project="carl",
        entity="tnt",
        group=cfg.group,
        dir=os.getcwd(),
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
    )
    set_seed_everywhere(cfg.seed)

    EnvCls = partial(getattr(envs, cfg.env), **cfg.carl)
    contexts = sample_contexts(cfg.env, **cfg.contexts)
    eval_contexts = sample_contexts(cfg.env, **cfg.contexts)
    if contexts:
        table = wandb.Table(
            columns=sorted(contexts[0].keys()),
            data=[
                [contexts[idx][key] for key in sorted(contexts[idx].keys())]
                for idx in contexts.keys()
            ],
        )
        wandb.log({"train/contexts": table}, step=0)
        eval_table = wandb.Table(
            columns=sorted(eval_contexts[0].keys()),
            data=[
                [eval_contexts[idx][key] for key in sorted(eval_contexts[idx].keys())]
                for idx in eval_contexts.keys()
            ],
        )
        wandb.log({"eval/contexts": eval_table}, step=0)

    env = EnvCls(contexts=contexts, context_encoder=get_encoder(cfg))
    eval_env = EnvCls(contexts=eval_contexts)
    env = coax.wrappers.TrainMonitor(env, name="sac")
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

    print(OmegaConf.to_yaml(cfg))
    print(env)
    print(f"Observation Space: ", env.observation_space)
    print(f"Action Space: ", env.action_space)
    print(f"Contexts: ", contexts)

    if cfg.algorithm == "sac":
        func_dict, avg_return = sac(cfg, env, eval_env)
    elif cfg.algorithm == "ddpg":
        func_dict, avg_return = ddpg(cfg, env, eval_env)
    else:
        raise ValueError(f"Unknown algorithm {cfg.algorithm}")

    coax.utils.dump(func_dict, Path(wandb.run.dir) / "func_dict.pkl.lz4")

    return avg_return


if __name__ == "__main__":
    train()
