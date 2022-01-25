import haiku as hk
import jax
import jax.numpy as jnp
from omegaconf import DictConfig


def context_gating_func(cfg: DictConfig):
    context_seq = hk.Sequential(
        (
            hk.Linear(cfg.context_branch.width),
            jax.nn.relu,
            hk.Linear(cfg.context_branch.width),
            jax.nn.sigmoid,
        )
    )

    def context_gating_seq(x, S):
        if cfg.zero_context:
            x = x * context_seq(jnp.zeros_like(S["context"]))
        elif cfg.state_context:
            state_context = jnp.take(S["state"], cfg.context_state_indices, axis=-1)
            x = x * context_seq(state_context)
        else:
            x = x * context_seq(S["context"])
        return x

    return context_gating_seq
