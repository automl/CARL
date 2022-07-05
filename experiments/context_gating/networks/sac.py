from logging import raiseExceptions
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
from ..networks.common import context_gating_func, context_LSTM

def pi_func(cfg, env):
    def pi(S, is_training):

        if cfg.carl.dict_observation_space and not cfg.carl.hide_context:
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                )
            )

            # encode the state
            x = state_seq(S["state"])  
            width = cfg.network.width

            # Gate the context according to the requirement
            if cfg.gating_type == 'Hadamard':
                context_gating = context_gating_func(cfg)
            elif cfg.gating_type == 'LSTM':
                assert cfg.network.width == cfg.context_branch.width
                
                context_gating = context_LSTM(cfg)

            if cfg.pi_context:
                # Get the state modulated by the context
                x = context_gating(x, S)
                width = cfg.context_branch.width

            # convert gating output to action dimensions
            pi_seq = hk.Sequential(
                (
                    hk.Linear(width),
                    jax.nn.relu,
                    hk.Linear(onp.prod(env.action_space.shape) * 2, w_init=jnp.zeros),
                    hk.Reshape((*env.action_space.shape, 2)),
                )
            )
            x = pi_seq(x)

        else:
            # directly map state to actions
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(onp.prod(env.action_space.shape) * 2, w_init=jnp.zeros),
                    hk.Reshape((*env.action_space.shape, 2)),
                )
            )
            x = state_seq(S)

        mu, logvar = x[..., 0], x[..., 1]
        return {"mu": mu, "logvar": logvar}

    return pi


def q_func(cfg, env):
    def q(S, A, is_training):
        if cfg.carl.dict_observation_space and not cfg.carl.hide_context:

            # Encode the state each time
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                )
            )

            # Encode state and action combined
            X = jnp.concatenate((S["state"], A), axis=-1)
            x = state_seq(X)

            # Gate the context according to the requirement
            if cfg.gating_type == 'Hadamard':
                context_gating = context_gating_func(cfg)
            elif cfg.gating_type == 'LSTM':
                assert cfg.network.width == cfg.context_branch.width
                
                context_gating = context_LSTM(cfg)

            if cfg.q_context:
                x = context_gating(x, S)

            # convert to Q value
            q_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(1, w_init=jnp.zeros),
                    jnp.ravel,
                )
            )
            x = q_seq(x)

        else:
            X = jnp.concatenate((S, A), axis=-1)
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(1, w_init=jnp.zeros),
                    jnp.ravel,
                )
            )
            x = state_seq(X)
        return x

    return q
