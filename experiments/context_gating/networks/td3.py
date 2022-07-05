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
                    hk.Linear(cfg.context_branch.width),
                    jax.nn.relu
                )
            )
            
            x = state_seq(S["state"])

            # Gate the context according to the requirement
            if cfg.gating_type == 'Hadamard':
                context_gating = context_gating_func(cfg)
            elif cfg.gating_type == 'LSTM':
                context_gating = context_LSTM(cfg)

            if cfg.pi_context:
                x = context_gating(x, S)
            pi_seq = hk.Sequential(
                (
                    hk.Linear(256),
                    jax.nn.relu,
                    hk.Linear(onp.prod(env.action_space.shape),
                              w_init=jnp.zeros),
                    hk.Reshape(env.action_space.shape),
                )
            )
            mu = pi_seq(x)
        else:
            state_seq = hk.Sequential(
                (
                    hk.Linear(512),
                    jax.nn.relu,
                    hk.Linear(256),
                    jax.nn.relu,
                    hk.Linear(onp.prod(env.action_space.shape),
                              w_init=jnp.zeros),
                    hk.Reshape(env.action_space.shape),
                )
            )
            mu = state_seq(S)
        if cfg.tanh_policy:
            mu = jax.nn.tanh(mu)
        return {"mu": mu, "logvar": jnp.full_like(mu, cfg.pi_log_sigma)}

    return pi


def q_func(cfg, env):
    def q(S, A, is_training):
        if cfg.carl.dict_observation_space and not cfg.carl.hide_context:
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.context_branch.width), 
                    jax.nn.relu
                )
            )
            X = jnp.concatenate((S["state"], A), axis=-1)
            x = state_seq(X)

            # Gate the context according to the requirement
            if cfg.gating_type == 'Hadamard':
                context_gating = context_gating_func(cfg)
            elif cfg.gating_type == 'LSTM':            
                context_gating = context_LSTM(cfg)

            if cfg.q_context:
                x = context_gating(x, S)
            q_seq = hk.Sequential(
                (hk.Linear(256), jax.nn.relu, hk.Linear(
                    1, w_init=jnp.zeros), jnp.ravel)
            )
            x = q_seq(x)
        else:
            X = jnp.concatenate((S, A), axis=-1)
            state_seq = hk.Sequential(
                (
                    hk.Linear(512),
                    jax.nn.relu,
                    hk.Linear(256),
                    jax.nn.relu,
                    hk.Linear(1, w_init=jnp.zeros),
                    jnp.ravel,
                )
            )
            x = state_seq(X)
        return x

    return q
