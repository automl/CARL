from logging import raiseExceptions
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
from ..networks.common import get_gating_function


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
            context_gating = get_gating_function(cfg=cfg)

            if cfg.pi_context:
                # Get the state modulated by the context
                x = context_gating(x, S)
                width = cfg.context_branch.width

            # convert gating output to action dimensions
            pi_seq = hk.Sequential(
                (
                    hk.Linear(width),
                    jax.nn.relu,
                    hk.Linear(onp.prod(env.action_space.shape) * 2),
                    hk.Reshape((*env.action_space.shape, 2)),
                )
            )
            x = pi_seq(x)

        else:
            # directly map state to actions
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    hk.LayerNorm(-1, create_scale=True, create_offset=True),
                    jax.nn.tanh,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(onp.prod(env.action_space.shape) * 2),
                    hk.Reshape((*env.action_space.shape, 2)),
                )
            )
            x = state_seq(S)

        # continuous action space
        mu, logvar = x[..., 0], x[..., 1]
        ret = {"mu": mu, "logvar": logvar}

        # discrete action space
        # ret = {'logits': seq(S)}  # logits shape: (batch_size, num_actions)
        return ret

    return pi


def v_func(cfg, env):
    def v(S, is_training):
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
            X = S["state"]
            x = state_seq(X)

            # Gate the context according to the requirement
            context_gating = get_gating_function(cfg=cfg)

            if cfg.q_context:
                x = context_gating(x, S)

            # convert to Q value
            q_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(1),
                    jnp.ravel,
                )
            )
            x = q_seq(x)

        else:
            X = S
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    hk.LayerNorm(-1, create_scale=True, create_offset=True),
                    jax.nn.tanh,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(1),
                    jnp.ravel,
                )
            )
            x = state_seq(X)
        return x

    return v
