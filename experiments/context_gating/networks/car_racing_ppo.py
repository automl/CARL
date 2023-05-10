import haiku as hk
import jax
import numpy as onp


def pixel_pi_func(cfg, env):
    def pi(S, is_training):
        state_seq = hk.Sequential(
                (
                    hk.Conv2D(8, kernel_shape=4, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(16, kernel_shape=3, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(32, kernel_shape=3, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(64, kernel_shape=3, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(128, kernel_shape=3, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(256, kernel_shape=3, stride=3),
                    jax.nn.relu,
                )
            )
        S = onp.swapaxes(S,1,3).squeeze()
        x = state_seq(S.astype(float)).squeeze()
        pi_seq = hk.Sequential(
            (
                hk.Linear(cfg.network.width),
                hk.LayerNorm(-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(onp.prod(env.action_space.shape) * 2),
                #hk.Reshape((*env.action_space.shape, 2)),
            )
        )
        x = pi_seq(x)
        if len(x.shape) == 1:
            x = x[onp.newaxis, ...]
        x = x.reshape((x.shape[0],*env.action_space.shape, 2)).squeeze()

        # continuous action space
        mu, logvar = x[...,0].squeeze(), x[...,1].squeeze(),
        if len(mu.shape) == 1:
            mu = mu[onp.newaxis, :]
            logvar = logvar[onp.newaxis, :]
        ret = {"mu": mu, "logvar": logvar}

        # discrete action space
        # ret = {'logits': seq(S)}  # logits shape: (batch_size, num_actions)
        return ret

    return pi


def pixel_v_func(cfg, env):
    def v(S, is_training):
        state_seq = hk.Sequential(
                (
                    hk.Conv2D(8, kernel_shape=4, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(16, kernel_shape=3, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(32, kernel_shape=3, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(64, kernel_shape=3, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(128, kernel_shape=3, stride=2),
                    jax.nn.relu,
                    hk.Conv2D(256, kernel_shape=3, stride=3),
                    jax.nn.relu,
                )
            )
        S = onp.swapaxes(S,1,3).squeeze()
        x = state_seq(S.astype(float)).squeeze()
        v_seq = hk.Sequential(
            (
                hk.Linear(cfg.network.width),
                jax.nn.relu,
                hk.Linear(1),
            )
        )
        x = v_seq(x)
        if len(x.shape) > 1:
            x = x.squeeze()
        return x

    return v

