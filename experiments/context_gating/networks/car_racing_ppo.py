import haiku as hk
import jax


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
                    hk.Conv2D(128, kernel_shape=3, stride=1),
                    jax.nn.relu,
                    hk.Conv2D(256, kernel_shape=3, stride=1),
                    jax.nn.relu,
                    hk.Flatten,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.action_dim),
                    jax.nn.softplus
                )
            )
        x = state_seq(S.astype(float))

        # continuous action space
        mu, logvar = x[..., 0], x[..., 1]
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
                    hk.Conv2D(128, kernel_shape=3, stride=1),
                    jax.nn.relu,
                    hk.Conv2D(256, kernel_shape=3, stride=1),
                    jax.nn.relu,
                    hk.Flatten,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(1),
                )
            )
        x = state_seq(S.astype(float))
        return x

    return v

