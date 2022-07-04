import haiku as hk
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from .lstms import cLSTM


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


def context_LSTM(cfg: DictConfig):
    """
    Function to handle the LSTM class
    """

    core_lstm = cLSTM(
        hidden_size=cfg.context_branch.width,
        init_state=hk.LSTMState(
            hidden=jnp.zeros([1, cfg.context_branch.width]),
            cell=jnp.zeros([1, cfg.context_branch.width]),
        ),
    )

    context_seq = hk.Sequential(
        (
            hk.Linear(cfg.context_branch.width),
            jax.nn.relu,
            hk.Linear(cfg.context_branch.width),
            jax.nn.sigmoid,
        )
    )

    def unroll(encoded_state, obs):
        """
        HK style Function to unroll the LSTM

        Args:
            encoded_state   :   Encoded state
            obs             :   Full dict-observatin

        Returns:
            LSTM hidden state after 1 sstep unroll
        """

        # encode the context
        encoded_context = context_seq(obs["context"])

        output, _ = core_lstm(
            state=encoded_state,  # Every new state is passed input
            context=encoded_context,  # the encoded context for hidden initialization
        )

        return output

    return unroll
