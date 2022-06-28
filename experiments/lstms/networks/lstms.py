import haiku as hk
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

import pdb

from typing import Tuple

import pdb


def squarify(x):
    batch_size = x.shape[0]
    if len(x.shape) > 1:
        representation_dim = x.shape[-1]
        return jnp.reshape(
            jnp.tile(x, batch_size), (batch_size, batch_size, representation_dim)
        )
    return jnp.reshape(jnp.tile(x, batch_size), (batch_size, batch_size))


class cLSTM(hk.LSTM):
    """
    A class to use LSTM as a slow network, adapted from the hk.LSTM module
    """

    def __init__(self, hidden_size, init_state):
        """Constructs a cLSTM.

        Args:
            hidden_size : Hidden layer size.
            init_state  : Initial hidden state
        """
        super().__init__(hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.prev_lstm_state = None
        self.previous_context = None

        self.set_state(init_state)

    def __call__(
        self,
        state: jnp.ndarray,
        context: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, hk.LSTMState]:
        """
        A modified version of the standard LSTM call that checks for context change
        and initializes the hidden state whenever a new context value is provided

        Args:
            state   :   Encoded State
            context :   Encoded context

        Returns:
            - LSTM hidden state after one unroll
            - Full LSTM state
        """

        if len(state.shape) > 2 or not state.shape:
            raise ValueError("LSTM input must be rank-1 or rank-2.")

        if not self._check_context(context):
            # If the context has not changed
            # Propagate the hidden state by ensuring common batch

            prev_state = self._prepare_hidden_state(context)

        else:
            # If the context has changed
            # initialize the hidden state with context
            prev_state = hk.LSTMState(hidden=context, cell=self.prev_lstm_state.cell)

        # print('state', state.shape)
        # print('prev_state.hidden', prev_state.hidden.shape)
        # pdb.set_trace()

        x_and_h = jnp.concatenate([state, prev_state.hidden], axis=-1)

        # print('x_and_h', x_and_h.shape)

        gated = hk.Linear(4 * self.hidden_size)(x_and_h)

        # print(f'gated:',gated.shape)

        i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)

        # print('i:', i.shape)
        # print('g:', g.shape)
        # print('f:', f.shape)
        # print('o:', o.shape)

        f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.

        # print('f', f.shape)

        c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)

        # print('c', c.shape)

        h = jax.nn.sigmoid(o) * jnp.tanh(c)

        # print('h:', h.shape)
        new_state = hk.LSTMState(h, c)

        # pdb.set_trace()

        self.set_state(new_state)

        return h, new_state

    def set_state(self, state: hk.LSTMState) -> hk.LSTMState:
        """
        Function to set the state

        TODO: Additional logic here potentially
        """

        self.prev_lstm_state = state

    def _check_context(self, context: jnp.ndarray) -> bool:
        """
        Function to check for context change
        """

        if not self.previous_context:
            self.previous_context = context
            return True
        else:

            if jnp.array_equiv(context, self.previous_context):
                return False
            else:
                True

    def _prepare_hidden_state(self, state):
        """
        Function to preprocess hidden state in case of batched
        operation

        """

        if self.prev_lstm_state.hidden.shape[0] < state.shape[0]:

            prev_state = hk.LSTMState(
                hidden=jnp.tile(A=self.prev_state.hidden, reps=state.shape(0)),
                cell=jnp.tile(A=self.prev_state.hidden, reps=state.shape(0)),
            )
        elif self.prev_lstm_state.hidden.shape[0] > state.shape[0]:

            prev_state = hk.LSTMState(
                hidden=jnp.tile(A=self.prev_state.hidden[0], reps=state.shape(0)),
                cell=jnp.tile(A=self.prev_state.hidden[0], reps=state.shape(0)),
            )
        else:
            prev_state = self.prev_lstm_state

        return prev_state


def context_LSTM(cfg: DictConfig):
    """
    function to Handle the LSTM class
    """

    # a LSTM module to unroll - use haiku
    core_lstm = cLSTM(
        hidden_size=cfg.lstm.encode_width,
        init_state=hk.LSTMState(
            hidden=jnp.zeros([1, cfg.lstm.encode_width]),
            cell=jnp.zeros([1, cfg.lstm.encode_width]),
        ),
    )

    def unroll(state, context):
        """
        HK style Function to unroll the LSTM

        Args:
            state   :   Encoded state
            context :   Encoded context

        Returns:
            LSTM hidden state after 1 sstep unroll
        """

        output, _ = core_lstm(
            state=state,  # Every new state is passed as context input
            context=context,  # the
        )

        return output

    return unroll
