import haiku as hk
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
from typing import Tuple


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

        x_and_h = jnp.concatenate([state, prev_state.hidden], axis=-1)

        gated = hk.Linear(4 * self.hidden_size)(x_and_h)

        i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
        f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
        c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)

        new_state = hk.LSTMState(h, c)

        self.set_state(new_state)

        return h, new_state

    def set_state(self, state: hk.LSTMState) -> hk.LSTMState:
        """
        Function to set the state
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
