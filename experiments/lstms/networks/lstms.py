import haiku as hk
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

import pdb

from typing import Tuple


class cLSTM(hk.LSTM):
  

  def __init__(self, hidden_size, init_state):
    """Constructs an LSTM.

    Args:
      hidden_size : Hidden layer size.
      init_state  : Initial hidden state
    """
    super().__init__(hidden_size=hidden_size)
    self.hidden_size = hidden_size
    self.prev_lstm_state = init_state
  
  
  def __call__(
      self,
      state: jnp.ndarray,
      context: jnp.ndarray, 
      new_context: bool = False ) -> Tuple[jnp.ndarray, hk.LSTMState]:
    
    if len(state.shape) > 2 or not state.shape:
      raise ValueError("LSTM input must be rank-1 or rank-2.")
    

    if not new_context:
        prev_state = self.prev_lstm_state
    else:
        prev_state = hk.LSTMState(
            hidden = context,
            cell = self.prev_lstm_state.cell
        )

    
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
    
    self.prev_lstm_state = state



def context_LSTM(cfg: DictConfig):
    
    # a LSTM module to unroll - use haiku
    core_lstm = cLSTM(
                    hidden_size=cfg.lstm.encode_width,
                    init_state = hk.LSTMState(
                        hidden  = jnp.zeros([cfg.lstm.encode_width]),
                        cell    = jnp.zeros([cfg.lstm.encode_width])
                    )
                )

    def unroll(state, context):

        # unroll the lstm
        output, _  = core_lstm(
                        state = state,             # Every new state is passed as context input
                        context = context,         # the
                        new_context = cfg.env_reset
                    )


        return output
        
    return unroll