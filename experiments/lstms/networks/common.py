import haiku as hk
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

import pdb

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

        # If the contexts are zeros ? 
        if cfg.zero_context:
            x = x * context_seq(jnp.zeros_like(S["context"]))
        
        elif cfg.state_context:
            state_context = jnp.take(S["state"], cfg.context_state_indices, axis=-1)
            x = x * context_seq(state_context)
        
        else:
            x = x * context_seq(S["context"])

        return x

    return context_gating_seq


# def context_LSTM(cfg: DictConfig):
    
#     # a LSTM module to unroll - use haiku
#     core_lstm = hk.LSTM(
#                     hidden_size=cfg.lstm.encode_width,
#                 )

#     def unroll(state, context, prev_lstm_state):
#         '''
#         What we need to do is.
#             - Initialze:
#                 - the hidden state of lstm with context 
#                 - cell state with 0 vector 
#             - Use the initialization to unroll the LSTM  

#         How do we check for when to do it?
#             - Check for the context vector changing
#             - update a flag in the config to TRUE
#             - put a check for this flag here
        


#         Does re-initialization happen 
#                 - No, everything is wrapped in a single fuction. So, all we need to do
#                 is get the initial module read and just
                        
#         Where is the implicit meta-learning happening across contexts?
#             - in the cell state 
#             - Initialization is adding a bit of additional benifit, but it is the state of 
#             the cell that is retaining meta-learned information


#         '''

#         # TODO: load this somehow between functon calls
#         if cfg.env_reset:
#             # Initialize the hidden state of the lstm with 
#             # the newly sampled context
#             init_state = hk.LSTMState(
#                             hidden=context,           # Set initial hidden state to context encoding
#                             cell=prev_lstm_state.cell            # Set initial state to zero  
#                         )
#         else:
#             # otherwise, bootstrap from the previously propagated state
#             init_state = prev_lstm_state

#         # unroll the lstm
#         output, lstm_state  = core_lstm(
#                         inputs = state,             # Every new state is passed as context input
#                         prev_state = init_state         # the
#                     )


#         return output, lstm_state
        
#     return unroll