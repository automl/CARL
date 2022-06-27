import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp

from ..networks.lstms import context_LSTM

import pdb

def pi_func(cfg, env):
    
    lstm_state = None

    def pi(S, is_training):


        if cfg.carl.dict_observation_space and not cfg.carl.hide_context:

            nonlocal lstm_state 

            if not lstm_state:
                if cfg.lstm.encode:
                    # Initialize with shape of encoded wwidth
                    lstm_state = hk.LSTMState(
                                    hidden=jnp.zeros([cfg.lstm.encode_width]),
                                    cell=jnp.zeros([cfg.lstm.encode_width]))
                else:
                    # Initialize with 
                    lstm_state = hk.LSTMState(
                        hidden  =   jnp.zeros (shape = len(S["context"])),
                        cell    =   jnp.zeros (shape= len(S["context"])) 
                    )

            # print(lstm_state.hidden.shape)
            # pdb.set_trace()

            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                )
            )

            x =  state_seq(S['state'])

            # print(x.shape)
            # pdb.set_trace()


            # Assign the LSTM        
            context_gating = context_LSTM(cfg)
         
            
            if cfg.lstm.encode:
                # Encode context
                context_seq = hk.Sequential(
                            (
                                hk.Linear(cfg.lstm.encode_width),
                                jax.nn.relu,
                                hk.Linear(cfg.lstm.encode_width),
                                jax.nn.sigmoid,                 # TODO check for sigmoid vs relu activations
                            )
                        )
                # Encode context and state to requisite width
                # TODO test without encoding
                context = context_seq(S["context"])

                # print(context.shape)
                # pdb.set_trace()

            else:
                # Pass the state and context unencoded 
                context = S['context']
            
            width = cfg.lstm.encode_width
            if cfg.pi_context:
                x, new_state = context_gating(
                                    context = context.reshape(-1), 
                                    state = x.reshape(-1), 
                                    prev_lstm_state = lstm_state
                                )
                
                
                x = x.reshape(1,-1)
                # print(x.shape)
                # pdb.set_trace()
                # Store the lstm_state
                lstm_state = new_state                  
            
            # readout of LSTM 
            # from branch width to action space
            pi_seq = hk.Sequential(
                (
                    hk.Linear(width),
                    jax.nn.relu,
                    hk.Linear(onp.prod(env.action_space.shape) * 2, w_init=jnp.zeros),
                    hk.Reshape((*env.action_space.shape, 2)),
                )
            )

            # convert to action space
            x = pi_seq(x)
           
        else:
            # directly map state to actions
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                    hk.Linear(cfg.lstm.encode_width),
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

    lstm_state = None

    def q(S, A, is_training):
        if cfg.carl.dict_observation_space and not cfg.carl.hide_context:
            
            # initialize lstm_state
            nonlocal lstm_state 
            if not lstm_state:
                if cfg.lstm.encode:
                    # Initialize with shape of encoded wwidth
                    lstm_state = hk.LSTMState(
                                    hidden=jnp.zeros([cfg.lstm.encode_width]),
                                    cell=jnp.zeros([cfg.lstm.encode_width]))
                else:
                    # Initialize with 
                    lstm_state = hk.LSTMState(
                        hidden  =   jnp.zeros (shape = len(S["context"])),
                        cell    =   jnp.zeros (shape= len(S["context"])) 
                    )
            
            # Encodet the state each time
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                )
            )
            
            # Encode state and action combined
            X = jnp.concatenate((S["state"], A), axis=-1)
            x = state_seq(X)

            # initialize the gating function
            context_gating = context_LSTM(cfg)

             
            if cfg.lstm.encode:
                # Encode context
                context_seq = hk.Sequential(
                            (
                                hk.Linear(cfg.lstm.encode_width),
                                jax.nn.relu,
                                hk.Linear(cfg.lstm.encode_width),
                                jax.nn.sigmoid,                 # TODO check for sigmoid vs relu activations
                            )
                        )
                # Encode context and state to requisite width
                # TODO test without encoding
                context = context_seq(S["context"])

            else:
                # Pass the state and context unencoded 
                context = S['context']

            if cfg.q_context:
                x, new_state = context_gating(
                                    context = context.reshape(-1), 
                                    state = x.reshape(-1), 
                                    prev_lstm_state = lstm_state
                                )

                
                x = x.reshape(1,-1)
                
                # retain the LSTM state
                lstm_state = new_state


            # LSTM readout
            q_seq = hk.Sequential(
                (
                    hk.Linear(cfg.lstm.encode_width),
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
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                    hk.Linear(cfg.lstm.encode_width),
                    jax.nn.relu,
                    hk.Linear(1, w_init=jnp.zeros),
                    jnp.ravel,
                )
            )
            x = state_seq(X)
        return x

    return q
