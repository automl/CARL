import gym
import coax
import optax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import wandb

from carl.envs.classic_control.carl_pendulum import CARLPendulumEnv
from rich import print as printr

from ..networks.ppo import pi_func, v_func
from ..utils import evaluate, log_wandb, dump_func_dict


def ppo(cfg, env, eval_env):
    func_pi = pi_func(cfg, env)
    func_v = v_func(cfg, env)

    # function approximators
    v = coax.V(func_v, env)
    pi = coax.Policy(func_pi, env)

    # slow-moving avg of pi
    pi_behavior = pi.copy()

    # specify how to update policy and value function
    ppo_clip = coax.policy_objectives.PPOClip(pi, optimizer=optax.adam(cfg.learning_rate))
    simple_td = coax.td_learning.SimpleTD(v, optimizer=optax.adam(cfg.learning_rate))

    # specify how to trace the transitions
    tracer = coax.reward_tracing.NStep(n=cfg.n_step, gamma=cfg.gamma)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=cfg.replay_capacity)

    while env.T < cfg.max_num_frames:
        s = env.reset()

        for t in range(env.env.cutoff):
            a, logp = pi_behavior(s, return_logp=True)
            s_next, r, done, info = env.step(a)

            # add transition to buffer
            tracer.add(s, a, r, done, logp)
            while tracer:
                buffer.add(tracer.pop())

            # update
            if len(buffer) == buffer.capacity:
                for _ in range(4 * buffer.capacity // cfg.batch_size):  # ~4 passes
                    transition_batch = buffer.sample(batch_size=cfg.batch_size)
                    metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
                    metrics_pi = ppo_clip.update(transition_batch, td_error)
                    env.record_metrics(metrics_v)
                    env.record_metrics(metrics_pi)

                buffer.clear()
                pi_behavior.soft_update(pi, tau=cfg.tau)

            if done:
                break

            s = s_next

            if env.period(name="evaluate", T_period=cfg.eval_freq):
                path = dump_func_dict(locals())
                average_returns = evaluate(pi, eval_env, cfg.eval_episodes)
                wandb.log(
                    {
                        "eval/return_hist": wandb.Histogram(average_returns),
                        "eval/return": onp.mean(average_returns),
                    },
                    commit=False,
                )
        log_wandb(env)
    average_returns = evaluate(pi, eval_env, cfg.eval_episodes)
    path = dump_func_dict(locals())
    return onp.mean(average_returns)

# # pick environment
# env = CARLPendulumEnv()
# env = coax.wrappers.TrainMonitor(env)

# printr(env.observation_space.shape)
# printr(env.action_space.shape)

# network_width = 64


# def func_v(S, is_training):
#     # custom haiku function
#     value = hk.Sequential([
#         hk.Linear(network_width),
#         jax.nn.relu,
#         hk.Linear(network_width),
#         jax.nn.relu,
#         hk.Linear(network_width),
#         jax.nn.relu,
#         hk.Linear(1),
#         jnp.ravel,
#     ])
#     return value(S)  # output shape: (batch_size,)


# def func_pi(S, is_training):
#     # custom haiku function (for discrete actions in this example)
#     seq = hk.Sequential([
#         hk.Linear(network_width),
#         jax.nn.relu,
#         hk.Linear(network_width),
#         jax.nn.relu,
#         hk.Linear(network_width),
#         jax.nn.relu,
#         hk.Linear(onp.prod(env.action_space.shape) * 2),
#         hk.Reshape((*env.action_space.shape, 2)),
#     ])
#     x = seq(S)
#     mu, logvar = x[..., 0], x[..., 1]        
#     ret = {"mu": mu, "logvar": logvar}

#     # ret = {'logits': seq(S)}  # logits shape: (batch_size, num_actions)
#     return ret  


# # function approximators
# v = coax.V(func_v, env)
# pi = coax.Policy(func_pi, env)


# # slow-moving avg of pi
# pi_behavior = pi.copy()


# # specify how to update policy and value function
# ppo_clip = coax.policy_objectives.PPOClip(pi, optimizer=optax.adam(0.001))
# simple_td = coax.td_learning.SimpleTD(v, optimizer=optax.adam(0.001))


# # specify how to trace the transitions
# tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
# buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)


# for ep in range(1000):
#     s = env.reset()

#     for t in range(env.env.cutoff):
#         a, logp = pi_behavior(s, return_logp=True)
#         s_next, r, done, info = env.step(a)

#         # add transition to buffer
#         tracer.add(s, a, r, done, logp)
#         while tracer:
#             buffer.add(tracer.pop())

#         # update
#         if len(buffer) == buffer.capacity:
#             for _ in range(4 * buffer.capacity // 32):  # ~4 passes
#                 transition_batch = buffer.sample(batch_size=32)
#                 metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
#                 metrics_pi = ppo_clip.update(transition_batch, td_error)
#                 env.record_metrics(metrics_v)
#                 env.record_metrics(metrics_pi)

#             buffer.clear()
#             pi_behavior.soft_update(pi, tau=0.1)

#         if done:
#             break

#         s = s_next
