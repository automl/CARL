import coax
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import wandb

from ..networks.c51 import q_func
from ..utils import evaluate, log_wandb, dump_func_dict


def c51(cfg, env, eval_env):
    func_q = q_func(cfg, env)

    # main function approximators
    q = coax.StochasticQ(func_q, env, value_range=(cfg.q_min_value, cfg.q_max_value), num_bins=cfg.network.num_atoms)
    pi = coax.BoltzmannPolicy(q, temperature=cfg.pi_temperature)

    # target network
    q_targ = q.copy()

    # experience tracer
    tracer = coax.reward_tracing.NStep(
        n=cfg.n_step, gamma=cfg.gamma, record_extra_info=True
    )
    buffer = coax.experience_replay.SimpleReplayBuffer(
        capacity=cfg.replay_capacity, random_seed=cfg.seed
    )

    qlearning = coax.td_learning.DoubleQLearning(q, q_targ=q_targ, optimizer=optax.adam(cfg.learning_rate))

    while env.T < cfg.max_num_frames:
        s = env.reset()

        for t in range(env.env.cutoff):
            a = pi(s)
            s_next, r, done, info = env.step(a)

            # trace rewards and add transition to replay buffer
            tracer.add(s, a, r, done)
            while tracer:
                buffer.add(tracer.pop())

            # learn
            if len(buffer) >= cfg.warmup_num_frames:
                transition_batch = buffer.sample(batch_size=cfg.batch_size)

                metrics = {}

                metrics.update(qlearning.update(transition_batch))

                env.record_metrics(metrics)

                # sync target networks
                q_targ.soft_update(q, tau=cfg.q_targ_tau)

            if done:
                break

            s = s_next

        if env.period(name="evaluate", T_period=cfg.eval_freq):
            path = dump_func_dict(locals())
            average_returns = evaluate(pi, eval_env, cfg.eval_episodes)
            eval_avg_G = onp.mean(average_returns)
            wandb.log(
                {
                    "eval/return_hist": wandb.Histogram(average_returns),
                    "eval/return": eval_avg_G,
                },
                commit=False,
            )
            print(f"eval_avg_G: {eval_avg_G}")
        log_wandb(env)
    average_returns = evaluate(pi, eval_env, cfg.eval_episodes)
    return onp.mean(average_returns)
