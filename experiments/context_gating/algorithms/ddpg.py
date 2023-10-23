from multiprocessing import Process

import coax
import jax
import numpy as onp
import optax
import wandb

from ..networks.ddpg import pi_func, q_func
from ..utils import dump_func_dict, evaluate, log_wandb


def ddpg(cfg, env, eval_env):
    func_pi = pi_func(cfg, env)
    func_q = q_func(cfg, env)

    # main function approximators
    pi = coax.Policy(func_pi, env)
    q = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)

    # target network
    q_targ = q.copy()
    pi_targ = pi.copy()

    # experience tracer
    tracer = coax.reward_tracing.NStep(n=cfg.n_step, gamma=cfg.gamma)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=cfg.replay_capacity, random_seed=cfg.seed)

    # updaters
    qlearning = coax.td_learning.QLearning(
        q, pi_targ, q_targ, loss_function=coax.value_losses.mse, optimizer=optax.adam(cfg.learning_rate))
    determ_pg = coax.policy_objectives.DeterministicPG(pi, q_targ, optimizer=optax.adam(cfg.learning_rate_pg))

    # action noise
    noise = coax.utils.OrnsteinUhlenbeckNoise(random_seed=cfg.seed, **cfg.action_noise.kwargs)

    # train
    while env.T < cfg.max_num_frames:
        s = env.reset()
        noise.reset()
        noise.sigma *= cfg.noise_decay  # slowly decrease noise scale

        for t in range(env.env.cutoff):
            a = noise(pi(s))
            s_next, r, done, _ = env.step(a)

            # trace rewards and add transition to replay buffer
            tracer.add(s, a, r, done)
            while tracer:
                buffer.add(tracer.pop())

            # learn
            if len(buffer) >= cfg.warmup_num_frames:
                transition_batch = buffer.sample(batch_size=128)

                metrics = {'OrnsteinUhlenbeckNoise/sigma': noise.sigma}
                metrics.update(determ_pg.update(transition_batch))
                metrics.update(qlearning.update(transition_batch))
                env.record_metrics(metrics)

                # sync target networks
                q_targ.soft_update(q, tau=cfg.q_targ_tau)
                pi_targ.soft_update(pi, tau=cfg.pi_targ_tau)

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
    average_returns = evaluate(pi, eval_env, cfg.n_final_eval_episodes * cfg.context_sampler.n_samples)
    path = dump_func_dict(locals())
    return onp.mean(average_returns)