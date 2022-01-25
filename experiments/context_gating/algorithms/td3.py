from multiprocessing import Process

import coax
import jax
import numpy as onp
import optax
import wandb

from ..networks.td3 import pi_func, q_func
from ..utils import dump_func_dict, evaluate, log_wandb


def td3(cfg, env, eval_env):
    func_pi = pi_func(cfg, env)
    func_q = q_func(cfg, env)

    # main function approximators
    pi = coax.Policy(func_pi, env, random_seed=cfg.seed)
    q1 = coax.Q(
        func_q,
        env,
        action_preprocessor=pi.proba_dist.preprocess_variate,
        random_seed=cfg.seed,
    )
    q2 = coax.Q(
        func_q,
        env,
        action_preprocessor=pi.proba_dist.preprocess_variate,
        random_seed=cfg.seed + 1,
    )

    # target network
    q1_targ = q1.copy()
    q2_targ = q2.copy()
    pi_targ = pi.copy()

    # experience tracer
    tracer = coax.reward_tracing.NStep(n=cfg.n_step, gamma=cfg.gamma)
    buffer = coax.experience_replay.SimpleReplayBuffer(
        capacity=cfg.replay_capacity, random_seed=cfg.seed
    )

    qlearning1 = coax.td_learning.ClippedDoubleQLearning(
        q1, pi_targ_list=[pi_targ], q_targ_list=[q1_targ, q2_targ],
        loss_function=coax.value_losses.mse, optimizer=optax.adam(cfg.learning_rate))
    qlearning2 = coax.td_learning.ClippedDoubleQLearning(
        q2, pi_targ_list=[pi_targ], q_targ_list=[q1_targ, q2_targ],
        loss_function=coax.value_losses.mse, optimizer=optax.adam(cfg.learning_rate))
    determ_pg = coax.policy_objectives.DeterministicPG(
        pi, q1_targ, optimizer=optax.adam(cfg.learning_rate)
    )

    # action noise
    if cfg.action_noise.type == "ornsteinuhlenbeck":
        noise = coax.utils.OrnsteinUhlenbeckNoise(random_seed=cfg.seed, **cfg.action_noise.kwargs)
    else:
        noise = None

    # train
    while env.T < cfg.max_num_frames:
        s = env.reset()

        if isinstance(noise, coax.utils.OrnsteinUhlenbeckNoise):
            noise.reset()
            noise.sigma *= cfg.noise_decay  # slowly decrease noise scale

        for t in range(env.env.cutoff):
            a = pi(s)
            if noise is not None:
                a = noise(a)
            s_next, r, done, _ = env.step(a)

            # trace rewards and add transition to replay buffer
            tracer.add(s, a, r, done)
            while tracer:
                buffer.add(tracer.pop())

            # learn
            if len(buffer) >= cfg.warmup_num_frames:
                transition_batch = buffer.sample(batch_size=cfg.batch_size)

                metrics = {}
                if isinstance(noise, coax.utils.OrnsteinUhlenbeckNoise):
                    metrics["OrnsteinUhlenbeckNoise/sigma"] = noise.sigma

                qlearning = qlearning1 if jax.random.bernoulli(
                    q1.rng) else qlearning2
                metrics.update(qlearning.update(transition_batch))

                if env.T >= cfg.warmup_pi_num_frames and env.T % 2 == 0:
                    metrics.update(determ_pg.update(transition_batch))

                env.record_metrics(metrics)

                # sync target networks
                q1_targ.soft_update(q1, tau=cfg.q_targ_tau)
                q2_targ.soft_update(q2, tau=cfg.q_targ_tau)
                pi_targ.soft_update(pi, tau=cfg.pi_targ_tau)

            if done:
                break

            s = s_next
        # if env.period(name='generate_gif', T_period=cfg.render_freq) and env.T > cfg.q_warmup_num_frames:
        #     T = env.T - env.T % cfg.render_freq  # round
        #     gif_path = f"{os.getcwd()}/gifs/T{T:08d}.gif"
        #     coax.utils.generate_gif(
        #         env=env, policy=pi, filepath=gif_path)
        #     wandb.log({"eval/episode": wandb.Video(
        #         gif_path, caption=str(T), fps=30)}, commit=False)
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
    return onp.mean(average_returns)
