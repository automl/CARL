import coax
import numpy as onp
import optax
import wandb

from ..networks.ddpg import pi_func, q_func
from ..utils import evaluate, log_wandb


def ddpg(cfg, env, eval_env):
    func_pi = pi_func(cfg, env)
    func_q = q_func(cfg, env)

    # main function approximators
    pi = coax.Policy(func_pi, env, random_seed=cfg.seed)
    q = coax.Q(
        func_q,
        env,
        action_preprocessor=pi.proba_dist.preprocess_variate,
        random_seed=cfg.seed,
    )

    # target network
    q_targ = q.copy()
    pi_targ = pi.copy()

    # experience tracer
    tracer = coax.reward_tracing.NStep(n=cfg.n_step, gamma=cfg.gamma)
    buffer = coax.experience_replay.SimpleReplayBuffer(
        capacity=cfg.replay_capacity, random_seed=cfg.seed
    )

    qlearning = coax.td_learning.QLearning(
        q,
        pi_targ,
        q_targ,
        loss_function=coax.value_losses.mse,
        optimizer=optax.adam(cfg.learning_rate),
    )
    determ_pg = coax.policy_objectives.DeterministicPG(
        pi, q_targ, optimizer=optax.adam(cfg.learning_rate)
    )

    # action noise
    noise = coax.utils.OrnsteinUhlenbeckNoise(**cfg.noise_kwargs)

    # train
    while env.T < cfg.max_num_frames:
        s = env.reset()
        noise.reset()
        noise.sigma *= cfg.noise_decay  # slowly decrease noise scale

        for t in range(env.env.cutoff):
            a = noise(pi(s))
            s_next, r, done, info = env.step(a)

            # trace rewards and add transition to replay buffer
            tracer.add(s, a, r, done)
            while tracer:
                buffer.add(tracer.pop())

            # learn
            if len(buffer) >= cfg.warmup_num_frames:
                transition_batch = buffer.sample(batch_size=cfg.batch_size)

                metrics = {"OrnsteinUhlenbeckNoise/sigma": noise.sigma}
                metrics.update(determ_pg.update(transition_batch))
                metrics.update(qlearning.update(transition_batch))
                env.record_metrics(metrics)

                # sync target networks
                q_targ.soft_update(q, tau=cfg.q_targ_tau)
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
    return {
        "pi": pi,
        "q": q,
        "q_targ": q_targ,
    }, onp.mean(average_returns)
