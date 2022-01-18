import coax
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import wandb

from ..networks.sac import pi_func, q_func
from ..utils import evaluate, log_wandb, dump_func_dict


def sac(cfg, env, eval_env):
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
        random_seed=cfg.seed,
    )

    # target network
    q1_targ = q1.copy()
    q2_targ = q2.copy()

    # experience tracer
    tracer = coax.reward_tracing.NStep(
        n=cfg.n_step, gamma=cfg.gamma, record_extra_info=True
    )
    buffer = coax.experience_replay.SimpleReplayBuffer(
        capacity=cfg.replay_capacity, random_seed=cfg.seed
    )
    policy_regularizer = coax.regularizers.NStepEntropyRegularizer(
        pi, beta=cfg.alpha / tracer.n, gamma=tracer.gamma, n=[tracer.n]
    )

    qlearning1 = coax.td_learning.SoftClippedDoubleQLearning(
        q1,
        pi_targ_list=[pi],
        q_targ_list=[q1_targ, q2_targ],
        loss_function=coax.value_losses.mse,
        optimizer=optax.adam(cfg.learning_rate),
        policy_regularizer=policy_regularizer,
    )
    qlearning2 = coax.td_learning.SoftClippedDoubleQLearning(
        q2,
        pi_targ_list=[pi],
        q_targ_list=[q1_targ, q2_targ],
        loss_function=coax.value_losses.mse,
        optimizer=optax.adam(cfg.learning_rate),
        policy_regularizer=policy_regularizer,
    )
    soft_pg = coax.policy_objectives.SoftPG(
        pi,
        [q1_targ, q2_targ],
        optimizer=optax.adam(cfg.learning_rate),
        regularizer=coax.regularizers.NStepEntropyRegularizer(
            pi, beta=cfg.alpha / tracer.n, gamma=tracer.gamma, n=jnp.arange(tracer.n)
        ),
    )
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

                # flip a coin to decide which of the q-functions to update
                qlearning = qlearning1 if jax.random.bernoulli(
                    q1.rng) else qlearning2
                metrics.update(qlearning.update(transition_batch))

                # delayed policy updates
                if (
                    env.T >= cfg.pi_warmup_num_frames
                    and env.T % cfg.pi_update_freq == 0
                ):
                    metrics.update(soft_pg.update(transition_batch))

                env.record_metrics(metrics)

                # sync target networks
                q1_targ.soft_update(q1, tau=cfg.q_targ_tau)
                q2_targ.soft_update(q2, tau=cfg.q_targ_tau)

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
