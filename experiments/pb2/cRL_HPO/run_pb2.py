import os
import sys
from functools import partial
import numpy as np
import yaml
import gym
from pathlib import Path
import argparse

import ray
from ray import tune
from ray.tune.schedulers.pb2 import PB2

import coax
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import wandb
from experiments.context_gating.networks.sac import pi_func, q_func

from carl.context.sampling import sample_contexts


class ActionLimitingWrapper(gym.Wrapper):
    def __init__(self, env, lower, upper):
        super().__init__(env)
        action_dim = self.env.action_space.low.shape
        self.action_space = gym.spaces.Box(low=onp.ones(action_dim) * lower, high=onp.ones(action_dim) * upper)

    def __getattr__(self, name):
        return getattr(self.env, name)


class StateNormalizingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def normalize_state(self, state):
        mean = onp.mean(state)
        var = onp.var(state)
        normalized = (state - mean) / var
        return normalized

    def reset(self):
        state = self.env.reset()
        return self.normalize_state(state)

    def step(self, action):
        s, a, r, d = self.env.step(action)
        return self.normalize_state(s), a, r, d

    def __getattr__(self, name):
        return getattr(self.env, name)


def setup_model(
    env,
    hide_context,
    context_feature_args,
    default_sample_std_percentage,
    cfg,
    checkpoint_dir,
):
    num_contexts = 100
    contexts = sample_contexts(
        env,
        context_feature_args,
        num_contexts,
        default_sample_std_percentage=default_sample_std_percentage,
    )
    env_logger = None
    from carl.envs import CARLDmcWalkerEnv

    EnvCls = partial(
        eval(env),
        contexts=contexts,
        logger=env_logger,
        hide_context=hide_context,
    )
    env = EnvCls()
    eval_env = EnvCls()

    # This is intended for dmc and might cause issues with other envs!
    env = StateNormalizingWrapper(ActionLimitingWrapper(env, lower=-1+1e-6, upper=1-1e-6))
    eval_env = StateNormalizingWrapper(ActionLimitingWrapper(eval_env, lower=-1+1e-6, upper=1-1e-6))

    network_config = {"carl": {"dict_observation_space": False, "hide_context": True},
                      "network": {"width": 32}}
    from omegaconf import OmegaConf
    network_config = OmegaConf.create(network_config)
    func_pi = pi_func(network_config, env)
    func_q = q_func(network_config, env)

    # main function approximators
    pi = coax.Policy(func_pi, env, random_seed=cfg["seed"])
    q1 = coax.Q(
        func_q,
        env,
        action_preprocessor=pi.proba_dist.preprocess_variate,
        random_seed=cfg["seed"],
    )
    q2 = coax.Q(
        func_q,
        env,
        action_preprocessor=pi.proba_dist.preprocess_variate,
        random_seed=cfg["seed"],
    )

    # target network
    q1_targ = q1.copy()
    q2_targ = q2.copy()

    # experience tracer
    tracer = coax.reward_tracing.NStep(
        n=cfg["n_step"], gamma=cfg["gamma"], record_extra_info=True
    )
    buffer = coax.experience_replay.SimpleReplayBuffer(
        capacity=cfg["replay_capacity"], random_seed=cfg["seed"]
    )
    policy_regularizer = coax.regularizers.NStepEntropyRegularizer(
        pi, beta=cfg["alpha"] / tracer.n, gamma=tracer.gamma, n=[tracer.n]
    )

    qlearning1 = coax.td_learning.SoftClippedDoubleQLearning(
        q1,
        pi_targ_list=[pi],
        q_targ_list=[q1_targ, q2_targ],
        loss_function=coax.value_losses.mse,
        optimizer=optax.adam(cfg["learning_rate"]),
        policy_regularizer=policy_regularizer,
    )
    qlearning2 = coax.td_learning.SoftClippedDoubleQLearning(
        q2,
        pi_targ_list=[pi],
        q_targ_list=[q1_targ, q2_targ],
        loss_function=coax.value_losses.mse,
        optimizer=optax.adam(cfg["learning_rate"]),
        policy_regularizer=policy_regularizer,
    )
    soft_pg = coax.policy_objectives.SoftPG(
        pi,
        [q1_targ, q2_targ],
        optimizer=optax.adam(cfg["learning_rate"]),
        regularizer=coax.regularizers.NStepEntropyRegularizer(
            pi, beta=cfg["alpha"] / tracer.n, gamma=tracer.gamma, n=jnp.arange(tracer.n)
        ),
    )

    if checkpoint_dir:
        checkpoint_dir = str(checkpoint_dir)
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        pi, q1, q2, q1_targ, q2_targ, qlearning1, qlearning2, soft_pg = coax.utils.load(checkpoint)

    return pi, buffer, qlearning1, qlearning2, soft_pg, q1_targ, q2_targ, q1, env, eval_env


def eval_model(pi, eval_env, config):
    eval_reward = 0
    for i in range(100):
        done = False
        state = eval_env.reset()
        while not done:
            action, _ = pi(state)
            state, reward, done, _ = eval_env.step(action)
            eval_reward += reward
    return eval_reward / 100


def train_sac(
    env,
    hide_context,
    context_feature_args,
    default_sample_std_percentage,
    config,
    checkpoint_dir=None,
):
    pi, buffer, qlearning1, qlearning2, soft_pg, q1_targ, q2_targ, q1, env, eval_env = setup_model(
        env=env,
        cfg=config,
        checkpoint_dir=checkpoint_dir,
        hide_context=hide_context,
        context_feature_args=context_feature_args,
        default_sample_std_percentage=default_sample_std_percentage,
    )

    et = 0

    while et < config["steps"]:
        s = env.reset()

        for t in range(env.env.cutoff):
            a = pi(s)
            s_next, r, done, info = env.step(a)
            et += 1
            # trace rewards and add transition to replay buffer
            tracer.add(s, a, r, done)
            while tracer:
                buffer.add(tracer.pop())

            # learn
            if len(buffer) >= cfg["warmup_num_frames"]:
                transition_batch = buffer.sample(batch_size=config["batch_size"])

                metrics = {}

                # flip a coin to decide which of the q-functions to update
                qlearning = qlearning1 if jax.random.bernoulli(q1.rng) else qlearning2
                metrics.update(qlearning.update(transition_batch))

                # delayed policy updates
                if (
                        env.T >= config["pi_warmup_num_frames"]
                        and env.T % config["pi_update_freq"] == 0
                ):
                    metrics.update(soft_pg.update(transition_batch))

                env.record_metrics(metrics)

                # sync target networks
                q1_targ.soft_update(q1, tau=config["q_targ_tau"])
                q2_targ.soft_update(q2, tau=config["q_targ_tau"])

            if done:
                break

            s = s_next
            if et%config["eval_interval"]==0:
                if checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    agent = (pi, q1, q2, q1_targ, q2_targ, qlearning1, qlearning2, soft_pg)
                    coax.utils.dump(agent, path)
                eval_reward = eval_model(pi, eval_env, config)
                tune.report(mean_accuracy=eval_reward, current_config=config)


def run_experiment(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using " "Ray Client.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="results/experiments/pb2")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--outdir", type=str, help="Result directory")
    parser.add_argument("--env", type=str, help="Environment to optimize for")
    parser.add_argument("--hide_context", action="store_true")
    parser.add_argument("--default_sample_std_percentage", type=float, default=0.1)
    parser.add_argument("--context_feature", type=str, help="Context feature to adapt")
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("--eval_interval", type=int, default=1e4)

    args, unknown_args = parser.parse_known_args()
    local_dir = os.path.join(args.outdir, "ray")
    args.default_sample_std_percentage = 0.1
    args.context_feature_args = [args.context_feature]
    checkpoint_dir = args.checkpoint_dir

    # checkpoint_dir = Path(checkpoint_dir)
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # args.num_envs = 1
    if args.server_address:
        ray.util.connect(args.server_address)
    else:
        ray.init()

    print("current workdir:", os.getcwd())

    pbt = PB2(
        perturbation_interval=1,
        hyperparam_bounds={
            "learning_rate": [0.00001, 0.02],
            "gamma": [0.8, 0.999],
            "q_targ_tau": [0.00001, 0.01],
            "alpha": [0.01, 0.5]
        },
        log_config=True,
        require_attrs=True,
    )

    defaults = {
        "batch_size": 256,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "seed": 0,
        #TODO: add continuous ones to configurable
        "pi_warmup_num_frames": 7500,
        "q_targ_tau": 0.005,
        "pi_update_freq": 4,
        "steps": args.steps,
        "evaL_interval": args.eval_interval,
        "warmup_num_frames": 5000,
        "alpha": 0.2,
        "replay_capacity": 1e6,
        "n_step": 5,
    }

    analysis = tune.run(
        partial(
            train_sac,
            args.env,
            args.hide_context,
            args.context_feature_args,
            args.default_sample_std_percentage,
        ),
        name=args.name,
        scheduler=pbt,
        metric="mean_accuracy",
        mode="max",
        verbose=3,
        stop={
            "training_iteration": 250,
        },
        num_samples=8,
        fail_fast=True,
        # Search defaults from zoo overwritten with brax demo
        config=defaults,
        local_dir=local_dir,
        log_to_file=True,
    )

    all_dfs = analysis.trial_dataframes
    for i, (name, df) in enumerate(all_dfs.items()):
        fname = Path(os.path.join(args.outdir, f"trail_df_{i}_{name.strip('_')}.csv"))
        fname.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(fname)
    print("Best hyperparameters found were: ", analysis.best_config)
    ray.shutdown()


if __name__ == "__main__":
    run_experiment(sys.argv[1:])
