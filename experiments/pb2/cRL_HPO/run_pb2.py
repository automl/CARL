import os
import sys
from functools import partial
import numpy as np
import yaml
from pathlib import Path

import ray
from ray import tune
from ray.tune.schedulers.pb2 import PB2

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.utils.hyperparameter_processing import preprocess_hyperparams
from src.train import get_parser
from src.context.sampling import sample_contexts


def setup_model(env, num_envs, hide_context, context_feature_args, default_sample_std_percentage, config, checkpoint_dir):
    hyperparams = {}
    env_wrapper = None
    num_contexts = 100
    contexts = sample_contexts(
            env,
            context_feature_args,
            num_contexts,
            default_sample_std_percentage=default_sample_std_percentage
        )
    env_logger = None
    from src.envs import CARLPendulumEnv, CARLBipedalWalkerEnv, CARLLunarLanderEnv
    EnvCls = partial(
        eval(env),
        contexts=contexts,
        logger=env_logger,
        hide_context=hide_context,
    )
    env = make_vec_env(EnvCls, n_envs=1, wrapper_class=env_wrapper)
    eval_env = make_vec_env(EnvCls, n_envs=1, wrapper_class=env_wrapper)

    if checkpoint_dir:
        checkpoint_dir = str(checkpoint_dir)
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model = PPO.load(checkpoint, env=env)
    else:
        model = PPO('MlpPolicy', env, **config)
    return model, eval_env


def eval_model(model, eval_env, config):
    eval_reward = 0
    for i in range(100):
        done = False
        state = eval_env.reset()
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = eval_env.step(action)
            eval_reward += reward
    return eval_reward / 100


def train_ppo(env, num_envs, hide_context, context_feature_args, default_sample_std_percentage, config, checkpoint_dir=None):
    model, eval_env = setup_model(
        env=env,
        num_envs=num_envs,
        config=config,
        checkpoint_dir=checkpoint_dir,
        hide_context=hide_context,
        context_feature_args=context_feature_args,
        default_sample_std_percentage=default_sample_std_percentage
    )
    model.learning_rate = config["learning_rate"]
    model.gamma = config["gamma"]
    #model.tau = config["tau"]
    model.ent_coef = config["ent_coef"]
    model.vf_coef = config["vf_coef"]
    model.gae_lambda = config["gae_lambda"]
    model.max_grad_norm = config["max_grad_norm"]

    for _ in range(100):
        model.learn(1e6)
        if checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            model.save(path)
        eval_reward = eval_model(model, eval_env, config)
        tune.report(
        mean_accuracy=eval_reward,
        current_config=config
    )


def run_experiment(args):
    parser = get_parser()
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
        "Ray Client.")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="results/experiments/pb2"
    )
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--outdir", type=str, help="Result directory")
    parser.add_argument("--env", type=str, help="Environment to optimize for")
    parser.add_argument("--hide_context", action="store_true")
    parser.add_argument("--default_sample_std_percentage", type=float, default=0.1)
    parser.add_argument("--context_feature", type=str, help="Context feature to adapt")

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
            'learning_rate': [0.00001, 0.02],
            'gamma': [0.8, 0.999],
            'gae_lambda': [0.8, 0.999],
            'ent_coef': [0.0, 0.5],
            'max_grad_norm': [0.0, 1.0],
            'vf_coef': [0.0, 1.0],
            #'tau': [0.0, 0.99]
        },
        log_config=True,
        require_attrs=True,
    )

    defaults = {
        'batch_size': 128,  # 1024,
        'learning_rate': 3e-5,
        'gamma': 0.99,  # 0.95,
    }

    analysis = tune.run(
        partial(
            train_ppo,
            args.env,
            args.num_envs,
            args.hide_context,
            args.context_feature_args,
            args.default_sample_std_percentage
        ),
        name=args.name,
        scheduler=pbt,
        metric="mean_accuracy",
        mode="max",
        verbose=3,
        stop={
            "training_iteration": 250,
            # "timesteps_total": 1e6,
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

if __name__ == '__main__':
    run_experiment(sys.argv[1:])
