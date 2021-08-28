import os
import sys
# sys.path.append(os.path.dirname(os.getcwd()))
# sys.path.append(os.getcwd())
import argparse
from functools import partial
import numpy as np
import yaml
from pathlib import Path

import ray
from ray import tune
from ray.tune.schedulers.pb2 import PB2

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.utils.hyperparameter_processing import preprocess_hyperparams
from src.run_stablebaselines import main, get_parser
from src.context_sampler import sample_contexts


def setup_model(env, hp_file, num_envs, hide_context, context_feature_args, default_sample_std_percentage, config, checkpoint_dir):
    with open(hp_file, "r") as f:
        hyperparams_dict = yaml.safe_load(f)
    hyperparams = hyperparams_dict[env]
    hyperparams, env_wrapper, normalize, normalize_kwargs = preprocess_hyperparams(hyperparams)

    num_contexts = 100
    contexts = sample_contexts(
            env,
            context_feature_args,
            num_contexts,
            default_sample_std_percentage=default_sample_std_percentage
        )
    env_logger = None
    from src.envs import CARLAnt
    EnvCls = partial(
        eval(env),
        contexts=contexts,
        logger=env_logger,
        hide_context=hide_context,
    )
    env = make_vec_env(EnvCls, n_envs=num_envs, wrapper_class=env_wrapper)
    eval_env = make_vec_env(EnvCls, n_envs=1, wrapper_class=env_wrapper)
    if normalize:
        env = VecNormalize(env, **normalize_kwargs)
        eval_normalize_kwargs = normalize_kwargs.copy()
        eval_normalize_kwargs["norm_reward"] = False
        eval_normalize_kwargs["training"] = False
        eval_env = VecNormalize(eval_env, **eval_normalize_kwargs)

    if checkpoint_dir:
        print(checkpoint_dir)
        checkpoint_dir = str(checkpoint_dir)
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model = PPO.load(checkpoint, env=env)
    else:
        model = PPO('MlpPolicy', env, **config)
    return model, eval_env


def eval_model(model, eval_env, config):
    eval_reward = []
    for i in range(100):
        done = False
        state = eval_env.reset()
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = eval_env.step(action)
            eval_reward.append(reward)
    return tune.report(
        mean_accuracy=np.mean(eval_reward),
        current_config=config
    )


def train_ppo(env, hp_file, num_envs, hide_context, context_feature_args, default_sample_std_percentage, config, checkpoint_dir=None):
    model, eval_env = setup_model(
        env=env,
        hp_file=hp_file,
        num_envs=num_envs,
        config=config,
        checkpoint_dir=checkpoint_dir,
        hide_context=hide_context,
        context_feature_args=context_feature_args,
        default_sample_std_percentage=default_sample_std_percentage
    )
    model.learning_rate = config["learning_rate"]
    model.gamma = config["gamma"]
    model.ent_coef = config["ent_coef"]
    model.vf_coef = config["vf_coef"]
    model.gae_lambda = config["gae_lambda"]
    model.max_grad_norm = config["max_grad_norm"]

    model.learn(4096)
    if checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        model.save(path)
    ret = eval_model(model, eval_env, config)
    print(ret)
    return ret


if __name__ == "__main__":
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

    args, unknown_args = parser.parse_known_args()
    args.env = "CARLAnt"
    args.outdir = os.path.join(os.getcwd(), "results/experiments/pb2", args.env)
    local_dir = os.path.join(args.outdir, "ray")
    args.hide_context = True
    args.default_sample_std_percentage = 0.25
    args.context_feature_args = ['friction']
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
        perturbation_interval=20,
        hyperparam_bounds={
            'learning_rate': [0.0001, 0.02],
            'gamma': [0.8, 0.999],
            'gae_lambda': [0.8, 0.999],
            'ent_coef': [0.0, 0.5],
            'max_grad_norm': [0.0, 1.0],
            'vf_coef': [0.0, 1.0],
        },
        log_config=True,
        require_attrs=True,
    )

    # default hyperparameters from hyperparameter.yml
    # HPs found for stable baselines' PPO on pybullet Ant
    defaults = {
        'batch_size': 128,  # 1024,
        'learning_rate': 3e-5,
        'n_steps': 512,  # args.num_envs*1024,
        'gamma': 0.99,  # 0.95,
        'gae_lambda': 0.9,  # 0.95,
        'n_epochs': 20,  # 4,
        'ent_coef': 0.0,  # 0.01,
        'sde_sample_freq': 4,
        'max_grad_norm': 0.5,
        'vf_coef': 0.5,
    }

    analysis = tune.run(
        partial(
            train_ppo,
            args.env,
            args.hp_file,
            args.num_envs,
            args.hide_context,
            args.context_feature_args,
            args.default_sample_std_percentage
        ),
        name="pb2",
        scheduler=pbt,
        metric="mean_accuracy",
        mode="max",
        verbose=False,
        stop={
            "training_iteration": 5e3,
        },
        num_samples=8,
        fail_fast=True,
        # Search defaults from zoo overwritten with brax demo
        config=defaults,
        local_dir=local_dir,
        log_to_file=True,
    )

    print("Best hyperparameters found were: ", analysis.best_config)
