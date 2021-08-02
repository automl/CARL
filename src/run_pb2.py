import argparse
from functools import partial
import numpy as np

import ray
from ray import tune
from ray.tune.schedulers.pb2 import PB2

from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env

def train_ppo(env, hp_file, num_envs, config):
    with open(hp_file, "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        hyperparams = hyperparams_dict[env]
        hyperparams, env_wrappers = preprocess_hyperparams(hyperparams)

    EnvCls = partial(eval(env), contexts=None)
    env = make_vec_env(EnvCls, n_envs=num_envs, wrapper_class=env_wrappers)
    model = PPO(env=env, verbose=1, **config)
    model.learn(2048)
    eval_env = make_vec_env(EnvCls, n_envs=1, wrapper_class=env_wrappers)
    eval_reward = []
    for i in range(100):
        done = False
        state = eval_env.reset()
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = eval_env.step(action)
            eval_reward.append(reward)
    tune.report(mean_accuray=np.mean(eval_reward),
                current_config=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default=MetaAnt, help="Environment to optimize hyperparameters for")
    parser.add_argument(
        "--hp_file", default="hyperparameter.yml", type=str
    )
    parser.add_argument(
        "--num_envs", type=int, default=os.cpu_count() or 1
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
        "Ray Client.")
    args, _ = parser.parse_known_args()
    if args.server_address:
        ray.util.connect(args.server_address)
    else:
        ray.init()

    pbt = PB2(
        perturbation_interval=20,
        hyperparam_bounds={
            # hyperparameter bounds.
            'batch_size': [32, 64, 128, 256, 512, 1024, 2048],
            'learning_rate': lambda: np.random.uniform(0.0001, 0.02),
            'gamma': lambda: np.random.uniform(0.8, 0.99),
            'gae_lambda': lambda: np.random.uniform(0.8, 0.99),
            'n_epochs': [4, 8, 16, 32],
            'ent_coef': lambda: np.random.uniform(0.0, 0.5),
            'sde_sample_freq': [3, 4, 5, 6, 7, 8],
            'max_grad_norm': lambda: np.random.uniform(0.0, 1.0),
            'vf_coef': lambda: np.random.uniform(0.0, 1.0),
            'clip_range': lambda: np.random.uniform(0.0, 0.8)
        })

    analysis = tune.run(
        partial(train_ppo, args.env. args.hp_file, args.num_envs),
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
        config={
            'batch_size': 1024,
            'learning_rate': 3e-4,
            'n_steps': args.num_envs*batch_size,
            'gamma': 0.95,
            'gae_lambda': 0.95,
            'n_epochs': 4,
            'ent_coef': 0.01,
            'sde_sample_freq': 4,
            'max_grad_norm': 0.5,
            'vf_coef': 0.5,
            'clip_range': 0.3
        })

    print("Best hyperparameters found were: ", analysis.best_config)