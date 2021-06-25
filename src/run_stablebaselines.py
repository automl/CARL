import gym
import numpy as np
import importlib
import configargparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import classic_control.meta_mountaincar
importlib.reload(classic_control.meta_mountaincar)
from classic_control.meta_mountaincar import MetaMountainCarEnv, CustomMountainCarEnv
# from gym.envs.classic_control import MountainCarEnv

import logging
importlib.reload(logging)
from logging import TrialLogger


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def get_parser() -> configargparse.ArgumentParser:
    """
    Creates new argument parser for running baselines.

    Returns
    -------
    parser : argparse.ArgumentParser

    """
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.ConfigparserConfigFileParser
    )
    parser.add_argument(
        "--outdir", type=str, default="tmp/test_logs", help="Output directory"
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate policy on",
    )

    # parser.add_argument(
    #     "--seeds",
    #     nargs="+",
    #     type=int,
    #     default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     help="Seeds for evaluation",
    # )

    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Seed for reproducibility",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="MetaMountainCarEnv",
        help="Environment",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="PPO",
        help="Stablebaselines3 agent",
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()

    env_id = "MountainCar-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])


    # set up logger
    logger = TrialLogger(args.outdir, parser=parser, trial_setup_args=args)
    logger.write_trial_setup()

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    if args.env == "MetaMountainCarEnv":
        env = MetaMountainCarEnv(logger=logger)
    else:
        raise ValueError(f"{args.env} not registered yet.")

    if args.agent == "PPO":
        agent_class = PPO
    else:
        raise ValueError(f"{args.agent} not registered yet.")


    model = agent_class('MlpPolicy', env, verbose=1)  # TODO add agent_kwargs
    # model.set_logger(new_logger)
    model.learn(total_timesteps=25000)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()

    # TODO add cmdline arguments

    # TODO check logging of stable baselines
    # TODO CREATE LOGGING
    # TODO create context changer for each method
    # TODO check if observation space is set correctly for every env (".env.")
    # TODO put config into info object

    # TODO add spawner
    # TODO create requirements
