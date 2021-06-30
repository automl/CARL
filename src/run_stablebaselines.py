import gym
import numpy as np
import importlib
import configargparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# from classic_control import MetaMountainCarEnv
# importlib.reload(classic_control.meta_mountaincar)
from gym.envs.classic_control import *
from gym.envs.box2d import LunarLander
from src.classic_control.meta_mountaincar import CustomMountainCarEnv
from src.classic_control.meta_mountaincarcontinuous import CustomMountainCarContinuousEnv
from src.classic_control import *
from src.box2d import *

import src.trial_logger
importlib.reload(src.trial_logger)
from src.trial_logger import TrialLogger

from src.context_sampler import sample_contexts


# TODO: what does this do? Do we even need it?
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
        default="MetaLunarLanderEnv",
        help="Environment",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="PPO",
        help="Stablebaselines3 agent",
    )

    parser.add_argument(
        "--num_contexts",
        type=int,
        default=100,
        help="Number of contexts to be sampled",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1e6,
        help="Number of training steps",
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()

    num_cpu = 4  # Number of processes to use
    # set up logger
    logger = TrialLogger(args.outdir, parser=parser, trial_setup_args=args)
    logger.write_trial_setup()

    # sample contexts using unknown args
    # TODO find good sample std, make sure it is a good default
    contexts = sample_contexts(args.env, unknown_args, args.num_contexts, default_sample_std=0.05)

    # make meta-env
    if args.env == "MetaMountainCarEnv":
        base_env = CustomMountainCarEnv()
    elif args.env == "MetaMountainCarContinuousEnv":
        base_env = CustomMountainCarContinuousEnv()
    elif args.env == "MetaLunarLanderEnv":
        base_env = LunarLander()
    else:
        try:
            base_env = eval(args.env[4:])()
        except ValueError:
            print(f"{args.env} not registered yet.")
    env = eval(args.env)(base_env, contexts, logger=logger)

    try:
        model = eval(args.agent)('MlpPolicy', env, verbose=1) # TODO add agent_kwargs
    except ValueError:
        print(f"{args.agent} is an unknown agent class. Please use a classname from stable baselines 3")

    # model.set_logger(new_logger)
    model.learn(total_timesteps=args.steps)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()

    # TODO add more cmdline arguments

    # TODO check logging of stable baselines
    # TODO check if every env properly rebuilds the observation space in _update_context

    # TODO create context changer for each method
    # TODO check if observation space is set correctly for every env (".env."):
    #   this is fine for acrobot/cartpole, but the other two change their obs spaces.
    #   I think this shouldn't happen --> let's dicuss
    # TODO put config into info object ?

    # TODO create requirements

    # ENVS
    # TODO add continuous mountain car as env -> Carolin [DONE]
    # TODO add lunar lander -> Carolin
    # TODO maybe add bipedal -> Carolin
    # TODO look at metacarracing -> Theresa, [Carolin]
    # TODO add ToadGAN -> Frederik

