from functools import partial
import os
import gym
import importlib
import configargparse
import yaml

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# from classic_control import MetaMountainCarEnv
# importlib.reload(classic_control.meta_mountaincar)
from gym.envs.box2d.lunar_lander import LunarLander
from src.envs.classic_control.meta_mountaincar import CustomMountainCarEnv
from src.envs.classic_control.meta_mountaincarcontinuous import CustomMountainCarContinuousEnv

from src.envs import *
from src.envs.box2d.meta_vehicle_racing import PARKING_GARAGE

import src.trial_logger
importlib.reload(src.trial_logger)
from src.trial_logger import TrialLogger

from src.context_sampler import sample_contexts
from utils.hyperparameter_processing import preprocess_hyperparams


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
        "--agent",
        type=str,
        default="PPO",
        help="Stablebaselines3 agent",
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of vectorized environments",
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

    parser.add_argument(
        "--env",
        type=str,
        default="MetaLunarLanderEnv",
        help="Environment",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1e4,
        help="Number of steps after which to evaluate",
    )

    parser.add_argument(
        "--context_feature_args",
        type=str,
        default=[],
        nargs="+",
        help="Context feature args. Specify the name of a context feature and optionally name_mean and name_mean.",
    )

    parser.add_argument(
        "--add_context_feature_names_to_logdir",
        action="store_true",
        help="Creates logdir in following way: {logdir}/{context_feature_name_0}__{context_feature_name_1}/{agent}_{seed}"
    )

    parser.add_argument(
        "--default_sample_std_percentage",
        default=0.05,
        help="Standard deviation as percentage of mean",
        type=float
    )
    
    parser.add_argument(
        "--hide_context",
        action="store_true",
        help="Standard deviation as percentage of mean",
    )

    parser.add_argument(
        "--hp_file",
        type=str,
        default="hyperparameter.yml",
        help="YML file with hyperparameter",
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()

    num_cpu = 4  # Number of processes to use
    # set up logger
    logger = TrialLogger(
        args.outdir,
        parser=parser,
        trial_setup_args=args,
        add_context_feature_names_to_logdir=args.add_context_feature_names_to_logdir
    )
    logger.write_trial_setup()

    if args.hp_file is not None:
        with open(args.hp_file, "r") as f:
            hyperparams_dict = yaml.safe_load(f)
            hyperparams = hyperparams_dict[args.env]
            env_wrappers = None
            hyperparams, env_wrappers = preprocess_hyperparams(hyperparams)

    print(env_wrappers)
    # sample contexts using unknown args
    # TODO find good sample std, make sure it is a good default
    contexts = sample_contexts(
        args.env,
        args.context_feature_args,
        args.num_contexts,
        default_sample_std_percentage=args.default_sample_std_percentage
    )

    # make meta-env
    EnvCls = partial(eval(args.env), contexts=contexts, logger=logger, hide_context=args.hide_context)
    env = make_vec_env(EnvCls, n_envs=args.num_envs, wrapper_class=env_wrappers)
    eval_env = make_vec_env(EnvCls, n_envs=1, wrapper_class=env_wrappers)
    log_path = f"{args.outdir}/{args.agent}_{args.seed}"
    eval_callback = EvalCallback(eval_env, log_path=log_path, eval_freq=args.eval_freq,
                                 n_eval_episodes=args.num_contexts,
                                 deterministic=True, render=False)

    try:
        model = eval(args.agent)(env=env, verbose=1, **hyperparams)  # TODO add agent_kwargs
    except ValueError:
        print(f"{args.agent} is an unknown agent class. Please use a classname from stable baselines 3")

    # model.set_logger(new_logger)
    model.learn(total_timesteps=args.steps, callback=eval_callback)

    #obs = env.reset()
    #for _ in range(1000):
    #    action, _states = model.predict(obs)
    #    obs, rewards, dones, info = env.step(action)
    #    env.render()
    #env.close()

    # TODO add more cmdline arguments

    # TODO check logging of stable baselines
    # TODO check if every env properly rebuilds the observation space in _update_context

    # TODO create context changer for each method
    # TODO check if observation space is set correctly for every env (".env."):
    #   this is fine for acrobot/cartpole, but the other two change their obs spaces.
    #   I think this shouldn't happen --> let's dicuss
    # TODO put config into info object ?

    # TODO create requirements

    # TODO if a default context is 0, the sampled/altered context will be also zero (because of 0 std)

    # ENVS
    # TODO add continuous mountain car as env -> Carolin [DONE]
    # TODO add lunar lander -> Carolin
    # TODO maybe add bipedal -> Carolin
    # TODO look at metacarracing -> Theresa, [Carolin]
    # TODO add ToadGAN -> Frederik

