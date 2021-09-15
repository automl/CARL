from functools import partial
import os
import gym
import importlib
from xvfbwrapper import Xvfb
import configargparse
import yaml
import json

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(os.getcwd())

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, DDPG, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# from classic_control import CARLMountainCarEnv
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
from src.utils.hyperparameter_processing import preprocess_hyperparams


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
        default="CARLPendulumEnv",
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
        "--state_context_features",
        default=None,
        nargs="+",
        help="Specifiy which context features should be added to state if hide_context is False. "
             "None: Add all context features to state. "
             "'changing_context_features' (str): Add only the ones changing in the contexts to state. "
             "List[str]: Add those to the state."
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
        default=os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "train/hyperparameters/hyperparameters_ppo.yml")),
        help="YML file with hyperparameter",
    )

    parser.add_argument(
        "--scale_context_features",
        type=str,
        default="no",
        choices=["no", "by_mean", "by_default"],
        help="Scale context features before appending them to the observations. 'no' means no scaling. 'by_mean' scales"
             " the context features by the mean of the training contexts features. 'by_default' scales the context "
             "features by the default context features which are assumend to be the mean of the context feature "
             "distribution."
    )

    parser.add_argument(
        "--context_file",
        type=str,
        default=None,
        help="Context file(name) containing all train contexts in json format as Dict[int, Dict[str, Any]]."
    )

    parser.add_argument(
        "--vec_env_cls",
        type=str,
        default="DummyVecEnv",
        choices=["DummyVecEnv", "SubprocVecEnv"]
    )

    parser.add_argument(
        "--use_xvfb",
        action="store_true"
    )

    parser.add_argument(
        "--no_eval_callback",
        action="store_true"
    )

    return parser


def get_contexts(args):
    if not args.context_file:
        contexts = sample_contexts(
            args.env,
            args.context_feature_args,
            args.num_contexts,
            default_sample_std_percentage=args.default_sample_std_percentage
        )
    else:
        with open(args.context_file, 'r') as file:
            contexts = json.load(file)
    return contexts


def main(args, unknown_args, parser):
    # set_random_seed(args.seed)  # TODO discuss seeding

    vec_env_cls_str = args.vec_env_cls
    if vec_env_cls_str == "DummyVecEnv":
        vec_env_cls = DummyVecEnv
    elif vec_env_cls_str == "SubprocVecEnv":
        vec_env_cls = SubprocVecEnv
    else:
        raise ValueError(f"{vec_env_cls_str} not supported.")
    use_xvfb = args.use_xvfb

    if use_xvfb:
        vdisplay = Xvfb()
        vdisplay.start()

    # set up logger
    logger = TrialLogger(
        args.outdir,
        parser=parser,
        trial_setup_args=args,
        add_context_feature_names_to_logdir=args.add_context_feature_names_to_logdir,
        init_sb3_tensorboard=False  # set to False if using SubprocVecEnv
    )

    hyperparams = {}
    env_wrapper = None
    normalize = False
    normalize_kwargs = {}
    # TODO create hyperparameter files for other agents as well, no hardcoding here
    if args.hp_file is not None and args.agent == "PPO":
        with open(args.hp_file, "r") as f:
            hyperparams_dict = yaml.safe_load(f)
        hyperparams = hyperparams_dict[args.env]
        if "n_envs" in hyperparams:
            args.num_envs = hyperparams["n_envs"]
        hyperparams, env_wrapper, normalize, normalize_kwargs = preprocess_hyperparams(hyperparams)

    if args.agent == "DDPG":
        hyperparams["policy"] = "MlpPolicy"
        args.num_envs = 1

    if args.agent == "A2C":
        hyperparams["policy"] = "MlpPolicy"

    if args.agent == "DQN":
        hyperparams["policy"] = "MlpPolicy"
        args.num_envs = 1

        if args.env == "CARLLunarLanderEnv":
            hyperparams = {
                #"n_timesteps": 1e5,
                "policy": 'MlpPolicy',
                "learning_rate": 6.3e-4,
                "batch_size": 128,
                "buffer_size": 50000,
                "learning_starts": 0,
                "gamma": 0.99,
                "target_update_interval": 250,
                "train_freq": 4,
                "gradient_steps": -1,
                "exploration_fraction": 0.12,
                "exploration_final_eps": 0.1,
                "policy_kwargs": dict(net_arch=[256, 256])
            }

    logger.write_trial_setup()

    # TODO make less hacky
    train_args_fname = os.path.join(logger.logdir, "trial_setup.json")
    with open(train_args_fname, 'w') as file:
        json.dump(args.__dict__, file, indent="\t")

    contexts = get_contexts(args)

    env_logger = logger if vec_env_cls is not SubprocVecEnv else None
    # make meta-env
    EnvCls = partial(
        eval(args.env),
        contexts=contexts,
        logger=env_logger,
        hide_context=args.hide_context,
        scale_context_features=args.scale_context_features,
        state_context_features=args.state_context_features
        # max_episode_length=1000   # set in meta env
    )
    env_cls = eval(args.env)
    env_kwargs = dict(
        contexts=contexts,
        # logger=logger,  # no logger because of SubprocVecEnv
        hide_context=args.hide_context,
        scale_context_features=args.scale_context_features
    )
    env = make_vec_env(EnvCls, n_envs=args.num_envs, wrapper_class=env_wrapper, vec_env_cls=vec_env_cls)
    eval_env = make_vec_env(EnvCls, n_envs=1, wrapper_class=env_wrapper, vec_env_cls=vec_env_cls)
    if normalize:
        env = VecNormalize(env, **normalize_kwargs)
        eval_normalize_kwargs = normalize_kwargs.copy()
        eval_normalize_kwargs["norm_reward"] = False
        eval_normalize_kwargs["training"] = False
        eval_env = VecNormalize(eval_env, **eval_normalize_kwargs)

    # Setup callbacks

    # eval callback actually records performance over all instances while progress writes performance of the last episode(s)
    # which can be a random set of instances
    eval_callback = EvalCallback(
        eval_env,
        log_path=logger.logdir,
        eval_freq=1, #args.eval_freq,
        n_eval_episodes=args.num_contexts,
        deterministic=True,
        render=False
    )
    callbacks = [eval_callback]
    everynstep_callback = EveryNTimesteps(n_steps=args.eval_freq, callback=eval_callback)
    callbacks = [everynstep_callback]
    if args.no_eval_callback:
        callbacks = None

    try:
        agent_cls = eval(args.agent)
    except ValueError:
        print(f"{args.agent} is an unknown agent class. Please use a classname from stable baselines 3")
    model = agent_cls(env=env, verbose=1, **hyperparams)  # TODO add agent_kwargs

    model.set_logger(logger.stable_baselines_logger)
    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(os.path.join(logger.logdir, "model.zip"))
    if normalize:
        model.get_vec_normalize_env().save(os.path.join(logger.logdir, "vecnormalize.pkl"))

    if use_xvfb:
        vdisplay.stop()


if __name__ == '__main__':
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    main(args, unknown_args, parser)
