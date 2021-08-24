from functools import partial
import os
import gym
import importlib
from xvfbwrapper import Xvfb
import configargparse
import yaml

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

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
        default="MetaPendulumEnv",
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
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "hyperparameter.yml")),
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

    return parser


# test render 🥲


def main(args, unknown_args, parser):
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

    logger.write_trial_setup()

    # sample contexts using unknown args
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

    env_logger = logger if vec_env_cls is not SubprocVecEnv else None
    # make meta-env
    EnvCls = partial(
        eval(args.env),
        contexts=contexts,
        logger=env_logger,
        hide_context=args.hide_context,
        scale_context_features=args.scale_context_features,
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


if __name__ == '__main__':
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    main(args, unknown_args, parser)
