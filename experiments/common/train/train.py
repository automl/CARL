from functools import partial
import os
from xvfbwrapper import Xvfb
import configargparse
import yaml
import json
from typing import Dict, Union, Optional, Type, Callable, Tuple
import numpy as np
import gym
import importlib

import sys
import inspect
import warnings
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(os.getcwd())

import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import configure

# from classic_control import CARLMountainCarEnv
# importlib.reload(classic_control.meta_mountaincar)

import carl.envs
from carl.envs.carl_env import CARLEnv

import carl.utils.trial_logger
importlib.reload(carl.utils.trial_logger)
from carl.utils.trial_logger import TrialLogger

from carl.context.sampling import sample_contexts
from experiments.common.train.hyperparameter_processing import preprocess_hyperparams
from experiments.common.train.eval_callback import DACEvalCallback
from experiments.common.train.eval_policy import evaluate_policy
from experiments.common.utils.json_utils import lazy_json_dump
from experiments.evaluation_protocol.evaluation_protocol_utils import get_train_contexts
from experiments.common.train.policies.cgate import get_cgate_policy


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


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
        "--dont_add_agentseed_to_logdir",
        action="store_true",
        help="Don't add {agent}_{seed} to logdir but directly log into provided logdir."
    )

    parser.add_argument(
        "--default_sample_std_percentage",
        default=0.05,
        help="Standard deviation as percentage of mean",
        type=float
    )
    
    parser.add_argument(
        "--hide_context",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="If true, do not append the context to the state.",
        # from https://docs.python.org/3/library/argparse.html#nargs
        # '?'. One argument will be consumed from the command line if possible, and produced as a single item.
        # If no command-line argument is present, the value from default will be produced.
        # N ote that for optional arguments, there is an additional case - the option string is present but not followed
        # by a command-line argument. In this case the value from const will be produced.
    )

    parser.add_argument(
        "--hp_file",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "hyperparameters/ppo.yml")),
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

    parser.add_argument(
        "--build_outdir_from_args",
        action="store_true",
        help="If set, build output directory based on env name, default sample perecentage and visibility."
    )

    parser.add_argument(
        "--steps_min",
        type=str,
        default=1e4,
        help="Minimum number of steps for each hyperparameter configuration during (BO)HB optimization."
    )

    parser.add_argument(
        "--follow_evaluation_protocol",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="If true, follow evaluation protocol for context set creation. Overrides context_feature_args."
    )

    parser.add_argument(
        "--evaluation_protocol_mode",
        type=str,
        choices=["A", "B", "C"],
        help="Evaluation protocols from Kirk et al., 2021."
    )

    parser.add_argument(
        "--use_cgate",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="If true, use cGate architecture."
    )

    return parser


def get_contexts(args):
    if not args.context_file:
        if args.follow_evaluation_protocol:
            contexts = get_train_contexts(
                env_name=args.env, seed=args.seed, n_contexts=args.num_contexts, mode=args.evaluation_protocol_mode)
        else:
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


def get_hps_from_file(hp_fn: str, env_name: str):
    with open(hp_fn, "r") as f:
        hyperparams_dict = yaml.safe_load(f)
    hyperparams = hyperparams_dict[env_name]
    hyperparams, env_wrapper, normalize_kwargs = preprocess_hyperparams(hyperparams)
    return hyperparams, env_wrapper, normalize_kwargs


def set_hps(
        env_name: str,
        agent_name: str,
        hp_fn: Optional[str],
        use_cgate: bool = False,
        opt_hyperparams: Optional[Union[Dict, "Configuration"]] = None
):
    hyperparams = {}
    env_wrapper = None
    normalize_kwargs = None
    schedule_kwargs = None
    # TODO create hyperparameter files for other agents as well, no hardcoding here
    if hp_fn is not None and agent_name == "PPO":
        hyperparams, env_wrapper, normalize_kwargs = get_hps_from_file(hp_fn=hp_fn, env_name=env_name)

    if agent_name == "DDPG":
        hyperparams["policy"] = "MlpPolicy"

        if env_name == "CARLAnt":
            hyperparams = {"batch_size": 128, "learning_rate": 3e-05, "gamma": 0.99, "gae_lambda": 0.8, "ent_coef": 0.0,
                           "max_grad_norm": 1.0, "vf_coef": 1.0}
            post = {"batch_size": 128, "learning_rate": 0.00038113442133180797, "gamma": 0.887637734413147,
                    "gae_lambda": 0.800000011920929, "ent_coef": 0.0, "max_grad_norm": 1.0, "vf_coef": 1.0}
            hyperparams["policy"] = "MlpPolicy"
            schedule_kwargs["use_schedule"] = True
            schedule_kwargs["switching_point"] = 4
            schedule_kwargs["hyperparams_post_switch"] = post

        if env_name == "CARLPendulumEnv":
            hyperparams, env_wrapper, normalize_kwargs = get_hps_from_file(
                hp_fn=os.path.join(os.path.dirname(__file__),
                                   "hyperparameters/ddpg.yml"), env_name=env_name)

        hyperparams["n_envs"] = 1

    if agent_name == "A2C":
        hyperparams["policy"] = "MlpPolicy"

    if agent_name == "DQN":
        hyperparams["policy"] = "MlpPolicy"

        if env_name == "CARLLunarLanderEnv":
            hyperparams = {
                # "n_timesteps": 1e5,
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

        hyperparams["n_envs"] = 1
    if agent_name == "SAC":
        hyperparams["policy"] = "MlpPolicy"

        if env_name == "CARLBipedalWalkerEnv":
            hyperparams = {
                "policy": 'MlpPolicy',
                "learning_rate": 7.3e-4,
                "buffer_size": 300000,
                "batch_size": 256,
                "ent_coef": 'auto',
                "gamma": 0.98,
                "tau": 0.02,
                "train_freq": 64,
                "gradient_steps": 64,
                "learning_starts": 10000,
                "use_sde": True,
                "policy_kwargs": dict(log_std_init=-3, net_arch=[400, 300]),
            }
        hyperparams["n_envs"] = 1

    if opt_hyperparams is not None:
        for k in opt_hyperparams:
            hyperparams[k] = opt_hyperparams[k]

    if use_cgate:
        hyperparams["policy"] = get_cgate_policy(agent_name=agent_name)
        hyperparams["policy_kwargs"] = dict()

    return hyperparams, env_wrapper, normalize_kwargs, schedule_kwargs


def get_env(
        env_name,
        n_envs: int = 1,
        env_kwargs: Optional[Dict] = None,
        wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
        wrapper_kwargs=None,
        normalize_kwargs: Optional[Dict] = None,
        agent_cls: Optional = None,  # only important for eval env to appropriately wrap
        eval_seed: Optional[int] = None,  # env is seeded in agent
        return_vec_env: bool = True,
        vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
        return_eval_env: bool = False,
) -> Union[CARLEnv, Tuple[CARLEnv]]:
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    if env_kwargs is None:
        env_kwargs = {}
    EnvCls = partial(getattr(carl.envs, env_name), **env_kwargs)

    make_vec_env_kwargs = dict(wrapper_class=wrapper_class, vec_env_cls=vec_env_cls, wrapper_kwargs=wrapper_kwargs)

    # Wrap, Seed and Normalize Env
    if return_vec_env:
        env = make_vec_env(EnvCls, n_envs=n_envs, **make_vec_env_kwargs)
    else:
        env = EnvCls()
        if wrapper_class is not None:
            env = wrapper_class(env, **wrapper_kwargs)
    n_eval_envs = 1
    # Eval policy works with more than one eval envs, but the number of contexts/instances must be divisible
    # by the number of eval envs without rest in order to catch all instances.
    if return_eval_env:
        if return_vec_env:
            eval_env = make_vec_env(EnvCls, n_envs=n_eval_envs, **make_vec_env_kwargs)
        else:
            eval_env = EnvCls()
            if wrapper_class is not None:
                eval_env = wrapper_class(env, **wrapper_kwargs)
        if agent_cls is not None:
            eval_env = agent_cls._wrap_env(eval_env)
        else:
            warnings.warn("agent_cls is None. Should be provided for eval_env to ensure that the correct wrappers are used.")
        if eval_seed is not None:
            eval_env.seed(eval_seed)  # env is seeded in agent

    if normalize_kwargs is not None and normalize_kwargs["normalize"]:
        del normalize_kwargs["normalize"]
        env = VecNormalize(env, **normalize_kwargs)

        if return_eval_env:
            eval_normalize_kwargs = normalize_kwargs.copy()
            eval_normalize_kwargs["norm_reward"] = False
            eval_normalize_kwargs["training"] = False
            eval_env = VecNormalize(eval_env, **eval_normalize_kwargs)

    ret = env
    if return_eval_env:
        ret = (env, eval_env)
    return ret


def main(args, unknown_args, parser, opt_hyperparams: Optional[Union[Dict, "Configuration"]] = None):
    print(args)
    if args.hide_context is True and args.use_cgate:
        msg = "Skip run because hide_context is True and use_cgate is True. When using cGate, the context " \
              "is always visible. Set hide_context to False if you want to use cGate."
        print(msg)
        return None
    # Manipulate args
    if args.follow_evaluation_protocol:
        args.context_feature_args = []

    # Get Vec Env Class
    vec_env_cls_str = args.vec_env_cls
    if vec_env_cls_str == "DummyVecEnv":
        vec_env_cls = DummyVecEnv
    elif vec_env_cls_str == "SubprocVecEnv":
        vec_env_cls = SubprocVecEnv
    else:
        raise ValueError(f"{vec_env_cls_str} not supported.")

    # Setup IO
    # Virtual Display
    use_xvfb = args.use_xvfb
    if use_xvfb:
        vdisplay = Xvfb()
        vdisplay.start()
    # Output Directory
    if args.build_outdir_from_args:
        hide_context_dir_str = "contexthidden" if args.hide_context else "contextvisible"
        state_context_features_str = "changing" if args.state_context_features is not None else ""
        if args.follow_evaluation_protocol:
            postdirs = f"{args.env}/evaluation_protocol-mode{args.evaluation_protocol_mode}-{state_context_features_str}{hide_context_dir_str}"
        else:
            postdirs = f"{args.env}/{args.default_sample_std_percentage}_{state_context_features_str}{hide_context_dir_str}"
        args.outdir = os.path.join(args.outdir, postdirs)

    # Setup Logger
    logger = TrialLogger(
        args.outdir,
        parser=parser,
        trial_setup_args=args,
        add_context_feature_names_to_logdir=args.add_context_feature_names_to_logdir,
    )
    init_sb3_tensorboard = False
    sb_loggers = ["stdout", "csv"]
    if init_sb3_tensorboard:
        sb_loggers.append("tensorboard")
    stable_baselines_logger = configure(str(logger.logdir), sb_loggers)

    # Get Hyperparameters
    hyperparams, env_wrapper, normalize_kwargs, schedule_kwargs = set_hps(
        env_name=args.env,
        agent_name=args.agent,
        hp_fn=args.hp_file,
        opt_hyperparams=opt_hyperparams,
        use_cgate=args.use_cgate
    )
    hp_content = {
        "hyperparameters": hyperparams,
        "env_wrapper": env_wrapper,
        "normalize_kwargs": normalize_kwargs,
        "schedule_kwargs": schedule_kwargs
    }
    hp_fn = os.path.join(logger.logdir, "hyperparameters.json")
    lazy_json_dump(data=hp_content, filename=hp_fn)
    if "n_envs" in hyperparams:
        args.num_envs = hyperparams["n_envs"]
        del hyperparams["n_envs"]
    normalize = False
    if normalize_kwargs is not None and normalize_kwargs["normalize"]:
        normalize = True

    # Write Training Information
    logger.write_trial_setup()
    train_args_fname = os.path.join(logger.logdir, "trial_setup.json")
    lazy_json_dump(data=args.__dict__, filename=train_args_fname)

    # Get Contexts
    contexts = get_contexts(args)
    contexts_fname = os.path.join(logger.logdir, "contexts_train.json")
    lazy_json_dump(data=contexts, filename=contexts_fname)

    # Get Agent Class
    try:
        agent_cls = getattr(stable_baselines3, args.agent)
    except ValueError:
        print(f"{args.agent} is an unknown agent class. Please use a classname from stable baselines 3")

    # Get Environment Class
    env_logger = logger if vec_env_cls is not SubprocVecEnv else None
    env_kwargs = dict(
        contexts=contexts,
        logger=env_logger,
        hide_context=args.hide_context,
        scale_context_features=args.scale_context_features,
        state_context_features=args.state_context_features,
        dict_observation_space=args.use_cgate
    )
    env, eval_env = get_env(
        env_name=args.env,
        n_envs=args.num_envs,
        env_kwargs=env_kwargs,
        wrapper_class=env_wrapper,
        vec_env_cls=vec_env_cls,
        return_eval_env=True,
        normalize_kwargs=normalize_kwargs,
        agent_cls=agent_cls,
        eval_seed=args.seed
    )

    # Setup Callbacks
    # Eval callback actually records performance over all instances while progress writes performance of the last
    # episode(s) which can be a random set of instances.
    eval_callback = DACEvalCallback(
        eval_env=eval_env,
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
    chkp_cb = CheckpointCallback(save_freq=args.eval_freq, save_path=os.path.join(logger.logdir, "models"))
    if callbacks is None:
        callbacks = [chkp_cb]
    else:
        callbacks.append(chkp_cb)

    # Instantiate Agent
    model = agent_cls(env=env, verbose=1, seed=args.seed, **hyperparams)
    final_ep_mean_reward = None
    if schedule_kwargs is not None and schedule_kwargs["use_schedule"] == True:
        switching_point = schedule_kwargs["switching_point"]
        post = schedule_kwargs["hyperparams_post_switch"]
        for it in range(100):
            model.learn(1e6)
            switched = False
            if it >= switching_point and not switched:  # TODO do we still need this?
                model.learning_rate = post["learning_rate"]
                model.gamma = post["gamma"]
                model.ent_coef = post["ent_coef"]
                model.vf_coef = post["vf_coef"]
                model.gae_lambda = post["gae_lambda"]
                model.max_grad_norm = post["max_grad_norm"]
                switched = True
    else:
        # Train Agent
        model.learn(total_timesteps=args.steps, callback=callbacks)

    # Save Agent
    model.save(os.path.join(logger.logdir, "model.zip"))
    if normalize:
        model.get_vec_normalize_env().save(os.path.join(logger.logdir, "vecnormalize.pkl"))

    logdir = model.logger.get_dir()
    logfile = os.path.join(logdir, "progress.csv")

    # Evaluate Final Model
    try:
        episode_rewards, episode_lengths, episode_instances = evaluate_policy(
                model=model,
                env=eval_env,
                n_eval_episodes=args.num_contexts,
                deterministic=True,
                render=False,
                return_episode_rewards=True,
                warn=True,
        )
        final_ep_mean_reward = np.mean(episode_rewards)
    except Exception as e:
        print(e)

    if use_xvfb:
        vdisplay.stop()

    return final_ep_mean_reward


if __name__ == '__main__':
    import gym
    gym.logger.set_level(40)
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    ret = main(args, unknown_args, parser)
