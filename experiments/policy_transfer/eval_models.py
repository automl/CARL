from pathlib import Path
import os
import glob
import sys
import inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)  # go up twice
sys.path.insert(0, parentdir)
print(os.getcwd())
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import json
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG, PPO, A2C, DQN
from functools import partial
import configparser

from experiments.common.train.train import get_parser
from carl.context.sampling import sample_contexts
from carl.envs import *


def cf_args_str_to_list(context_feature_args):
    s = context_feature_args
    l = [i.strip("[]' ") for i in s.split(",")]
    return l


# TODO build args


def load_model(model_fname):
    model_fname = Path(model_fname)

    # parser = configargparse.ConfigparserConfigFileParser()
    # parser = configargparse.ArgParser(
    #     config_file_parser_class=configargparse.ConfigparserConfigFileParser
    # )
    path = model_fname.parent
    if "models" in str(path):
        path = Path(path).parent
    config_fname = path / "trial_setup.ini"

    # parser._config_file_parser = configargparse.ConfigparserConfigFileParser()
    # with open(config_fname, 'r') as file:
    #     config_file_contents = file.read()
    #     # config = parser.parse(config_file_contents)
    # print(config_file_contents)
    # args, unknown_args = parser.parse_known_args(args=None, config_file_contents=config_file_contents)
    # # # with open(config_fname, 'r') as file:
    # # #     config = parser.parse(file)
    # # print(args, config_file_contents)
    # print(args)

    config = configparser.ConfigParser()
    config.read(str(config_fname))
    hide_context = config["DEFAULT"].getboolean("hide_context", False)
    train_seed = config["DEFAULT"]["seed"]
    agent_str = config["DEFAULT"]["agent"]
    context_feature_args = config["DEFAULT"]["context_feature_args"]  # TODO test this
    context_feature_args = cf_args_str_to_list(context_feature_args)
    context_features = [
        n for n in context_feature_args if "std" not in n and "mean" not in n
    ]
    default_sample_std_percentage = config["DEFAULT"].getfloat(
        "default_sample_std_percentage"
    )

    config_fname_json = model_fname.parent / "trial_setup.json"
    if config_fname_json.is_file():
        with open(config_fname_json, "r") as file:
            config_json = json.load(file)
        hide_context = config_json["hide_context"]
        train_seed = config_json["seed"]

    agent = eval(agent_str)
    model = agent.load(path=model_fname)

    info = {"seed": train_seed, "context_features": context_features}

    return model, info


def setup_env(path, contexts=None, wrappers=None, vec_env_class=None, env_kwargs={}):
    if "models" in str(path):
        path = Path(path).parent
    config_fname = Path(path) / "trial_setup.ini"

    config = configparser.ConfigParser()
    config.read(str(config_fname))
    hide_context = config["DEFAULT"].getboolean("hide_context", False)
    train_seed = config["DEFAULT"]["seed"]
    env_str = config["DEFAULT"]["env"]
    agent_str = config["DEFAULT"]["agent"]
    context_feature_args = config["DEFAULT"]["context_feature_args"]  # TODO test this
    context_feature_args = cf_args_str_to_list(context_feature_args)
    context_features = [
        n for n in context_feature_args if "std" not in n and "mean" not in n
    ]
    default_sample_std_percentage = config["DEFAULT"].getfloat(
        "default_sample_std_percentage"
    )
    context_file = config["DEFAULT"].get("context_file")
    if context_file is None:
        context_file = "contexts_train.json"
    env_class = eval(env_str)

    config_fname_json = Path(path) / "trial_setup.json"
    if config_fname_json.is_file():
        with open(config_fname_json, "r") as file:
            config_json = json.load(file)
        hide_context = config_json["hide_context"]
        state_context_features = config_json["state_context_features"]

    if contexts is None:
        if context_file is None:
            contexts = sample_contexts(
                env_str,
                context_feature_args,
                num_contexts,
                default_sample_std_percentage=default_sample_std_percentage,
            )
        else:
            context_file = Path(path) / context_file
            with open(context_file, "r") as file:
                contexts = json.load(file)

    EnvCls = partial(
        env_class,
        contexts=contexts,
        hide_context=hide_context,
        state_context_features=state_context_features,
        **env_kwargs,
    )

    def create_env_fn():
        env = EnvCls()
        if wrappers is not None:
            for wrapper in wrappers:
                env = wrapper(env)
        return env

    if vec_env_class is not None:
        env = vec_env_class([lambda: create_env_fn()])
    else:
        env = create_env_fn()

    vecnormstatefile = Path(path) / "vecnormalize.pkl"
    if vecnormstatefile.is_file():
        env = VecNormalize.load(str(vecnormstatefile), env)

    return env


if __name__ == "__main__":
    parser = get_parser()

    env_name = "CARLLunarLanderEnv"
    outdir = "results/experiments/policytransfer"
    outdir = os.path.join(outdir, env_name)
    outdir = "results/singlecontextfeature_0.5_hidecontext/box2d/MetaBipedalWalkerEnv"
    outdir = (
        "results/base_vs_context/box2d/CARLLunarLanderEnv/0.5_changingcontextvisible"
    )
    n_eval_eps = 10
    num_contexts = 100
    model_fnames = glob.glob(os.path.join(outdir, "*", "*", "models", "*.zip"))
    model_fnames = []
    for root, dirs, filenames in os.walk(outdir):
        for filename in filenames:
            if "rl_model" in filename:
                model_fnames.append(os.path.join(root, filename))
    model_fnames = [m for m in model_fnames if "model" in m]

    k_ep_rew_mean = "ep_rew_mean"
    k_ep_rew_std = "ep_rew_std"

    data = []
    for i, model_fname in enumerate(model_fnames):
        msg = f"Eval {i+1}/{len(model_fnames)}: {model_fname}."
        print(msg)
        model_fname = Path(model_fname)
        step = -1
        if "rl_model" in model_fname.stem:
            step = int(model_fname.stem.split("_")[-2])
        env = setup_env(
            path=model_fname.parent, contexts=None, vec_env_class=DummyVecEnv
        )
        model, info = load_model(model_fname)
        train_seed = info["seed"]
        context_features = info["context_features"]
        mean_reward, std_reward = evaluate_policy(
            model,
            env,  # model.get_env(),
            n_eval_episodes=n_eval_eps,
            return_episode_rewards=True,
        )
        D = pd.Series(
            {
                k_ep_rew_mean: np.mean(mean_reward),
                k_ep_rew_std: np.mean(std_reward),
                "train_seed": train_seed,  # [train_seed] * n_eval_eps,
                "model_fname": model_fname,
                # "context_features": context_features,
                "step": step,  # [step] * n_eval_eps,
                "n_episodes": n_eval_eps,
            }
        )
        data.append(D)

    if len(model_fnames) > 1:
        save_path = os.path.commonpath(model_fnames)
    else:
        p = model_fnames[0]
        save_path = p.split("DQN")[0]  # TODO make dynamic
    save_path = Path(save_path) / "eval_train.csv"
    df = pd.DataFrame(data)
    df.to_csv(save_path)
