"""
POLICY TRANSFER
=======================================================================

g = 9.80665 m/s²
train on moon = 0.166g

test policy transfer:
    in distribution:
        Mars (0.377g)
        Pluto (0.071g)
    out of distribution:
        Jupiter (2.36g)
        Neptune (1.12g)

https://nssdc.gsfc.nasa.gov/planetary/factsheet/planet_table_ratio.html
"""
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
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from xvfbwrapper import Xvfb
import pandas as pd
import json
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from functools import partial
import configparser

from src.train import get_parser, main
from src.context.sampling import get_default_context_and_bounds
from src.envs import CARLVehicleRacingEnv, CARLLunarLanderEnv
from src.envs.box2d.carl_vehicle_racing import RaceCar, AWDRaceCar, StreetCar, TukTuk, BusSmallTrailer, PARKING_GARAGE

# Experiment 1: LunarLander
g_earth = - 9.80665  # m/s², beware of coordinate systems

gravities = {
    "Jupiter": g_earth * 2.36,
    "Neptune": g_earth * 1.12,
    "Earth": g_earth * 1,
    "Mars": g_earth * 0.377,
    "Moon": g_earth * 0.166,
    "Pluto": g_earth * 0.071,
}

planets_train = ["Moon"]
planet_train = "Moon"
planets_test_in = ["Mars", "Pluto"]
planets_test_out = ["Jupiter", "Neptune"]

outdir = "results/experiments/policytransfer"

# Experiment 2: CARLVehicleRacingEnv
vehicle_train = "RaceCar"
vehicles = {
    "RaceCar": PARKING_GARAGE.index(RaceCar),
    "StreetCar": PARKING_GARAGE.index(StreetCar),
    "TukTuk": PARKING_GARAGE.index(TukTuk),
    "AWDRaceCar": PARKING_GARAGE.index(AWDRaceCar),
    "BusSmallTrailer": PARKING_GARAGE.index(BusSmallTrailer),
}


def get_contexts(env_name, context_feature_key, context_feature_mapping, context_feature_id):
    env_default_context, env_bounds = get_default_context_and_bounds(env_name=env_name)
    env_default_context[context_feature_key] = context_feature_mapping[context_feature_id]
    contexts = {context_feature_key: env_default_context}
    return contexts


def get_train_contexts_ll(gravities, context_feature_key, n_contexts, env_default_context):
    mean = gravities["Mars"]
    std = 1.45  # m/s²
    random_gravities = np.random.normal(loc=mean, scale=std, size=n_contexts)

    contexts_train = {i: env_default_context.copy() for i in range(n_contexts)}
    for i, (key, context) in enumerate(contexts_train.items()):
        context[context_feature_key] = random_gravities[i]
        contexts_train[key] = context
    return contexts_train


def get_uniform_intervals_exp1():
    return [(-20, -15), (-5, -1e-3)]


def get_train_contexts_ll_exp1(n_contexts, env_default_context):
    intervals = get_uniform_intervals_exp1()
    n_c_per_interval = n_contexts // len(intervals)

    contexts = {}
    for interval_idx, interval in enumerate(intervals):
        gravities = np.random.uniform(*interval, size=n_c_per_interval)
        for i in range(n_c_per_interval):
            context = env_default_context.copy()
            context["GRAVITY_Y"] = gravities[i]
            contexts[i + interval_idx * n_c_per_interval] = context
    return contexts


def define_setting(args):
    args.steps = 1e6  # use 1M steps
    # args.steps = 1000
    args.env = "CARLLunarLanderEnv"
    args.agent = "DQN"
    if args.env == "CARLLunarLanderEnv":
        context_feature_key = "GRAVITY_Y"
        context_feature_id = planet_train
        context_feature_mapping = gravities
    elif args.env == "CARLVehicleRacingEnv":
        context_feature_key = "VEHICLE"
        context_feature_id = vehicle_train
        context_feature_mapping = vehicles
        args.hide_context = True
    else:
        raise NotImplementedError
    args.outdir = os.path.join(outdir, "exp0", args.env, "hidden", context_feature_key)
    args.hide_context = True  # for hidden: set to true and set "visible" from string above to "hidden"
    args.state_context_features = ["GRAVITY_Y"]
    args.no_eval_callback = True

    env_default_context, env_bounds = get_default_context_and_bounds(env_name=args.env)
    env_default_context[context_feature_key] = context_feature_mapping[context_feature_id]
    contexts_train = {context_feature_key: env_default_context}

    if args.env == "CARLLunarLanderEnv":
        n_contexts = 100

        # # uniform interval [0.1, 0.5]
        # interval = (0.1, 0.5)
        # random_gravities = np.random.uniform(*interval, size=n_contexts)

        # normal distribution gMars, 1.45
        contexts_train = get_train_contexts_ll(gravities, context_feature_key, n_contexts, env_default_context)

    contexts_train_fn = Path(os.path.join(args.outdir), f"{args.agent}_{args.seed}", "contexts_train.json")  # sorry hacky
    contexts_train_fn = Path(os.path.join(args.outdir), "contexts_train.json")
    contexts_train_fn.parent.mkdir(parents=True, exist_ok=True)
    if args.context_file:
        with open(args.context_file, 'r') as file:
            contexts_train = json.load(file)

    args.context_file = str(contexts_train_fn)
    with open(contexts_train_fn, 'w') as file:
        json.dump(contexts_train, file, indent="\t")

    return args


def train_env(args, unknown_args, parser):
    # experiment specific settings
    args = define_setting(args)
    main(args, unknown_args, parser)


if __name__ == '__main__':
    vdisplay = Xvfb()
    vdisplay.start()

    import matplotlib.pyplot as plt
    import seaborn as sns

    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    args, contexts_train = define_setting(args)

    env_name = "CARLLunarLanderEnv"
    model_fnames = glob.glob(os.path.join(outdir, env_name, "*", "model.zip"))
    # model_fnames = ["tmp/test_logs/PPO_123456/model"]

    if env_name == "CARLLunarLanderEnv":
        env_class = CARLLunarLanderEnv
        context_feature_key = "GRAVITY"
        context_feature_id_train = planet_train
        planets_test = [planet_train] + planets_test_in + planets_test_out
        context_feature_ids = planets_test
        context_feature_mapping = gravities
    elif env_name == "CARLVehicleRacingEnv":
        env_class = CARLVehicleRacingEnv
        context_feature_key = "VEHICLE"
        context_feature_id_train = vehicle_train
        context_feature_ids = list(vehicles.keys())
        context_feature_mapping = vehicles
    else:
        raise NotImplementedError

    n_eval_eps = 10
    title = "In vs Out Distribution"
    k_ep_rew_mean = "reward_mean"
    k_ep_rew_std = "reward_std"
    data = []
    for model_fname in model_fnames:
        model_fname = Path(model_fname)
        agenttype_seed_str = model_fname.parent.stem
        train_seed = int(agenttype_seed_str.split("_")[-1])

        config = configparser.ConfigParser()
        config.read(str(model_fname.parent / "trial_setup.ini"))
        hide_context = config['DEFAULT'].get("hide_context", False)

        for context_feature_id in context_feature_ids:
            print(f">> Evaluating {env_name} trained on/with {context_feature_id_train} on/with {context_feature_id} (train seed {train_seed}).")

            env_default_context, env_bounds = get_default_context_and_bounds(env_name=env_name)
            env_default_context[context_feature_key] = context_feature_mapping[context_feature_id]
            contexts = {context_feature_key: env_default_context}

            EnvCls = partial(
                env_class,
                contexts=contexts,
                hide_context=hide_context,
            )
            # env = env_class(contexts=contexts)
            env = DummyVecEnv([EnvCls])
            vecnormstatefile = model_fname.parent / "vecnormalize.pkl"
            if vecnormstatefile.is_file():
                env = VecNormalize.load(str(vecnormstatefile), env)
            model = PPO.load(path=model_fname, env=env)
            mean_reward, std_reward = evaluate_policy(
                model,
                model.get_env(),
                n_eval_episodes=n_eval_eps,
                return_episode_rewards=True
            )
            if context_feature_id == context_feature_id_train:
                context_feature_id += "*"
            D = pd.DataFrame({
                "planet": [context_feature_id] * n_eval_eps,
                k_ep_rew_mean: mean_reward,
                k_ep_rew_std: std_reward,
                "train_seed": [train_seed] * n_eval_eps,
            })
            data.append(D)
    data = pd.concat(data)

    fig = plt.figure(figsize=(6, 8), dpi=200)
    axes = fig.subplots(nrows=2, ncols=1, sharex=True)

    ax = axes[0]
    ax = sns.boxplot(data=data, x="planet", y=k_ep_rew_mean, ax=ax, palette="colorblind")
    ax.set_xlabel("")

    ax = axes[1]
    ax = sns.boxplot(data=data, x="planet", y=k_ep_rew_std, ax=ax, palette="colorblind")

    fig.suptitle(title)

    fig.set_tight_layout(True)
    plt.show()





