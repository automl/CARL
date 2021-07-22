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
from functools import partial
from pathlib import Path
import os
import glob
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)  # go up twice
sys.path.insert(0, parentdir)
print(os.getcwd())
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from xvfbwrapper import Xvfb
import pandas as pd

from src.run_stablebaselines import get_parser
from src.trial_logger import TrialLogger
from src.context_sampler import get_default_context_and_bounds
from src.envs import *


g_earth = 9.80665  # m/s²

gravities = {
    "Earth": g_earth * 1,
    "Moon": g_earth * 0.166,
    "Mars": g_earth * 0.377,
    "Pluto": g_earth * 0.071,
    "Jupiter": g_earth * 2.36,
    "Neptune": g_earth * 1.12,
}

planets_train = ["Moon"]
planet_train = "Moon"
planets_test_in = ["Mars", "Pluto"]
planets_test_out = ["Jupiter", "Neptune"]

outdir = "results/experiments/policytransfer"


def define_setting(args):
    args.steps = 1e6
    env_default_context, env_bounds = get_default_context_and_bounds(env_name=args.env)
    env_default_context["GRAVITY"] = - gravities[planet_train]  # negative gravity! beware of coordinate systems
    contexts_train = {planet_train: env_default_context}

    return args, contexts_train


def train_env():
    vdisplay = Xvfb()
    vdisplay.start()

    parser = get_parser()
    args, unknown_args = parser.parse_known_args()

    # ==========================================================
    # experiment specific settings
    args, contexts = define_setting(args)
    # ==========================================================

    logger = TrialLogger(
        args.outdir,
        parser=parser,
        trial_setup_args=args,
        add_context_feature_names_to_logdir=args.add_context_feature_names_to_logdir
    )
    logger.write_trial_setup()

    # make meta-env
    EnvCls = partial(eval(args.env), contexts=contexts, logger=logger, hide_context=args.hide_context)
    env = make_vec_env(EnvCls, n_envs=args.num_envs)

    try:
        model = eval(args.agent)('MlpPolicy', env, verbose=1)  # TODO add agent_kwargs
    except ValueError:
        print(f"{args.agent} is an unknown agent class. Please use a classname from stable baselines 3")

    # model.set_logger(new_logger)
    model.learn(total_timesteps=args.steps)
    model.save(os.path.join(args.outdir, f"{args.agent}_{args.seed}", "model"))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    args, contexts_train = define_setting(args)

    model_fnames = glob.glob(os.path.join(args.outdir, args.env, "*", "model.zip"))

    n_eval_eps = 50
    title = "In vs Out Distribution"
    k_ep_rew_mean = "reward_mean"
    k_ep_rew_std = "reward_std"
    data = []
    for model_fname in model_fnames:
        model_fname = Path(model_fname)
        agenttype_seed_str = model_fname.parent.stem
        train_seed = int(agenttype_seed_str.split("_")[-1])

        planets_test = [planet_train] + planets_test_in + planets_test_out
        for planet in planets_test:
            print(f"Landing on {planet}.")
            env_default_context, env_bounds = get_default_context_and_bounds(env_name=args.env)
            env_default_context["GRAVITY"] = - gravities[planet]  # negative gravity! beware of coordinate systems
            contexts = {planet: env_default_context}

            env = MetaLunarLanderEnv(contexts=contexts)
            model = PPO.load(path=model_fname, env=env)
            mean_reward, std_reward = evaluate_policy(
                model,
                model.get_env(),
                n_eval_episodes=n_eval_eps,
                return_episode_rewards=True
            )
            if planet == planet_train:
                planet += "*"
            D = pd.DataFrame({
                "planet": [planet] * n_eval_eps,
                k_ep_rew_mean: mean_reward,
                k_ep_rew_std: std_reward,
                "train_seed": [train_seed] * n_eval_eps,
            })
            data.append(D)
    data = pd.concat(data)

    fig = plt.figure(figsize=(6, 4), dpi=200)
    axes = fig.subplots(nrows=2, ncols=1, sharex=True)

    ax = axes[0]
    ax = sns.boxplot(data=data, x="planet", y=k_ep_rew_mean, ax=ax, palette="colorblind")
    ax.set_xlabel("")

    ax = axes[1]
    ax = sns.boxplot(data=data, x="planet", y=k_ep_rew_std, ax=ax, palette="colorblind")

    fig.suptitle(title)

    fig.set_tight_layout(True)
    plt.show()





