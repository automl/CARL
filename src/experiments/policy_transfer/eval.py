import os
import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from src.eval.eval_models import load_model, setup_env
from src.experiments.policy_transfer.policy_transfer import get_train_contexts_ll, gravities, get_train_contexts_ll_exp1
from src.context.sampling import get_default_context_and_bounds


class ChangeSeedCallback(object):
    def __init__(self, seeds):
        self.seeds = seeds
        self.idx = 0

    def __call__(self, locals, globals={}):
        env = locals["env"]
        done = locals["done"]
        if done:
            seed = int(self.seeds[self.idx])
            # print(f"changed seed to {seed}")
            if isinstance(env, DummyVecEnv):
                for e in env.envs:
                    e.seed(seed)
            else:
                env.seed(seed)
            self.idx += 1
            if self.idx == len(self.seeds):
                self.idx = 0


if __name__ == "__main__":
    envname = "CARLLunarLanderEnv"
    outdir_orig = "results/experiments/policytransfer/new"
    context_feature_name = "GRAVITY_Y"
    visibility_str = "visible"
    outdir = os.path.join(outdir_orig, envname, visibility_str, context_feature_name)
    contexts_train_fname = os.path.join(outdir_orig, envname, visibility_str, context_feature_name, "contexts_train.json")
    eval_data_fname = Path(outdir) / "eval_data.csv"
    figfname = Path(outdir) / "policy_transfer_lunar_lander_eval_comparison.png"
    is_exp0 = False
    collect_data = False if eval_data_fname.is_file() else True
    collect_data = True

    # create test contexts (train distribution)
    context_feature_key = "GRAVITY_Y"
    n_contexts = 100
    n_planets_exp1 = 20
    g_testplanets_exp1 = np.linspace(-25, 0, n_planets_exp1)
    g_testplanets_exp1[-1] = -1e-3  # TODO dont hardcode
    n_eval_eps = n_contexts
    eval_seeds = np.arange(0, n_eval_eps, dtype=int) + 13000
    env_default_context, env_bounds = get_default_context_and_bounds(env_name=envname)
    with open(contexts_train_fname, 'r') as file:
        contexts_train = json.load(file)
    if is_exp0:
        contexts_test = get_train_contexts_ll(gravities, context_feature_key, n_contexts, env_default_context)
    else:
        contexts_test = get_train_contexts_ll_exp1(n_contexts, env_default_context)

    planets_test_in = ["Mars", "Moon"]
    planets_test_out = ["Jupiter", "Neptune", "Pluto"]

    if collect_data:
        data = []
        model_fnames = glob.glob(os.path.join(outdir, "*", "model.zip"))
        # model_fnames = glob.glob(os.path.join(outdir, "DQN_1", "model.zip"))
        for model_fname in model_fnames:
            model_fname = Path(model_fname)
            # agent_seed_dir = model_fname.parts[-2]
            # seed = int(agent_seed_dir.split("_")[-1])
            print(model_fname)
            model, info = load_model(model_fname)

            # test contexts (train distribution)
            context_feature_ids = list(gravities.keys())
            if is_exp0:
                test_setup = {
                    "train\ncontexts": contexts_train,
                    "train\ndistribution": contexts_test,
                }
                for context_feature_id in context_feature_ids:
                    context = env_default_context.copy()
                    context[context_feature_key] = gravities[context_feature_id]
                    contexts = {0: context}
                    test_setup[context_feature_id] = contexts
            else:
                test_setup = {
                    "train\ncontexts": contexts_train,
                    "train\ndistribution": contexts_test,
                }
                for i, g in enumerate(g_testplanets_exp1):
                    context = env_default_context.copy()
                    context[context_feature_key] = g
                    contexts = {0: context}
                    context_feature_id = f"{g:.0f} m/s²"
                    if -1 < g < 0:
                        context_feature_id = f"{g:.3f} m/s²"
                    test_setup[context_feature_id] = contexts

            for context_feature_id, contexts in test_setup.items():
                # if context_feature_id == "Neptune":
                #     continue
                print(context_feature_id)
                env_kwargs = {"max_episode_length": 1000, 'high_gameover_penalty': True}
                env = setup_env(model_fname.parent, contexts=contexts, vec_env_class=None, wrappers=[Monitor], env_kwargs=env_kwargs)
                change_seed_cb = ChangeSeedCallback(seeds=eval_seeds)
                mean_reward, std_reward = evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=n_eval_eps,
                    return_episode_rewards=True,
                    callback=change_seed_cb
                )
                D = pd.DataFrame({
                    "planet": [context_feature_id] * n_eval_eps,
                    "ep_rew_mean": mean_reward,
                    "ep_rew_std": std_reward,
                    "ep_length": env.get_episode_lengths(),
                    "train_seed": [info['seed']] * n_eval_eps,
                })
                data.append(D)
        data = pd.concat(data)
        data.to_csv(eval_data_fname, sep=";")

    sns.set_context("paper")
    data = pd.read_csv(eval_data_fname, sep=";")
    train_contexts_key = '$\mathcal{I}_{train}$'
    data['planet'][data['planet'] == 'train\ncontexts'] = train_contexts_key
    data = data.rename(columns={'train\ncontexts': train_contexts_key})
    custom_dict = {
        train_contexts_key: 0,
        'train\ndistribution': 1,
        'Jupiter': 7,
        'Neptune': 6,
        'Earth': 5,
        'Mars': 4,
        'Moon': 3,
        'Pluto': 2
    }
    data = data.sort_values(by=['planet'], key=lambda x: x.map(custom_dict))
    data = data[data['planet'] != 'train\ndistribution']
    filter_by_ep_length = False
    plot_ep_length = False
    max_ep_length = 1000
    if filter_by_ep_length:
        data = data[data["ep_length"] < max_ep_length]
    palette = "colorblind"
    hue = 'train_seed'
    hue = None
    figsize = (5, 3)
    dpi = 250
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # ax = fig.add_subplot(111)
    if plot_ep_length:
        axes = fig.subplots(nrows=2, ncols=1, sharex=True)
    else:
        ax = fig.add_subplot(111)

    if plot_ep_length:
        ax = axes[0]
    ax = sns.violinplot(
        data=data,
        x="planet",
        y="ep_rew_mean",
        ax=ax,
        hue=hue,
        cut=0,
        scale='width',
        inner=None,
        split=True,
        linewidth=0.1,
        saturation=0.8,
        palette=palette
    )
    ax = sns.stripplot(
        data=data,
        x="planet",
        y="ep_rew_mean",
        ax=ax,
        hue=hue,
        size=1.5,
        edgecolors=[0.,0.,0.],
        linewidths=0,
        color='black',
        # palette=palette
    )
    ax.set_ylim(-10000, 500)
    ax.set_ylabel("mean reward")

    if plot_ep_length:
        ax.set_xlabel("")
        ax = axes[1]
        ax = sns.violinplot(
            data=data, x="planet", y="ep_length", ax=ax, hue=hue, cut=0, palette=palette, )
        # ax = sns.swarmplot(data=data, x="planet", y="ep_length", ax=ax, hue=hue, size=2, palette=palette)

    fig.set_tight_layout(True)
    fig.savefig(figfname, bbox_inches="tight")
    plt.show()

    df_train = data[data["planet"] == "train\ncontexts"]
    ep_rew_thresh = -5000
    df_train_bad = df_train[df_train["ep_rew_mean"] < ep_rew_thresh]
    unique, counts = np.unique(df_train_bad["Unnamed: 0"], return_counts=True)
    seeds = data['train_seed'].unique()
    n_seeds = len(seeds)
    gravities_train = [c["GRAVITY_Y"] for c in contexts_train.values()] * n_seeds
    failing_gravities = [gravities_train[i] for i in df_train_bad["Unnamed: 0"].to_list()]
    seed_list = df_train_bad["train_seed"]

    df_grav = pd.DataFrame({
        # "g_train": gravities_train,
        "g_fail": failing_gravities,
        "seed": seed_list
    })

    seed_list_train = []
    for seed in seeds:
        seed_list_train.extend([seed]*(len(gravities_train) // n_seeds))
    df_grav_train = pd.DataFrame({
        "g_train": gravities_train,
        "seed": seed_list_train
    })

    # fig = plt.figure(figsize=(6, 4))
    # ax = fig.add_subplot(111)
    # ax = sns.histplot(data=df_grav_train, x="g_train", bins=50, ax=ax, label="train", fill=False)
    # ax = sns.histplot(data=df_grav, x="g_fail", bins=50, ax=ax, hue=None, label="fail", multiple="dodge")
    # ax.legend()
    # ax.set_xlabel("gravity")
    # ax.set_ylabel("count")
    # fig.set_tight_layout(True)
    # plt.show()

