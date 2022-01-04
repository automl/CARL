import pandas as pd
from typing import Union, Dict, List, Optional
from pathlib import Path
import os
import glob
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mplc
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.eval.gather_data import extract_info
from src.experiments.train_on_protocol import get_context_features, get_ep_contexts
from src.experiments.evaluation_protocol import ContextFeature


def gather_results(
        path: Union[str, Path],
        eval_file_id: Union[str, Path] = "eval/evaluation_protocol/*.npz",
        trial_setup_fn: str = "trial_setup.json"
):
    dirs = glob.glob(os.path.join(path, "**", trial_setup_fn), recursive=True)

    data = []
    for result_dir in dirs:
        result_dir = Path(result_dir).parent
        # agent
        # seed
        # context feature args
        info = extract_info(path=result_dir, info_fn=trial_setup_fn)
        mode = info["evaluation_protocol_mode"]
        eval_fns = glob.glob(str(result_dir / eval_file_id), recursive=True)
        for eval_fn in eval_fns:
            eval_fn = Path(eval_fn)
            D = None
            if not eval_fn.is_file():
                print(f"Eval fn {eval_fn} does not exist. Skip.")
                continue

            eval_data = np.load(str(eval_fn))
            timesteps = eval_data["timesteps"]
            episode_lengths = eval_data["ep_lengths"]
            instance_ids = np.squeeze(eval_data["episode_instances"])
            episode_rewards = eval_data["results"]

            key = None
            if "test_id" in eval_data:
                key = "test_id"
            elif "context_distribution_type" in eval_data:
                key = "context_distribution_type"

            context_distribution_type = eval_data[key] if key is not None else None

            timesteps = np.concatenate([timesteps[:, np.newaxis]] * episode_rewards.shape[-1], axis=1)

            timesteps = timesteps.flatten()
            instance_ids = instance_ids.flatten()
            instance_ids = instance_ids.astype(dtype=np.int)
            episode_rewards = episode_rewards.flatten()
            episode_lengths = episode_lengths.flatten()

            n = len(timesteps)

            D = pd.DataFrame({
                "step": timesteps,
                "instance_id": instance_ids,
                "episode_reward": episode_rewards,
                "episode_length": episode_lengths,
                "context_distribution_type": context_distribution_type,
                "mode": mode
            })

            # merge info and results
            n = len(D)
            for k, v in info.items():
                D[k] = [v] * n

            if D is not None:
                data.append(D)

    if data:
        data = pd.concat(data)

    return data


def get_patches(
        context_features: List[ContextFeature],
        color_interpolation: Optional[Union[str, List[float], np.array]] = None,
        color_extrapolation_single: Optional[Union[str, List[float], np.array]] = None,
        color_extrapolation_all: Optional[Union[str, List[float], np.array]] = None,
        color_interpolation_combinatorial: Optional[Union[str, List[float], np.array]] = None,
        patch_kwargs: Optional[Dict] = None
):
    if patch_kwargs is None:
        patch_kwargs = {}
    if "zorder" not in patch_kwargs:
        patch_kwargs["zorder"] = 0
    if "linewidth" not in patch_kwargs:
        patch_kwargs["linewidth"] = 0.

    colors = sns.color_palette("colorblind")
    color_T = colors[0]
    color_I = colors[1]
    color_ES = colors[2]
    color_EB = colors[3]
    color_IC = colors[4]

    if color_interpolation is None:
        color_interpolation = color_I
    if color_extrapolation_single is None:
        color_extrapolation_single = color_ES
    if color_extrapolation_all is None:
        color_extrapolation_all = color_EB
    if color_interpolation_combinatorial is None:
        color_interpolation_combinatorial = color_IC

    cf0, cf1 = context_features

    # Draw Quadrants
    patches = []

    # Extrapolation along single factors, Q_ES
    xy = (cf0.mid, cf1.lower)
    width = cf0.upper - cf0.mid
    height = cf1.mid - cf1.lower
    Q_ES = Rectangle(xy=xy, width=width, height=height, color=color_extrapolation_single, **patch_kwargs)
    patches.append(Q_ES)

    xy = (cf0.lower, cf1.mid)
    height = cf1.upper - cf1.mid
    width = cf0.mid - cf0.lower
    Q_ES = Rectangle(xy=xy, width=width, height=height, color=color_extrapolation_single, **patch_kwargs)
    patches.append(Q_ES)

    # Extrapolation along both factors
    xy = (cf0.mid, cf1.mid)
    height = cf1.upper - cf1.mid
    width = cf0.upper - cf0.mid
    Q_EB = Rectangle(xy=xy, width=width, height=height, color=color_extrapolation_all, **patch_kwargs)
    patches.append(Q_EB)

    # Interpolation
    if mode == "A":
        xy = (cf0.lower, cf1.lower)
        height = cf1.mid - cf1.lower
        width = cf0.mid - cf0.lower
        Q_I = Rectangle(xy=xy, width=width, height=height, color=color_interpolation, **patch_kwargs)
        patches.append(Q_I)
    elif mode == "B":
        xy = (cf0.lower, cf1.lower)
        width = cf0.mid - cf0.lower
        height = cf1.lower_constraint - cf1.lower
        Q_I = Rectangle(xy=xy, width=width, height=height, color=color_interpolation, **patch_kwargs)
        patches.append(Q_I)

        xy = (cf0.lower, cf1.lower_constraint)
        width = cf0.lower_constraint - cf0.lower
        height = cf1.mid - cf1.lower_constraint
        Q_I = Rectangle(xy=xy, width=width, height=height, color=color_interpolation, **patch_kwargs)
        patches.append(Q_I)

    # Combinatorial Interpolation
    if mode == "B":
        xy = (cf0.lower_constraint, cf1.lower_constraint)
        height = cf1.mid - cf1.lower_constraint
        width = cf0.mid - cf0.lower_constraint
        Q_IC = Rectangle(xy=xy, width=width, height=height, color=color_interpolation_combinatorial, **patch_kwargs)
        patches.append(Q_IC)
    elif mode == "C":
        xy = (cf0.lower, cf1.lower)
        height = cf1.mid - cf1.lower
        width = cf0.mid - cf0.lower
        Q_IC = Rectangle(xy=xy, width=width, height=height, color=color_interpolation_combinatorial, **patch_kwargs)
        patches.append(Q_IC)

    return patches


def get_solved_threshold(env_name):
    thresh = None
    if env_name == "CARLCartPoleEnv":
        thresh = 195
    elif env_name == "CARLPendulumEnv":
        thresh = -175
    return thresh


if __name__ == '__main__':
    path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/evaluation_protocol/base_vs_context/classic_control/CARLCartPoleEnv"
    # path = "/home/benjamin/Dokumente/code/tmp/CARL/src/tmp/test_logs/CARLCartPoleEnv"
    results = gather_results(path=path)
    
    cmap = cm.get_cmap("viridis")

    assert results["env"].nunique() == 1
    env = results["env"].iloc[0]

    context_features = get_context_features(env_name=env)
    context_feature_names = [cf.name for cf in context_features]

    seeds = results["seed"].unique()
    modes = results["mode"].unique()
    n_protocols = len(modes)
    n_contexts = 100  # TODO no hardcoding

    # Create LUT
    print("Create LUT")
    # (mode, seed) -> context_dict
    contexts_LUT = {}
    for mode in modes:
        for seed in seeds:
            context_dict = get_ep_contexts(env_name=env, n_contexts=n_contexts, seed=seed, mode=mode)
            contexts_LUT[(mode, seed)] = context_dict

    # Populate results with context feature values
    print("Get context feature values")

    def lookup(mode, seed, context_distribution_type, instance_id):
        context_dict = contexts_LUT[(mode, seed)]
        contexts = context_dict[context_distribution_type]
        context = contexts.iloc[instance_id]
        return context

    def apply_func(row):
        return lookup(
            mode=row["mode"], seed=row["seed"], context_distribution_type=row["context_distribution_type"],
            instance_id=row["instance_id"]
        )
    ret = results.apply(apply_func, axis=1)

    # Match indices to be able to concat
    ret.index = results.index
    results = pd.concat([results, ret], axis=1)

    # Create figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    axes = fig.subplots(nrows=1, ncols=n_protocols, sharex=True, sharey=True)

    results.sort_values("mode", inplace=True)
    groups = results.groupby("mode")
    cf0, cf1 = context_features
    xlim = (cf0.lower, cf0.upper)
    ylim = (cf1.lower, cf1.upper)

    for i, (group_id, group_df) in enumerate(groups):
        ax = axes[i]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        mode = modes[i]

        episode_reward_min = group_df["episode_reward"].min()
        episode_reward_max = group_df["episode_reward"].max()
        ptp = episode_reward_max - episode_reward_min
        def scale(x):
            return (x - episode_reward_min) / ptp
        context_distribution_types = group_df["context_distribution_type"].unique()

        performances = {}
        for context_distribution_type in context_distribution_types:
            sub_df = group_df[group_df["context_distribution_type"] == context_distribution_type]
            performance = np.mean(sub_df["episode_reward"]), np.std(sub_df["episode_reward"])
            performances[context_distribution_type] = performance

        # TODO match performance of instance to actual point

        colors = sns.color_palette("colorblind")
        color_T = colors[0]
        color_I = colors[1]
        color_ES = colors[2]
        color_EB = colors[3]
        color_IC = colors[4]

        ec_test = "black"
        markerfacecolor_alpha = 0.

        patches = get_patches(
            context_features=context_features,
            color_interpolation=color_I,
            color_extrapolation_single=color_ES,
            color_extrapolation_all=color_EB,
            color_interpolation_combinatorial=color_IC,
            patch_kwargs={"alpha": 0.3}
        )

        for patch in patches:
            ax.add_patch(patch)

        columns = context_feature_names + ["episode_reward",]

        contexts_train = group_df[group_df["context_distribution_type"] == "train"][columns]
        contexts_ES = group_df[group_df["context_distribution_type"] == "test_extrapolation_single"][columns]
        contexts_EA = group_df[group_df["context_distribution_type"] == "test_extrapolation_all"][columns]
        contexts_I = group_df[group_df["context_distribution_type"] == "test_interpolation"][columns]
        contexts_IC = group_df[group_df["context_distribution_type"] == "test_interpolation_combinatorial"][columns]

        def scatter(ax, contexts):
            cols = contexts.columns
            context_feature_names = cols[:-1]
            performance_key = cols[-1]
            x = contexts[context_feature_names[0]]
            y = contexts[context_feature_names[1]]
            perf = contexts[performance_key].to_numpy()
            perf_scaled = scale(perf)
            c = cmap(perf_scaled)
            ax.scatter(x=x, y=y, c=c)
            return ax

        # Plot train context
        if len(contexts_train) > 0:
            ax = scatter(ax, contexts_train)
        # Extrapolation single
        if len(contexts_ES) > 0:
            ax = scatter(ax, contexts_ES)
        # Extrapolation all factors
        if len(contexts_EA) > 0:
            ax = scatter(ax, contexts_EA)
        # Interpolation (Train Distribution)
        if len(contexts_I) > 0:
            ax = scatter(ax, contexts_I)
        # Combinatorial Interpolation
        if len(contexts_IC) > 0:
            ax = scatter(ax, contexts_IC)

        # Draw colorbar
        norm = mpl.colors.Normalize(vmin=episode_reward_min, vmax=episode_reward_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        colorbar = fig.colorbar(
            ax=ax, cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical',
            label='Episode Reward'
        )
        solved_threshold = get_solved_threshold(env_name=env)
        if solved_threshold is not None:
            colorbar.add_lines(levels=[solved_threshold], colors=["black"], linewidths=[2])

        # # Plot train context
        # if len(contexts_train) > 0:
        #     ax = sns.scatterplot(data=contexts_train, x=cf0.name, y=cf1.name, color=color_T, ax=ax, edgecolor=color_T)
        # # Extrapolation single
        # if len(contexts_ES) > 0:
        #     ax = sns.scatterplot(data=contexts_ES, x=cf0.name, y=cf1.name,
        #                      color=mplc.to_rgba(color_ES, markerfacecolor_alpha), ax=ax, edgecolor=ec_test)
        # # Extrapolation all factors
        # if len(contexts_EA) > 0:
        #     ax = sns.scatterplot(data=contexts_EA, x=cf0.name, y=cf1.name,
        #                      color=mplc.to_rgba(color_EB, markerfacecolor_alpha), ax=ax, edgecolor=ec_test)
        # # Interpolation (Train Distribution)
        # if len(contexts_I) > 0:
        #     ax = sns.scatterplot(data=contexts_I, x=cf0.name, y=cf1.name,
        #                          color=mplc.to_rgba(color_I, markerfacecolor_alpha), ax=ax, edgecolor=ec_test)
        # # Combinatorial Interpolation
        # if len(contexts_IC) > 0:
        #     ax = sns.scatterplot(data=contexts_IC, x=cf0.name, y=cf1.name,
        #                          color=mplc.to_rgba(color_IC, markerfacecolor_alpha), ax=ax, edgecolor=ec_test)

        # Add axis descriptions
        ax.set_xlabel(cf0.name)
        if i == 0:
            ax.set_ylabel(cf1.name)
        ax.set_title(mode)

    fig.set_tight_layout(True)
    plt.show()

