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
from scipy.interpolate import griddata

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
        patch_kwargs: Optional[Dict] = None,
        draw_frame: bool = False
):
    def update_colors(color, patch_kwargs: Dict, draw_frame: bool):
        if draw_frame:
            patch_kwargs["edgecolor"] = mplc.to_rgba(color)
            patch_kwargs["facecolor"] = (0., 0., 0., 0.)
            if "linewidth" not in patch_kwargs:
                patch_kwargs["linewidth"] = 4.
        else:
            patch_kwargs["color"] = color
            if "linewidth" not in patch_kwargs:
                patch_kwargs["linewidth"] = 0.
            if "zorder" not in patch_kwargs:
                patch_kwargs["zorder"] = 0
        return patch_kwargs

    if patch_kwargs is None:
        patch_kwargs = {}

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
    patch_kwargs = update_colors(color_extrapolation_single, patch_kwargs, draw_frame)
    patch = Rectangle(xy=xy, width=width, height=height, **patch_kwargs)
    # patch.set_alpha(None)
    patches.append(patch)

    xy = (cf0.lower, cf1.mid)
    height = cf1.upper - cf1.mid
    width = cf0.mid - cf0.lower
    patch_kwargs = update_colors(color_extrapolation_single, patch_kwargs, draw_frame)
    patch = Rectangle(xy=xy, width=width, height=height, **patch_kwargs)
    # patch.set_alpha(None)
    patches.append(patch)

    # Extrapolation along both factors
    xy = (cf0.mid, cf1.mid)
    height = cf1.upper - cf1.mid
    width = cf0.upper - cf0.mid
    patch_kwargs = update_colors(color_extrapolation_all, patch_kwargs, draw_frame)
    patch = Rectangle(xy=xy, width=width, height=height, **patch_kwargs)
    # patch.set_alpha(None)
    patches.append(patch)

    # Interpolation
    if mode == "A":
        xy = (cf0.lower, cf1.lower)
        height = cf1.mid - cf1.lower
        width = cf0.mid - cf0.lower
        patch_kwargs = update_colors(color_interpolation, patch_kwargs, draw_frame)
        patch = Rectangle(xy=xy, width=width, height=height, **patch_kwargs)
        # patch.set_alpha(None)
        patches.append(patch)
    elif mode == "B":
        xy = (cf0.lower, cf1.lower)
        width = cf0.mid - cf0.lower
        height = cf1.lower_constraint - cf1.lower
        patch_kwargs = update_colors(color_interpolation, patch_kwargs, draw_frame)
        patch = Rectangle(xy=xy, width=width, height=height, **patch_kwargs)
        # patch.set_alpha(None)
        patches.append(patch)

        xy = (cf0.lower, cf1.lower_constraint)
        width = cf0.lower_constraint - cf0.lower
        height = cf1.mid - cf1.lower_constraint
        patch_kwargs = update_colors(color_interpolation, patch_kwargs, draw_frame)
        patch = Rectangle(xy=xy, width=width, height=height, **patch_kwargs)
        # patch.set_alpha(None)
        patches.append(patch)

    # Combinatorial Interpolation
    if mode == "B":
        xy = (cf0.lower_constraint, cf1.lower_constraint)
        height = cf1.mid - cf1.lower_constraint
        width = cf0.mid - cf0.lower_constraint
        patch_kwargs = update_colors(color_interpolation_combinatorial, patch_kwargs, draw_frame)
        patch = Rectangle(xy=xy, width=width, height=height, **patch_kwargs)
        # patch.set_alpha(None)
        patches.append(patch)
    elif mode == "C":
        xy = (cf0.lower, cf1.lower)
        height = cf1.mid - cf1.lower
        width = cf0.mid - cf0.lower
        patch_kwargs = update_colors(color_interpolation_combinatorial, patch_kwargs, draw_frame)
        patch = Rectangle(xy=xy, width=width, height=height, **patch_kwargs)
        # patch.set_alpha(None)
        patches.append(patch)

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
    draw_points = False
    draw_mean_per_region = True
    plot_train = False

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
    # (mode, seed) -> context_dict
    contexts_LUT_fn = Path("tmp/contexts_evaluation_protocol_LUT.csv")
    contexts_LUT_fn.parent.mkdir(parents=True, exist_ok=True)
    index_col = ["mode", "seed", "context_distribution_type", "instance_id"]
    if contexts_LUT_fn.is_file():
        print("Load LUT")
        contexts_LUT = pd.read_csv(str(contexts_LUT_fn), index_col=index_col)
    else:
        print("Create LUT")
        contexts_LUT = []
        for mode in modes:
            for seed in seeds:
                context_dict = get_ep_contexts(env_name=env, n_contexts=n_contexts, seed=seed, mode=mode)
                for context_distribution_type, contexts in context_dict.items():
                    if type(contexts) == list and len(contexts) == 0:
                        continue
                    # contexts["context_distribution_type"] = [context_distribution_type] * len(contexts)
                    # contexts["mode"] = [mode] * len(contexts)
                    # contexts["seed"] = [seed] * len(contexts)
                    # contexts["instance_id"] = np.arange(0, len(contexts))
                    arrays = [
                        [mode] * len(contexts),
                        [seed] * len(contexts),
                        [context_distribution_type] * len(contexts),
                        np.arange(0, len(contexts))
                    ]
                    tuples = list(zip(*arrays))
                    index = pd.MultiIndex.from_tuples(tuples, names=index_col)
                    contexts.index = index
                    contexts_LUT.append(contexts)

        contexts_LUT = pd.concat(contexts_LUT)
        contexts_LUT.to_csv(str(contexts_LUT_fn))

    # Populate results with context feature values
    print("Get context feature values")
    def lookup(mode, seed, context_distribution_type, instance_id):
        context = contexts_LUT.loc[mode, seed, context_distribution_type, instance_id]
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

    print("Draw!")
    # Create figure
    figsize = (9, 3) if draw_mean_per_region else (18, 6)
    fig = plt.figure(figsize=figsize, dpi=300)
    nrows = 1
    axes = fig.subplots(nrows=nrows, ncols=n_protocols, sharex=True, sharey=True)

    context_distribution_types = results["context_distribution_type"].unique()
    groups = results.groupby("mode")
    cf0, cf1 = context_features
    xlim = (cf0.lower, cf0.upper)
    ylim = (cf1.lower, cf1.upper)

    index = 0  # 0 for mean, 1 for std  # TODO Rename to more verbose
    performances_list = []
    for i, (group_id, group_df) in enumerate(groups):
        performances = {}
        for context_distribution_type in context_distribution_types:
            if not plot_train and context_distribution_type == "train":
                continue
            sub_df = group_df[group_df["context_distribution_type"] == context_distribution_type]
            if len(sub_df) > 0:
                performance = np.mean(sub_df["episode_reward"]), np.std(sub_df["episode_reward"])
            else:
                performance = (np.nan, np.nan)
            performances[context_distribution_type] = performance
        performances_list.append(performances)

    perf = []
    for performances in performances_list:
        for v in performances.values():
            perf.append(v)
    perf = np.array(perf)
    perf_min = np.nanmin(perf)
    perf_max = np.nanmax(perf)
    perf_ptp = perf_max - perf_min

    def scale(x):
        return (x - perf_min) / perf_ptp

    for i, (group_id, group_df) in enumerate(groups):
        ax = axes[i]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        mode = group_id

        episode_reward_min = group_df["episode_reward"].min()
        episode_reward_max = group_df["episode_reward"].max()
        ptp = episode_reward_max - episode_reward_min

        performances = {}
        for context_distribution_type in context_distribution_types:
            sub_df = group_df[group_df["context_distribution_type"] == context_distribution_type]
            if len(sub_df) > 0:
                performance = np.mean(sub_df["episode_reward"]), np.std(sub_df["episode_reward"])
            else:
                performance = (np.nan, np.nan)
            performances[context_distribution_type] = performance

        if not plot_train:
            del_ids = group_df["context_distribution_type"] == "train"
            group_df = group_df[~del_ids]

        columns = context_feature_names + ["episode_reward", ]
        contexts_train = group_df[group_df["context_distribution_type"] == "train"][columns]
        contexts_ES = group_df[group_df["context_distribution_type"] == "test_extrapolation_single"][columns]
        contexts_EA = group_df[group_df["context_distribution_type"] == "test_extrapolation_all"][columns]
        contexts_I = group_df[group_df["context_distribution_type"] == "test_interpolation"][columns]
        contexts_IC = group_df[group_df["context_distribution_type"] == "test_interpolation_combinatorial"][columns]

        if draw_mean_per_region:
            color_T = cmap(scale(performances["train"][index]))
            color_I = cmap(scale(performances["test_interpolation"][index]))
            color_ES = cmap(scale(performances["test_extrapolation_single"][index]))
            color_EB = cmap(scale(performances["test_extrapolation_all"][index]))
            color_IC = cmap(scale(performances["test_interpolation_combinatorial"][index]))
            patches = get_patches(
                context_features=context_features,
                color_interpolation=color_I,
                color_extrapolation_single=color_ES,
                color_extrapolation_all=color_EB,
                color_interpolation_combinatorial=color_IC,
                patch_kwargs={"alpha": 1, "linewidth": 0, "zorder": 0},
                draw_frame=False,
            )
            for patch in patches:
                ax.add_patch(patch)
        else:
            colors = sns.color_palette("colorblind")
            color_T = colors[0]
            color_I = colors[1]
            color_ES = colors[2]
            color_EB = colors[3]
            color_IC = colors[4]

            ec_test = "black"
            markerfacecolor_alpha = 0.

            if draw_points:
                patch_kwargs = {"alpha": 1, "linewidth": 0, "zorder": 0}
                draw_frame = False
            else:
                patch_kwargs = {"linewidth": 3, "zorder": 1e6}
                draw_frame = True

            patches = get_patches(
                context_features=context_features,
                color_interpolation=color_I,
                color_extrapolation_single=color_ES,
                color_extrapolation_all=color_EB,
                color_interpolation_combinatorial=color_IC,
                patch_kwargs=patch_kwargs,
                draw_frame=draw_frame,
            )

            for patch in patches:
                ax.add_patch(patch)

            def scatter(ax, contexts):
                cols = contexts.columns
                context_feature_names = cols[:-1]
                performance_key = cols[-1]
                x = contexts[context_feature_names[0]]
                y = contexts[context_feature_names[1]]
                perf = contexts[performance_key].to_numpy()
                perf_scaled = scale(perf)
                c = cmap(perf_scaled)
                ax.scatter(x=x, y=y, c=c, alpha=0.5)
                return ax

            if draw_points:
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

            else:
                # Draw heatmap
                points = group_df[context_feature_names].to_numpy()
                values = group_df["episode_reward"].to_numpy()
                # target grid to interpolate to
                n_points = 300
                xi = np.linspace(group_df[context_feature_names[0]].min(), group_df[context_feature_names[0]].max(), n_points)
                yi = np.linspace(group_df[context_feature_names[1]].min(), group_df[context_feature_names[1]].max(), n_points)
                Xi, Yi = np.meshgrid(xi, yi)
                zi = griddata(points=points, values=values, xi=(Xi, Yi), method='linear')
                # ax = sns.heatmap(data=zi, vmin=episode_reward_min, vmax=episode_reward_max, cmap='viridis', ax=ax)
                extent = [xi[0], xi[-1], yi[0], yi[-1]]  # left right bottom top
                extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
                ax.imshow(zi, vmin=episode_reward_min, vmax=episode_reward_max, cmap='viridis', extent=extent, origin='lower')
                ax.set_aspect("auto")

        # Draw colorbar
        if draw_mean_per_region:
            norm = mpl.colors.Normalize(vmin=perf_min, vmax=perf_max)
        else:
            norm = mpl.colors.Normalize(vmin=episode_reward_min, vmax=episode_reward_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        colorbar = fig.colorbar(
            ax=ax, cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical',
            label='Episode Reward'
        )
        if not draw_mean_per_region:
            solved_threshold = get_solved_threshold(env_name=env)
            if solved_threshold is not None:
                colorbar.add_lines(levels=[solved_threshold], colors=["black"], linewidths=[2])



        # # Fix ticks
        # n_ticks = 5
        # xmin, xmax = xi[0], xi[-1]
        # ymin, ymax = yi[0], yi[-1]
        # xticks = np.linspace(xmin, xmax, n_ticks)
        # yticks = np.linspace(ymin, ymax, n_ticks)
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)

        # Add axis descriptions
        ax.set_xlabel(cf0.name)
        if i == 0:
            ax.set_ylabel(cf1.name)
        ax.set_title(mode)

    fig.set_tight_layout(True)
    plt.show()

