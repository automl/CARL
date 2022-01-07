import pandas as pd
from typing import Union, Dict, List, Optional
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mplc
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from src.experiments.evaluation_protocol.evaluation_protocol_utils import create_ep_contexts_LUT, \
    read_ep_contexts_LUT, gather_ep_results
from src.experiments.evaluation_protocol.evaluation_protocol_experiment_definitions import get_context_features, get_solved_threshold
from src.experiments.evaluation_protocol.evaluation_protocol import ContextFeature


def get_ep_mplpatches(
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


def add_colorbar_to_ax(vmin, vmax, cmap, label):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    colorbar = fig.colorbar(
        ax=ax, cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical',
        label=label
    )
    return colorbar


def get_agg_index(agg_per_region: str = "mean"):
    if agg_per_region == "mean":
        index = 0
    elif agg_per_region == "std":
        index = 1
    else:
        raise ValueError
    return index


def get_agg_minmax(
        results: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], agg_per_region: str = "mean"):
    index = get_agg_index(agg_per_region)

    if type(results) == pd.core.groupby.generic.DataFrameGroupBy:
        groups = results
    else:
        groups = results.groupby("mode")

    performances_list = []
    for i, (group_id, group_df) in enumerate(groups):
        performances = {}
        context_distribution_types = group_df["context_distribution_type"].unique()
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
            perf.append(v[index])
    perf = np.array(perf)
    perf_min = np.nanmin(perf)
    perf_max = np.nanmax(perf)

    return perf_min, perf_max


if __name__ == '__main__':
    path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/evaluation_protocol/base_vs_context/classic_control/CARLCartPoleEnv"
    draw_points = False
    draw_agg_per_region = True
    agg_per_region = "mean"
    plot_train = False

    results = gather_ep_results(path=path)
    
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
        contexts_LUT = read_ep_contexts_LUT(contexts_LUT_fn)
    else:
        print("Create LUT")
        contexts_LUT = create_ep_contexts_LUT(
            env_name=env, n_contexts=n_contexts, modes=modes, seeds=seeds, contexts_LUT_fn=contexts_LUT_fn)

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

    if draw_agg_per_region:
        perf_min, perf_max = get_agg_minmax(
            results.groupby(["context_visible", "mode"]), agg_per_region=agg_per_region)
        perf_ptp = perf_max - perf_min

    maingroups = results.groupby("context_visible")
    for visibility, maingroup_df in maingroups:
        print("Draw!")
        # Create figure
        figsize = (9, 3) if draw_agg_per_region else (18, 6)
        fig = plt.figure(figsize=figsize, dpi=300)
        nrows = 1
        axes = fig.subplots(nrows=nrows, ncols=n_protocols, sharex=True, sharey=True)

        context_distribution_types = maingroup_df["context_distribution_type"].unique()
        groups = maingroup_df.groupby("mode")
        cf0, cf1 = context_features
        xlim = (cf0.lower, cf0.upper)
        ylim = (cf1.lower, cf1.upper)

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

            if draw_agg_per_region:
                index = get_agg_index(agg_per_region=agg_per_region)
                color_T = cmap(scale(performances["train"][index]))
                color_I = cmap(scale(performances["test_interpolation"][index]))
                color_ES = cmap(scale(performances["test_extrapolation_single"][index]))
                color_EB = cmap(scale(performances["test_extrapolation_all"][index]))
                color_IC = cmap(scale(performances["test_interpolation_combinatorial"][index]))
                patches = get_ep_mplpatches(
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

                patches = get_ep_mplpatches(
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
            colorbar_label = "Episode Reward"
            if draw_agg_per_region: # and i == len(groups) - 1:
                colorbar = add_colorbar_to_ax(perf_min, perf_max, cmap, colorbar_label)
                if i != len(groups) - 1:
                    colorbar.remove()
            else:
                colorbar = add_colorbar_to_ax(episode_reward_min, episode_reward_max, cmap, colorbar_label)
                solved_threshold = get_solved_threshold(env_name=env)
                if solved_threshold is not None:
                    colorbar.add_lines(levels=[solved_threshold], colors=["black"], linewidths=[2])

            # Add axis descriptions
            ax.set_xlabel(cf0.name)
            if i == 0:
                ax.set_ylabel(cf1.name)
            ax.set_title(mode)

        fig.set_tight_layout(True)
        plt.show()
        
        fig_fname = Path(f"figures/results_env-{env}_visibility-{visibility}.png")
        fig_fname.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_fname, bbox_inches="tight")

