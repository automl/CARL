import numpy as np
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import seaborn as sns
import pandas as pd
from typing import Union, Dict
from mpl_toolkits.axes_grid1 import make_axes_locatable

from experiments.common.eval.gather_data import gather_results


def plot_hue(group_df, key_huegroup, ax, xname, yname, colors: Union[str, Dict]):
    groups_sub = group_df.groupby(key_huegroup)
    n = len(groups_sub)
    if type(colors) == str:
        colors = sns.color_palette(color_palette_name, n)
    legend_handles = []
    labels = []
    for j, (df_id, subset_group_df) in enumerate(groups_sub):
        n_seeds = subset_group_df['seed'].nunique()
        msg = f"{plot_id}, {group_id}, {df_id}: n_seeds={n_seeds}"
        print(msg)
        color = list(colors.values())[j]
        if df_id == default_name:
            color = color_default_context
        else:
            color = colors[df_id]
        ax = sns.lineplot(data=subset_group_df, x=xname, y=yname, ax=ax, color=color, marker='', hue=None)
        legend_handles.append(Line2D([0], [0], color=color))
        label = df_id
        labels.append(label)
    xmin = group_df['step'].min()
    xmax = group_df['step'].max()
    xlims = (xmin, xmax)
    ax.set_xlim(*xlims)
    return ax, labels, legend_handles


def sort_legend_handles(labels, legend_handles, default_name):
    if default_name in labels:
        idx = labels.index(default_name)
        name_item = labels.pop(idx)
        labels.insert(0, name_item)
        handle_item = legend_handles.pop(idx)
        legend_handles.insert(0, handle_item)
    return labels, legend_handles


if __name__ == '__main__':
    path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/rerun2/base_vs_context/classic_control/CARLPendulumEnv"
    path2 = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/compounding/base_vs_context/classic_control/CARLPendulumEnv"
    results = gather_results(path=path)
    results2 = gather_results(path=path2)
    results = pd.concat([results, results2])
    del results2

    experiment = "compoundingn"
    paperversion = True
    logx = False

    key_plotgroup = ["context_visible", "agent"]
    key_axgroup = "context_variation_magnitude"
    key_huegroup = "context_feature_args"

    legend_title = "varying context feature"

    sns.set_style("whitegrid")
    xname = "step"
    yname = "episode_reward"
    default_name = "None"  # identifier for environment with standard context

    color_palette_name = "colorblind"
    color_default_context = "black"

    figsize = (4, 3)
    figsize_c = (2.5, 3)  # figsize for compounding variations
    labelfontsize = 12
    ticklabelsize = 10
    legendfontsize = 10
    titlefontsize = 12

    if "context_feature_args" in results.columns:
        # Get number of context features
        results["n_context_features"] = results["context_feature_args"].apply(len)
        results = results.sort_values(by="n_context_features")

        # Convert cf args list to string
        def cfargs_to_str(x):
            return ", ".join([str(el) for el in x])
        results["context_feature_args"] = results["context_feature_args"].apply(cfargs_to_str)

    # split
    # ids = results["context_feature_args"].apply(lambda x: "," in x)
    # results = results[~ids]
    # compounding_results = results[ids]

    env_names = results['env'].unique()
    if len(env_names) > 1:
        raise NotImplementedError("Try to plot across different envs.")
    env_name = env_names[0]
    plotgroups = results.groupby(by=key_plotgroup)
    for (plot_id, plot_df) in plotgroups:
        fig = plt.figure(figsize=figsize, dpi=200)
        nrows = 1
        ncols = plot_df[key_axgroup].nunique()
        axes = fig.subplots(nrows=nrows, ncols=ncols, sharey=True)

        # legendtitle
        title = None
        xlabel = None
        ylabel = None
        groups = plot_df.groupby(key_axgroup)
        xmin = plot_df[xname].min()
        xmax = plot_df[xname].max()
        xlims = (xmin, xmax)
        # ymin = plot_df[yname].min()
        # ymax = plot_df[yname].max()
        # ylims = (ymin, ymax)

        if key_huegroup is None:
            colors = color_palette_name
        else:
            hues = plot_df[key_huegroup].unique()
            colors = sns.color_palette(color_palette_name, len(hues))
            colors = {k: v for k, v in zip(hues, colors)}

        for i, (group_id, group_df) in enumerate(groups):
            if type(axes) == list or type(axes) == np.ndarray:
                ax = axes[i]
            else:
                ax = axes

            ids = group_df["context_feature_args"].apply(lambda x: "," in x)
            df_c = group_df[ids]
            group_df = group_df[~ids]

            df_c = df_c.append(group_df[group_df["context_feature_args"] == "None"])
            df_c = df_c.append(group_df[group_df["context_feature_args"] == "m"])

            # ylims = (df_c[yname].min(), df_c[yname].max())

            ax, labels, legend_handles = plot_hue(
                group_df, key_huegroup, ax, xname, yname, colors
            )

            # Annotations
            if key_axgroup == "context_variation_magnitude":
                title = f"$\sigma_{{rel}}={group_id}$"
            if key_axgroup == "context_feature_args":
                title = group_id
            if yname == "episode_reward":
                ylabel = "mean return\nacross contexts $\mathcal{C}_{train}$"
            if title:
                ax.set_title(title, fontsize=titlefontsize)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=labelfontsize)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=labelfontsize)

            if logx:
                ax.set_xscale("log")
            ax.tick_params(labelsize=ticklabelsize)
            ax.set_xlim(*xlims)
            # ylims = ax.get_ylim()

            # Sort labels, put default name at front
            labels, legend_handles = sort_legend_handles(labels, legend_handles, default_name)

            if i == 1:
                ncols = len(legend_handles)
                title_fontsize = None
                bbox_to_anchor = (0.55, 0.205)
                facecolor = None
                framealpha = None
                legend = fig.legend(
                    handles=legend_handles,
                    labels=labels,
                    loc='lower center',
                    title=legend_title,
                    ncol=1,
                    fontsize=legendfontsize-2,
                    columnspacing=0.5,
                    handletextpad=0.5,
                    handlelength=1.5,
                    bbox_to_anchor=bbox_to_anchor,
                    title_fontsize=title_fontsize,
                    facecolor=facecolor,
                    framealpha=framealpha,
                )

            if i == len(groups) - 1:
                # plot compounding
                fig_c = plt.figure(figsize=figsize_c, dpi=200)
                ax = fig_c.add_subplot(111)
                ax, labels, legend_handles = plot_hue(
                    df_c, key_huegroup, ax, xname, yname, colors
                )
                ylims = ax.get_ylim()
                for axi in axes:
                    axi.set_ylim(*ylims)
                # ax.set_ylim(*ylims)
                title = f"$\sigma_{{rel}}={group_id}$"
                ax.set_title(title)
                ylabel = "mean return\nacross contexts $\mathcal{C}_{train}$"
                ax.set_ylabel(ylabel, fontsize=labelfontsize)

                # Make space for legend
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size=0.7, pad=0.0)
                cax.set_axis_off()

                # Plot legend
                labels, legend_handles = sort_legend_handles(labels, legend_handles, default_name)
                labels = [l.replace("m, l, dt, ", "m, l, dt,\n") for l in labels]
                ncols = len(legend_handles)
                title_fontsize = None
                bbox_to_anchor = (0.75, 0.205)
                facecolor = None
                framealpha = None
                legend = fig_c.legend(
                    handles=legend_handles,
                    labels=labels,
                    loc='lower center',
                    title=None,  # legend_title,
                    ncol=1,
                    fontsize=legendfontsize - 2,
                    columnspacing=0.5,
                    handletextpad=0.5,
                    handlelength=1.5,
                    bbox_to_anchor=bbox_to_anchor,
                    title_fontsize=title_fontsize,
                    facecolor=facecolor,
                    framealpha=framealpha,
                )


        fig.set_tight_layout(True)
        fig_c.set_tight_layout(True)

        if experiment is not None:
            exp_str = str(experiment) + "__"
        else:
            exp_str = ""
        fig_fn = f"taskvariation_evalmeanrew__{env_name}__{key_plotgroup}-{plot_id}.png"
        fig_ffn = Path(path) / fig_fn
        fig.savefig(fig_ffn, bbox_inches="tight")
        fig_fn = f"compounding_evalmeanrew__{env_name}__{key_plotgroup}-{plot_id}.png"
        fig_ffn = Path(path) / fig_fn
        fig_c.savefig(fig_ffn, bbox_inches="tight")
        print("saved at", fig_ffn)
        plt.show()
