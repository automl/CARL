import numpy as np
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import seaborn as sns
import pandas as pd

from experiments.common.eval.gather_data import gather_results


def plot_hue(group_df, key_huegroup, ax, xname, yname, color_palette_name):
    groups_sub = group_df.groupby(key_huegroup)
    n = len(groups_sub)
    colors = sns.color_palette(color_palette_name, n)
    legend_handles = []
    labels = []
    for j, (df_id, subset_group_df) in enumerate(groups_sub):
        n_seeds = subset_group_df["seed"].nunique()
        msg = f"{plot_id}, {group_id}, {df_id}: n_seeds={n_seeds}"
        print(msg)
        color = colors[j]
        if df_id == default_name:
            color = color_default_context
        ax = sns.lineplot(
            data=subset_group_df,
            x=xname,
            y=yname,
            ax=ax,
            color=color,
            marker="",
            hue=None,
        )
        legend_handles.append(Line2D([0], [0], color=color))
        label = df_id
        labels.append(label)
    xmin = group_df["step"].min()
    xmax = group_df["step"].max()
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


if __name__ == "__main__":
    path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/rerun2/base_vs_context/classic_control/CARLPendulumEnv"
    path2 = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/compounding/base_vs_context/classic_control/CARLPendulumEnv"
    results = gather_results(path=path)
    results2 = gather_results(path=path2)
    results = pd.concat([results, results2])
    del results2

    experiment = "compounding"
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

    figsize = (6, 3)
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

        results["context_feature_args"] = results["context_feature_args"].apply(
            cfargs_to_str
        )

    # split
    # ids = results["context_feature_args"].apply(lambda x: "," in x)
    # results = results[~ids]
    # compounding_results = results[ids]

    env_names = results["env"].unique()
    if len(env_names) > 1:
        raise NotImplementedError("Try to plot across different envs.")
    env_name = env_names[0]
    plotgroups = results.groupby(by=key_plotgroup)
    for (plot_id, plot_df) in plotgroups:
        fig = plt.figure(figsize=figsize, dpi=200)
        nrows = 1
        ncols = plot_df[key_axgroup].nunique() + 1
        axes = fig.subplots(nrows=nrows, ncols=ncols, sharey=True)

        # legendtitle
        title = None
        xlabel = None
        ylabel = None
        groups = plot_df.groupby(key_axgroup)
        xmin = plot_df["step"].min()
        xmax = plot_df["step"].max()
        xlims = (xmin, xmax)
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

            ax, labels, legend_handles = plot_hue(
                group_df, key_huegroup, ax, xname, yname, color_palette_name
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

            ax.set_xlim(*xlims)
            if logx:
                ax.set_xscale("log")
            ax.tick_params(labelsize=ticklabelsize)

            # Sort labels, put default name at front
            labels, legend_handles = sort_legend_handles(
                labels, legend_handles, default_name
            )

            if i == 1:
                ncols = len(legend_handles)
                title_fontsize = None
                bbox_to_anchor = (0.35, 0.205)
                facecolor = None
                framealpha = None
                legend = fig.legend(
                    handles=legend_handles,
                    labels=labels,
                    loc="lower center",
                    title=legend_title,
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

            if i == len(groups) - 1:
                # Plot compounding variations
                ax, labels, legend_handles = plot_hue(
                    df_c, key_huegroup, axes[-1], xname, yname, color_palette_name
                )
                title = f"$\sigma_{{rel}}={group_id}$"
                ax.set_title(title)
                labels, legend_handles = sort_legend_handles(
                    labels, legend_handles, default_name
                )
                ncols = len(legend_handles)
                title_fontsize = None
                bbox_to_anchor = (0.75, 0.205)
                facecolor = None
                framealpha = None
                legend = fig.legend(
                    handles=legend_handles,
                    labels=labels,
                    loc="lower center",
                    title=legend_title,
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

        if experiment is not None:
            exp_str = str(experiment) + "__"
        else:
            exp_str = ""
        fig_fn = f"{exp_str}evalmeanrew__{env_name}__{key_plotgroup}-{plot_id}.png"
        fig_ffn = Path(path) / fig_fn
        # fig.savefig(fig_ffn, bbox_inches="tight")
        print("saved at", fig_ffn)
        plt.show()
