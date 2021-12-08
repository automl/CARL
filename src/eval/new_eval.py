from src.eval.gather_data import gather_results
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


if __name__ == '__main__':
    path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/base_vs_context/classic_control/CARLAcrobotEnv"
    results = gather_results(path=path)

    key_plotgroup = ("context_visible", "agent")
    key_group = "context_variation_magnitude"
    key_group_sub = "context_feature_args"
    xname = "step"
    yname = "episode_reward"
    hue = None
    default_name = "None"  # identifier for environment with standard context

    color_palette_name = "colorblind"
    color_default_context = "black"

    figsize = (8, 4)
    labelfontsize = 12
    titlefontsize = 12
    ticklabelsize = 10
    legendfontsize = 10

    if "context_feature_args" in results.columns:
        cfargs = results["context_feature_args"]
        cfargs_new = []
        for cf in cfargs:
            cf = list(cf)
            cf.sort()
            cf_new = "__".join(cf)
            cfargs_new.append(cf_new)
        results["context_feature_args"] = cfargs_new
        context_features = results["context_feature_args"].unique()

    env_names = results['env'].unique()
    if len(env_names) > 1:
        raise NotImplementedError("Try to plot across different envs.")
    env_name = env_names[0]
    plotgroups = results.groupby(by=key_plotgroup)
    for (plot_id, plot_df) in plotgroups:
        fig = plt.figure(figsize=figsize, dpi=200)
        nrows = 1
        ncols = plot_df[key_group].nunique()
        axes = fig.subplots(nrows=nrows, ncols=ncols, sharey=True)

        figtitle = "context visible" if plot_id else "context hidden"
        title = None
        xlabel = None
        ylabel = None
        groups = plot_df.groupby(key_group)
        xmin = plot_df['step'].min()
        xmax = plot_df['step'].max()
        xlims = (xmin, xmax)
        for i, (group_id, group_df) in enumerate(groups):
            if type(axes) == list or type(axes) == np.ndarray:
                ax = axes[i]
            else:
                ax = axes

            groups_sub = group_df.groupby(key_group_sub)
            n = len(groups_sub)
            colors = sns.color_palette(color_palette_name, n)
            legend_handles = []
            labels = []
            for j, (df_id, subset_group_df) in enumerate(groups_sub):
                n_seeds = subset_group_df['seed'].nunique()
                msg = f"{plot_id}, {group_id}, {df_id}: n_seeds={n_seeds}"
                print(msg)
                color = colors[j]
                if df_id == default_name:
                    color = color_default_context
                ax = sns.lineplot(data=subset_group_df, x=xname, y=yname, ax=ax, color=color, marker='', hue=hue)
                legend_handles.append(Line2D([0], [0], color=color))
                labels.append(df_id)

            # Annotations
            if key_group == "context_variation_magnitude":
                title = f"$\sigma_{{rel}}={group_id}$"
            if yname == "episode_reward":
                ylabel = "mean reward\nacross instances $\mathcal{I}_{train}$"
            if title:
                ax.set_title(title, fontsize=titlefontsize)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=labelfontsize)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=labelfontsize)

            ax.set_xlim(*xlims)
            ax.set_xscale("log")
            ax.tick_params(labelsize=ticklabelsize)

            # Sort labels, put default name at front
            if default_name in labels:
                idx = labels.index(default_name)
                name_item = labels.pop(idx)
                labels.insert(0, name_item)
                handle_item = legend_handles.pop(idx)
                legend_handles.insert(0, handle_item)

            if i == 1:
                ncols = len(legend_handles)
                legend = fig.legend(
                    handles=legend_handles,
                    labels=labels,
                    loc='lower center',
                    title="varying context feature",
                    ncol=ncols,
                    fontsize=legendfontsize,
                    columnspacing=0.5,
                    handletextpad=0.5,
                    handlelength=1.5,
                    bbox_to_anchor=(0.5, 0.205)
                )

        fig.suptitle(figtitle)
        fig.set_tight_layout(True)
        fig_fn = f"evalmeanrew__{env_name}__{key_plotgroup}-{plot_id}.png"
        fig_ffn = Path(path) / fig_fn
        fig.savefig(fig_ffn, bbox_inches="tight")
        plt.show()
