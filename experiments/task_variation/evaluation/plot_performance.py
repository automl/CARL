import numpy as np
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import seaborn as sns

from experiments.common.eval.gather_data import gather_results


if __name__ == '__main__':
    # path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/base_vs_context/brax/CARLHalfcheetah"
    # path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/base_vs_context/classic_control/CARLMountainCarEnv"
    # # path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/base_vs_context/box2d/CARLBipedalWalkerEnv"
    # path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/base_vs_context/classic_control/CARLPendulumEnv"
    path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/rerun2/base_vs_context/classic_control/CARLPendulumEnv"
    path2 = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/compounding/base_vs_context/classic_control/CARLPendulumEnv"
    path = "/home/benjamin/Dokumente/code/tmp/CARL/carl/results/rerun/base_vs_context/brax/CARLHalfcheetah"
    results = gather_results(path=path)
    # results2 = gather_results(path=path2)
    # results = pd.concat([results, results2])

    paperversion = True
    plot_across_contextfeatures = False
    plot_across_magnitudes = not plot_across_contextfeatures
    logx = False

    if plot_across_contextfeatures:
        key_plotgroup = ["context_variation_magnitude", "agent"]
        key_axgroup = "context_feature_args"
        key_huegroup = "context_visible"
    elif plot_across_magnitudes:
        key_plotgroup = ["context_visible", "agent"]
        key_axgroup = "context_variation_magnitude"
        key_huegroup = "context_feature_args"

    sns.set_style("whitegrid")
    xname = "step"
    yname = "episode_reward"
    hue = None
    default_name = "None"  # identifier for environment with standard context
    use_first_none = False

    color_palette_name = "colorblind"
    color_default_context = "black"

    figsize = (8, 4) if plot_across_magnitudes else (12, 4)
    if paperversion:
        figsize = (6, 3)
        if plot_across_contextfeatures:
            figsize = (10, 3)
    labelfontsize = 12
    titlefontsize = 12
    ticklabelsize = 10
    legendfontsize = 10

    if "context_feature_args" in results.columns:
        # Get number of context features
        results["n_context_features"] = results["context_feature_args"].apply(len)
        results = results.sort_values(by="n_context_features")

        # Convert cf args list to string
        def cfargs_to_str(x):
            return ", ".join([str(el) for el in x])
        results["context_feature_args"] = results["context_feature_args"].apply(cfargs_to_str)

    env_names = results['env'].unique()
    if len(env_names) > 1:
        raise NotImplementedError("Try to plot across different envs.")
    env_name = env_names[0]
    plotgroups = results.groupby(by=key_plotgroup)
    for (plot_id, plot_df) in plotgroups:
        first_none = None

        fig = plt.figure(figsize=figsize, dpi=200)
        nrows = 1
        ncols = plot_df[key_axgroup].nunique()
        axes = fig.subplots(nrows=nrows, ncols=ncols, sharey=True)

        figtitle = None
        legend_title = None

        # figtitle
        if "context_visible" in key_plotgroup:
            context_visibility_str = "context visible" if plot_id[0] else "context hidden"
            figtitle = f"agent: {plot_id[1]}, {context_visibility_str}"
        elif "context_variation_magnitude" in key_plotgroup:
            figtitle = f"agent: {plot_id[key_plotgroup.index('agent')]}, " \
                       f"$\sigma_{{rel}}={plot_id[key_plotgroup.index('context_variation_magnitude')]}$"

        # legendtitle
        if "context_feature_args" == key_huegroup:
            legend_title = "varying context feature"
        elif "context_visible" == key_huegroup:
            legend_title = "context visibility"
        title = None
        xlabel = None
        ylabel = None
        if plot_across_contextfeatures:
            plot_df = plot_df.sort_values(by="n_context_features")
        groups = plot_df.groupby(key_axgroup)
        xmin = plot_df['step'].min()
        xmax = plot_df['step'].max()
        xlims = (xmin, xmax)
        group_ids = []
        group_dfs = []
        for group_id, group_df in groups:
            group_ids.append(group_id)
            group_dfs.append(group_df)
        if plot_across_contextfeatures:
            # sort along number of context features
            ids = np.array([x for x, y in sorted(enumerate(group_ids), key=lambda x: x[1].count(","))])
            group_ids = np.array(group_ids)[ids]
            group_dfs = np.array(group_dfs)[ids]
        for i, (group_id, group_df) in enumerate(zip(group_ids, group_dfs)):
            if type(axes) == list or type(axes) == np.ndarray:
                ax = axes[i]
            else:
                ax = axes

            groups_sub = group_df.groupby(key_huegroup)
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
                    if use_first_none:
                        if i == 0:
                            first_none = subset_group_df
                        else:
                            if first_none is not None:
                                subset_group_df = first_none
                ax = sns.lineplot(data=subset_group_df, x=xname, y=yname, ax=ax, color=color, marker='', hue=hue)
                legend_handles.append(Line2D([0], [0], color=color))
                labels.append(df_id)

            # Annotations
            if key_axgroup == "context_variation_magnitude":
                title = f"$\sigma_{{rel}}={group_id}$"
            if key_axgroup == "context_feature_args":
                title = group_id
            if yname == "episode_reward":
                ylabel = "mean reward\nacross contexts $\mathcal{C}_{train}$"
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
            if default_name in labels:
                idx = labels.index(default_name)
                name_item = labels.pop(idx)
                labels.insert(0, name_item)
                handle_item = legend_handles.pop(idx)
                legend_handles.insert(0, handle_item)

            if i == 1:
                # labels = [label.replace("__", ", ") for label in labels]
                ncols = len(legend_handles)
                legend = fig.legend(
                    handles=legend_handles,
                    labels=labels,
                    loc='lower center',
                    title=legend_title,
                    ncol=3, #ncols,
                    fontsize=legendfontsize,
                    columnspacing=0.5,
                    handletextpad=0.5,
                    handlelength=1.5,
                    bbox_to_anchor=(0.5, 0.205)
                )

        if not paperversion:
            fig.suptitle(figtitle)
        fig.set_tight_layout(True)

        if plot_across_contextfeatures:
            index = results["n_context_features"].max()  # plus one to account for default/None
            if index > 1:
                # Get the bounding boxes of the axes including text decorations
                r = fig.canvas.get_renderer()
                get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
                bboxes = np.array(list(map(get_bbox, axes.flat)), mtrans.Bbox).reshape(axes.shape)

                bbox = bboxes[index]
                x = 0.5*(bbox.x1 + bboxes[index+1].x0) + 1 * bboxes[0].x0  # account for ylabel offset?
                line = plt.Line2D([x, x], [0.05, 0.95], transform=fig.transFigure, color="black")
                fig.add_artist(line)


        fig_fn = f"evalmeanrew__{env_name}__{key_plotgroup}-{plot_id}.png"
        fig_ffn = Path(path) / fig_fn
        # fig.savefig(fig_ffn, bbox_inches="tight")
        plt.show()
