from src.eval.gather_data import gather_results
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/base_vs_context/classic_control"
    results = gather_results(path=path)
    key_group = "context_variation_magnitude"
    key_group_sub = "context_feature_args"
    xname = "step"
    yname = "episode_reward"
    hue = None
    default_name = "None"  # identifier for environment with standard context

    color_palette_name = "colorblind"
    color_default_context = "black"

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

    fig = plt.figure(figsize=(8, 6), dpi=200)
    nrows = 1
    ncols = results[key_group].nunique()
    axes = fig.subplots(nrows=nrows, ncols=ncols, sharey=True)

    title = None
    xlabel = None
    ylabel = None
    groups = results.groupby(key_group)
    for i, (group_id, group_df) in enumerate(groups):
        ax = axes[i]

        groups_sub = group_df.groupby(key_group_sub)
        n = len(groups_sub)
        colors = sns.color_palette(color_palette_name, n)
        legend_handles = []
        labels = []
        for j, (df_id, subset_group_df) in enumerate(groups_sub):
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

    fig.set_tight_layout(True)
    plt.show()
