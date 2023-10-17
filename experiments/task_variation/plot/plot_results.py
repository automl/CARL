import wandb
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Any, Dict, Union
import matplotlib as mpl
import numpy as np
from experiments.common.plot.plotting_style import set_rc_params


def load_data(
    experiment_setting: Dict,
    df_fname: Union[str, Path],
    reload_data: bool = False,
    groups: Optional[Dict] = None,
):
    if groups is None:
        groups = {
            "hidden_context": "hidden",
            "concat_context": "concat",
            "context_gating": "gating",
        }
    df_fname = Path(df_fname)
    if not df_fname.is_file() or reload_data:
        df_fname.parent.mkdir(parents=True, exist_ok=True)

        filters = experiment_setting["filters"]
        config_entries = experiment_setting["config_entries"]

        api = wandb.Api()
        runs = api.runs("tnt/carl", filters=filters)
        dfs = []
        runs = list(runs)
        for run in tqdm(runs):
            df = pd.DataFrame()
            for i, row in run.history(keys=metrics).iterrows():
                if all([metric in row for metric in metrics]):
                    df = df.append(row, ignore_index=True)
            for entry in config_entries:
                entry_list = entry.split(".")
                config_entry = run.config
                for e in entry_list:
                    config_entry = config_entry[e]
                if isinstance(config_entry, (list, tuple)):
                    config_entry = ", ".join(config_entry)
                if entry == "group":
                    df[entry] = groups[config_entry]
                else:
                    df[entry] = config_entry
            if len(df) > 1:
                dfs.append(df)
        df = pd.concat(dfs)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(df_fname, index=False)
    else:
        df = pd.read_csv(df_fname)

    return df


def set_legend(
    fig,
    ax,
    legendtitle: Optional[str] = None,
    legendfontsize: Optional[int] = None,
    ncols: Optional[int] = None,
):
    legend = ax.get_legend()
    legend.set_title(legendtitle)
    handles, labels = ax.get_legend_handles_labels()
    key = lambda t: t[0]
    labels, handles = zip(*sorted(zip(labels, handles), key=key))
    if experiment == "compounding":
        key = lambda x: x[0].count(",")
        labels, handles = zip(*sorted(zip(labels, handles), key=key))
    labels = list(labels)
    handles = list(handles)
    if ncols is None:
        ncols = len(handles)
    legend.remove()
    legend = fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        title=legendtitle,
        ncol=ncols,
        fontsize=legendfontsize,
        columnspacing=0.5,
        handletextpad=0.5,
        handlelength=1.5,
        bbox_to_anchor=(0.6, 0.205),
    )
    return legend


if __name__ == "__main__":
    experiment = "hidden_vs_visible"
    reload_data = False

    experiment_settings = {
        "compounding": {
            "filters": {"group": "hidden_context", "state": "finished"},
            "config_entries": {
                "group",
                "contexts.context_feature_args",
                "carl.state_context_features",
                "contexts.default_sample_std_percentage",
                "seed",
            },
            "plotting": {
                "xname": "_step",
                "yname": "eval/return",
                "hue": "contexts.context_feature_args",
                "legendtitle": "Varying Context Feature",
                "xlabel": "step",
                "ylabel": "mean reward",
                "group": None,  # "contexts.default_sample_std_percentage"
            },
        },
        "hidden_vs_visible": {
            "filters": {
                # "config.contexts.default_sample_std_percentage": {"$in": [0.5]},
                "group": {"$in": ["hidden_context", "concat_context"]},
                "state": "finished",
            },
            "config_entries": {
                "group",
                "contexts.context_feature_args",
                "carl.state_context_features",
                "contexts.default_sample_std_percentage",
                "seed",
            },
            "plotting": {
                "xname": "_step",
                "yname": "eval/return",
                "hue": "group",
                "legendtitle": "Visibility",
                "xlabel": "step",
                "ylabel": "mean reward",
                "group": "contexts.context_feature_args",
                "xticks": [0, 250_000, 500_000],
                "xticklabels": ["0", "250k", "500k"],
                # "style": "contexts.default_sample_std_percentage",
            },
        },
        "task_variation": {
            "filters": {
                "group": {"$in": ["hidden_context", "concat_context"]},
                "state": "finished",
            },
            "config_entries": {
                "group",
                "contexts.context_feature_args",
                "carl.state_context_features",
                "contexts.default_sample_std_percentage",
                "seed",
            },
            "plotting": {
                "xname": "_step",
                "yname": "eval/return",
                "hue": "contexts.context_feature_args",
                "legendtitle": "Varying Context Feature",
                "xlabel": "step",
                "ylabel": "mean reward\nacross contexts $\mathcal{C}_{train}$",
                "group": ["group", "contexts.default_sample_std_percentage"],
                "xticks": [0, 250_000, 500_000],
                "xticklabels": ["0", "250k", "500k"],
            },
        },
        "context_gating": {
            "filters": {
                "config.carl.gaussian_noise_std_percentage": {"$in": [0.4]},
                "config.carl.scale_context_features": "no",
                "state": "finished",
            },
            "config_entries": {
                "group",
                "contexts.context_feature_args",
                "carl.state_context_features",
                "contexts.default_sample_std_percentage",
                "seed",
            },
        },
    }
    metrics = ["eval/return"]

    df_fname = Path(".") / f"data_{experiment}.csv"
    experiment_setting = experiment_settings[experiment]
    df = load_data(
        experiment_setting=experiment_setting,
        reload_data=reload_data,
        df_fname=df_fname,
    )

    no_varying_context_feature_name = "None"
    no_context_feature_varying_color = "black"
    legendfontsize = 10
    recolor_novarycf = False
    if "contexts.context_feature_args" in df:
        df["contexts.context_feature_args"].fillna(
            value=no_varying_context_feature_name, inplace=True
        )
        if experiment == "task_variation" or experiment == "hidden_vs_visible":
            # filter everything that varies more than one context feature
            ids = df["contexts.context_feature_args"].apply(lambda x: "," not in x)
            df = df[ids]
    if "carl.state_context_features" in df:
        # filter nan and concat
        ids = np.logical_and(
            df["carl.state_context_features"] == np.nan, df["group"] == "concat"
        )
        df = df[~ids]
    exp = experiment_settings[experiment]

    if experiment == "hidden_vs_visible" or experiment == "task_variation":
        ids = df["contexts.context_feature_args"] == no_varying_context_feature_name
        df = df[~ids]

        # filter concat: concat all
        df["carl.state_context_features"].fillna("None", inplace=True)
        ids = np.logical_and(
            df["carl.state_context_features"] == "None", df["group"] == "concat"
        )
        df = df[~ids]

    if experiment == "hidden_vs_visible":
        # plot only one std
        std = 0.5
        ids = df["contexts.default_sample_std_percentage"] == std
        df = df[ids]

    def plgetter(name: str) -> Optional[Any]:
        return exp["plotting"].get(name)

    xname = plgetter("xname")
    yname = plgetter("yname")
    hue = plgetter("hue")
    xlabel = plgetter("xlabel")
    ylabel = plgetter("ylabel")
    axtitle = plgetter("axtitle")
    legendtitle = plgetter("legendtitle")
    groupkey = plgetter("group")
    xticks = plgetter("xticks")
    xticklabels = plgetter("xticklabels")
    style = plgetter("style")
    if groupkey is None:
        group_ids = [None]
        group_dfs = [df]
        groups = zip(group_ids, group_dfs)
    else:
        groups = df.groupby(groupkey)

    set_rc_params()
    if hue is None:
        palette = "colorblind"
    else:
        unique = df[hue].unique()
        colors = sns.palettes.color_palette("colorblind", len(unique))
        palette = {k: v for k, v in zip(unique, colors)}

    figsize = (4, 3)
    dpi = 300
    if experiment == "task_variation":
        # 1 fig per group_id[0]
        figsize = (6, 3)
        n_figs = len(df.groupby(groupkey[0]))
        n_axes = len(df.groupby(groupkey[1]))
        figs = [plt.figure(figsize=figsize, dpi=dpi) for i in range(n_figs)]
        ids = [[e] * n_axes for e in range(n_figs)]
        fig_ids = []
        for listl in ids:
            fig_ids.extend(listl)
        ax_ids = list(np.arange(0, n_axes)) * n_figs
        groups = df.groupby(groupkey[0])

        for i, (group_id, group_df) in enumerate(groups):
            group_str = "" if group_id is None else f"_{groupkey}{group_id}"
            axtitle = group_id if group_id is not None else axtitle
            fig_fname = df_fname.parent / ("plot_" + experiment + group_str + ".png")
            figsize = (4, 3)
            dpi = 300
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axes = fig.subplots(nrows=1, ncols=n_axes, sharey=True)

            subgroups = group_df.groupby(groupkey[1])

            for j, (subgroup_id, subgroup_df) in enumerate(subgroups):
                ax = axes[j]
                ax = sns.lineplot(
                    data=subgroup_df,
                    x=xname,
                    y=yname,
                    hue=hue,
                    palette=palette,
                    ax=ax,
                    style=style,
                )
                ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel(ylabel)
                if experiment == "task_variation":
                    axtitle = f"$\sigma_{{rel}}={subgroup_id}$"
                ax.set_title(axtitle)

                # Legend
                if j == 1:
                    legend = ax.get_legend()
                    legend.set_title(legendtitle)
                    handles, labels = ax.get_legend_handles_labels()
                    key = lambda t: t[0]
                    labels, handles = zip(*sorted(zip(labels, handles), key=key))
                    if experiment == "compounding":
                        key = lambda x: x[0].count(",")
                        labels, handles = zip(*sorted(zip(labels, handles), key=key))
                    labels = list(labels)
                    handles = list(handles)
                    ncols = len(handles)
                    legend.remove()
                    legend = fig.legend(
                        handles=handles,
                        labels=labels,
                        loc="lower center",
                        title=legendtitle,
                        ncol=3,  # ncols,
                        fontsize=legendfontsize,
                        columnspacing=0.5,
                        handletextpad=0.5,
                        handlelength=1.5,
                        bbox_to_anchor=(0.6, 0.205),
                    )
                    # ax.legend(handles, labels)
                else:
                    ax.get_legend().remove()

                xlim = group_df[xname].min(), group_df[xname].max()
                ax.set_xlim(*xlim)
                xmax = xlim[1]
                if xticks is not None:
                    ax.set_xticks(xticks)
                    ax.ticklabel_format(axis="x", style="scientific")
                if xticklabels is not None:
                    ax.set_xticklabels(xticklabels)

            fig.set_tight_layout(True)
            fig.savefig(fig_fname)
            plt.show()

    else:
        if experiment == "hidden_vs_visible":
            figsize = (6, 3)
            dpi = 300
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axes = fig.subplots(nrows=1, ncols=len(groups), sharey=True)

        for i, (group_id, group_df) in enumerate(groups):
            group_str = "" if group_id is None else f"_{groupkey}{group_id}"
            axtitle = group_id if group_id is not None else axtitle
            fig_fname = df_fname.parent / ("plot_" + experiment + group_str + ".png")
            if experiment == "hidden_vs_visible":
                ax = axes[i]
            else:
                figsize = (4, 3)
                dpi = 300
                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax = fig.add_subplot(111)

            ax = sns.lineplot(
                data=group_df,
                x=xname,
                y=yname,
                hue=hue,
                palette=palette,
                ax=ax,
                style=style,
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(axtitle)

            # Legend
            legend = ax.get_legend()
            legend.set_title(legendtitle)
            handles, labels = ax.get_legend_handles_labels()
            key = lambda t: t[0]
            labels, handles = zip(*sorted(zip(labels, handles), key=key))
            if experiment == "compounding":
                key = lambda x: x[0].count(",")
                labels, handles = zip(*sorted(zip(labels, handles), key=key))
            labels = list(labels)
            handles = list(handles)
            if "None" in labels:
                idx = labels.index(no_varying_context_feature_name)
                name_item = labels.pop(idx)
                labels.insert(0, name_item)
                handle_item = handles.pop(idx)
                handles.insert(0, handle_item)

                if recolor_novarycf:
                    idx = labels.index(no_varying_context_feature_name)
                    orig_color = handles[idx].get_color()
                    handles[idx].set_color(no_context_feature_varying_color)
                    children = ax.get_children()
                    for child in children:
                        if type(child) == mpl.lines.Line2D:
                            if child.get_color() == orig_color:
                                child.set_color(no_context_feature_varying_color)
                        elif type(child) == mpl.collections.PolyCollection:
                            facecolors = child.get_fc()
                            if np.all(
                                np.isclose(facecolors[0, :3], np.array(orig_color))
                            ):
                                color = np.array(
                                    mpl.colors.to_rgba(no_context_feature_varying_color)
                                )
                                color[-1] = facecolors[0, -1]
                                child.set_color(no_context_feature_varying_color)
            ax.legend(handles, labels)

            xlim = group_df[xname].min(), group_df[xname].max()
            ax.set_xlim(*xlim)
            if xticks is not None:
                ax.set_xticks(xticks)
                ax.ticklabel_format(axis="x", style="scientific")
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels)

            if experiment == "hidden_vs_visible":
                if i != 1:
                    ax.get_legend().remove()
                else:
                    legend = set_legend(
                        fig=fig,
                        ax=ax,
                        legendtitle=legendtitle,
                        legendfontsize=legendfontsize,
                        ncols=1,
                    )
                continue
            fig.set_tight_layout(True)
            fig.savefig(fig_fname)
            plt.show()
        if experiment == "hidden_vs_visible":
            fig.set_tight_layout(True)
            fig.savefig(fig_fname)
            plt.show()
