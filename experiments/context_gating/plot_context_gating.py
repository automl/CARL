from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional

from experiments.common.plot.plotting_style import set_rc_params
from experiments.common.plot.preprocessing import filter_cf, filter_group
from experiments.context_gating.eval_context_gating import load_data_context_gating


def adorn_ax(
    ax,
    xticks: Optional = None,
    xticklabels: Optional = None,
    xlabel: Optional = None,
    ylabel: Optional = None,
    axtitle: Optional = None,
):
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.ticklabel_format(axis="x", style="scientific")
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(axtitle)
    return ax


fname = Path("data_context_gating_pendulum.csv")
df = load_data_context_gating(fname, download=False)

# filter hidden
df = filter_group(df, ["concat", "encoder", "gating"])

# filter concat all
df["carl.state_context_features"].fillna("None", inplace=True)
ids = np.logical_and(df["carl.state_context_features"] != "None", df["group"] == "concat")
df_group = df["group"].copy()
df_group[ids] = "concat all"
df["group"] = df_group
# df = df[~ids]

filter_cfs = [
    ["m", "g", "l", "dt", "max_speed",],
    ["max_speed, m, l, g, dt"],
    [np.nan],
    [np.nan, "m", "g", "l", "dt", "max_speed", "max_speed, m, l, g, dt"]
]

orig_df = df.copy()

for allowed_cfs in filter_cfs:
    fig_fname = Path(f"plot_context_gating_pendulum_filter-{allowed_cfs}.png")
    fig_fname.parent.mkdir(parents=True, exist_ok=True)
    # filter context feature names
    df = filter_cf(orig_df,
         allowed_cfs
    )

    figsize = (6, 3)
    dpi = 300
    xname = "_step"
    yname = "eval/return"
    hue = "group"
    legendtitle = None  # "Context"
    xlabel = "step"
    ylabel = "mean return\nacross contexts $\mathcal{C}_{train}$"
    groupkey = ["contexts.default_sample_std_percentage"] #, "contexts.context_feature_args"]
    xticks = [0, 250_000, 500_000]
    xticklabels = ["0", "250k", "500k"]
    style = None  # "seed"
    figtitle = None  # str(allowed_cfs)

    set_rc_params()
    if hue is None:
        palette = "colorblind"
    else:
        unique = df[hue].unique()
        colors = sns.palettes.color_palette("colorblind", len(unique))
        palette = {k: v for k, v in zip(unique, colors)}

    if groupkey is None:
        group_ids = [None]
        group_dfs = [df]
        groups = zip(group_ids, group_dfs)
        n_groups = 1
    else:
        groups = df.groupby(groupkey)
        n_groups = len(groups)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(nrows=1, ncols=n_groups, sharey=True)
    for i, (group_id, group_df) in enumerate(groups):
        axtitle = f"$\sigma_{{rel}}={group_id}$"
        ax = axes[i]
        ax = sns.lineplot(ax=ax, data=group_df, x=xname, y=yname, hue=hue, palette=palette, style=style)
        if i != 1:
            ax.get_legend().remove()
        else:
            legend = ax.get_legend()
            legend.set_title(legendtitle)
        ax = adorn_ax(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            axtitle=axtitle,
            xticks=xticks,
            xticklabels=xticklabels,
        )
    fig.suptitle(figtitle)
    fig.set_tight_layout(True)
    fig.savefig(fig_fname, dpi=300, bbox_inches="tight")
    plt.show()
