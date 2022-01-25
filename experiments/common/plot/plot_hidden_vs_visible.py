import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional

from experiments.common.plot.plotting_style import set_rc_params
from experiments.common.plot.preprocessing import filter_group, filter_std, filter_cf


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

fname = "data/alldata.csv"
df = pd.read_csv(fname)

# filder encoder + gating
df = filter_group(df, ["hidden", "concat"])

# filter concat all
df["carl.state_context_features"].fillna("None", inplace=True)
ids = np.logical_and(df["carl.state_context_features"] != "None", df["group"] == "concat")
df = df[~ids]

# filter std
df = filter_std(df, [0.5])

# filter context feature names
df = filter_cf(df, [
    "m", "g", "l", "dt", "max_speed"
])

figsize = (9, 3)
dpi = 300
xname = "_step"
yname = "eval/return"
hue = "group"
legendtitle = "Visibility"
xlabel = "step"
ylabel = "mean reward"
groupkey = "contexts.context_feature_args"
xticks = [0, 250_000, 500_000]
xticklabels = ["0", "250k", "500k"]
style = None  # "seed"

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
    axtitle = group_id
    ax = axes[i]
    ax = sns.lineplot(ax=ax, data=group_df, x=xname, y=yname, hue=hue, palette=palette, style=style)
    if i != 1:
        ax.get_legend().remove()
    else:
        legend = ax.get_legend()
        legend.set_title("Context")
    ax = adorn_ax(
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        axtitle=axtitle,
        xticks=xticks,
        xticklabels=xticklabels,
    )
fig.set_tight_layout(True)
plt.show()



