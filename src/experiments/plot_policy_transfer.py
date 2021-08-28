import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

eval_data_fnames = [
    "results/experiments/policytransfer/new/CARLLunarLanderEnv/hidden/GRAVITY_Y/eval_data.csv",
    "results/experiments/policytransfer/new/CARLLunarLanderEnv/visible/GRAVITY_Y/eval_data.csv",
]
figfname = os.path.join(os.path.commonpath(eval_data_fnames), "policytransfer_hiddenvisible.png")
sns.set_context("paper")
data_list = [pd.read_csv(fn, sep=";") for fn in eval_data_fnames]
data_list[0]["visibility"] = ["hidden"] * len(data_list[0])
data_list[1]["visibility"] = ["visible"] * len(data_list[1])
data = pd.concat(data_list)
# for data in data_list:
train_contexts_key = '$\mathcal{I}_{train}$'
data['planet'][data['planet'] == 'train\ncontexts'] = train_contexts_key
data = data.rename(columns={'train\ncontexts': train_contexts_key})
custom_dict = {
    train_contexts_key: 0,
    'train\ndistribution': 1,
    'Jupiter': 7,
    'Neptune': 6,
    'Earth': 5,
    'Mars': 4,
    'Moon': 3,
    'Pluto': 2
}
data = data.sort_values(by=['planet'], key=lambda x: x.map(custom_dict))
data = data[data['planet'] != 'train\ndistribution']

filter_by_ep_length = False
plot_ep_length = False
max_ep_length = 1000
if filter_by_ep_length:
    data = data[data["ep_length"] < max_ep_length]
palette = "colorblind"
hue = 'train_seed'
hue = 'visibility'
figsize = (5, 3)
dpi = 250
fig = plt.figure(figsize=figsize, dpi=dpi)
# ax = fig.add_subplot(111)
if plot_ep_length:
    axes = fig.subplots(nrows=2, ncols=1, sharex=True)
else:
    ax = fig.add_subplot(111)

if plot_ep_length:
    ax = axes[0]
ax = sns.violinplot(
    data=data,
    x="planet",
    y="ep_rew_mean",
    ax=ax,
    hue=hue,
    cut=0,
    scale='width',
    inner=None,
    split=True,
    linewidth=0.1,
    saturation=0.8,
    palette=palette
)
ax = sns.stripplot(
    data=data,
    x="planet",
    y="ep_rew_mean",
    ax=ax,
    hue=hue,
    size=1.5,
    edgecolors=[0.,0.,0.],
    linewidths=0,
    color='black',
    split=True,
    # palette=palette
)
ax.set_ylim(-10000, 500)
ax.set_ylabel("mean reward")

# create legend
labels = ["hidden", "visible"]
legend_handles = []
n = len(labels)
colors = sns.color_palette(palette, n)
for i in range(n):
    color = colors[i]
    # legend_handles.append(Line2D([0], [0], color=color))
    legend_handles.append(FancyBboxPatch((0.,0.), 10, 10, color=color))
ncols = len(legend_handles)
ncols = 1
loc = 'right'
legend = ax.legend(
    handles=legend_handles,
    labels=labels,
    loc=loc,
    title="visibility",
    ncol=ncols,
    # fontsize=legendfontsize,
    columnspacing=0.5,
    handletextpad=0.5,
    handlelength=1.5,
    borderpad=0.25
    # bbox_to_anchor=(0.5, 0.205)
)

if plot_ep_length:
    ax.set_xlabel("")
    ax = axes[1]
    ax = sns.violinplot(
        data=data, x="planet", y="ep_length", ax=ax, hue=hue, cut=0, palette=palette, )
    # ax = sns.swarmplot(data=data, x="planet", y="ep_length", ax=ax, hue=hue, size=2, palette=palette)

fig.set_tight_layout(True)
fig.savefig(figfname, bbox_inches="tight")
plt.show()