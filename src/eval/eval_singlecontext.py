import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

path = "results/singlecontextfeature_0.25_hidecontext/box2d/MetaBipedalWalkerEnv"
# path = "results/singlecontextfeature_0.25_hidecontext/box2d/MetaLunarLanderEnv"

xname = "step"
yname = "ep_rew_mean"

plot_comparison = False
plot_mean_performance = True

"""
Assumend folder structure:

logdir / env_name / name of context feature / {agent}_{seed} 
"""
path = Path(path)
env_name = path.stem

progress_fname = "progress.csv"
default_name = "None"
# context_dirs = [Path(x[0]) for x in os.walk(path)]
context_dirs = [path / Path(p) for p in os.listdir(path)]
context_dirs = [p for p in context_dirs if os.path.isdir(p)]
cf_names = [p.stem for p in context_dirs]

ids = np.argsort(cf_names)
context_dirs = np.array(context_dirs)[ids]
cf_names = np.array(cf_names)[ids]

# where_baseline = np.where(cf_names == "None")[0]
# if where_baseline:
#     idx = where_baseline[0]


dirs_per_cf = {}
for i, cf_name in enumerate(cf_names):
    cf_dir = context_dirs[i]
    agent_seed_dirs = os.listdir(cf_dir)
    agent_seed_dirs = [os.path.join(cf_dir, p) for p in agent_seed_dirs]
    dirs_per_cf[cf_name] = agent_seed_dirs

data = {}
for cf_name, cf_dirs in dirs_per_cf.items():
    D = []
    for cf_dir in cf_dirs:
        cf_dir = Path(cf_dir)
        folder = cf_dir.stem
        agent, seed = folder.split("_")
        seed = int(seed)

        progress_fn = cf_dir / progress_fname
        df = pd.read_csv(progress_fn)
        n = len(df['time/total_timesteps'])
        D.append(pd.DataFrame({
            "seed": [seed] * n,
            "step": df['time/total_timesteps'].to_numpy(),
            "iteration": df['time/iterations'].to_numpy(),
            yname: df['rollout/ep_rew_mean'].to_numpy(),
        }))
    D = pd.concat(D)
    data[cf_name] = D


color_palette_name = "husl"  # TODO find palette with enough individual colors or use linestyles
color_default_context = "black"
if plot_mean_performance:
    means = []
    for cf_name, df in data.items():
        mean = df[yname].mean()
        std = df[yname].std()
        means.append({"context_feature": cf_name, "mean": mean, "std": std})
    means = pd.DataFrame(means)

    n = len(means)
    colors = np.array(sns.color_palette(color_palette_name, n))
    idx = means["context_feature"] == default_name
    colors[idx] = mpl.colors.to_rgb(color_default_context)

    figsize = (8, 6)
    dpi = 200
    fig = plt.figure(figsize=figsize, dpi=200)
    axes = fig.subplots(nrows=2, ncols=1, sharex=True)

    ax = axes[0]
    ax = sns.barplot(data=means, x="context_feature", y="mean", ax=ax, palette=colors)
    ax.set_xlabel("")
    ax = axes[1]
    ax = sns.barplot(data=means, x="context_feature", y="std", ax=ax, palette=colors)
    xticklabels = means["context_feature"]
    ax.set_xticklabels(xticklabels, rotation=30, fontsize=9, ha="right")
    title = f"{env_name}"
    fig.suptitle(title)
    fig.set_tight_layout(True)
    fname = path / f"ep_rew_mean_mean_std.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.show()

if plot_comparison:
    sns.set_style("white")

    figsize = (8, 6)
    dpi = 200
    fig = plt.figure(figsize=figsize, dpi=200)
    ax = fig.add_subplot(111)
    n = len(data)
    # n_colors_in_palette = n
    # if color_palette_name in sns.palettes.SEABORN_PALETTES:
    #     n_colors_in_palette = sns.palettes.SEABORN_PALETTES[color_palette_name]
    colors = sns.color_palette(color_palette_name, n)
    legend_handles = []
    labels = []
    for i, (cf_name, df) in enumerate(data.items()):
        color = colors[i]
        if cf_name == default_name:
            color = color_default_context
        ax = sns.lineplot(data=df, x=xname, y=yname, ax=ax, color=color)
        legend_handles.append(Line2D([0], [0], color=color))
        labels.append(cf_name)
    title = f"{env_name}"
    ax.set_title(title)

    # Sort labels, put default name at front
    idx = labels.index(default_name)
    name_item = labels.pop(idx)
    labels.insert(0, name_item)
    handle_item = legend_handles.pop(idx)
    legend_handles.insert(0, handle_item)

    ax.legend(handles=legend_handles, labels=labels)
    fig.set_tight_layout(True)
    plt.show()

    fname = path / f"ep_rew_mean_over_{xname}.png"
    fig.savefig(fname, bbox_inches="tight")



