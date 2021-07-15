import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

path = Path("results/singlecontextfeature_0.1_hidecontext/box2d/MetaLunarLanderEnv")
xname = "step"

"""
Assumend folder structure:

logdir / env_name / name of context feature / {agent}_{seed} 
"""
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
            "ep_rew_mean": df['rollout/ep_rew_mean'].to_numpy(),
        }))
    D = pd.concat(D)
    data[cf_name] = D

sns.set_style("white")

color_palette_name = "colorblind"  # TODO find palette with enough individual colors or use linestyles
color_default_context = "black"
figsize = (8, 6)
dpi = 200
fig = plt.figure(figsize=figsize, dpi=200)
ax = fig.add_subplot(111)
n = len(data)
colors = sns.color_palette(color_palette_name, n)
legend_handles = []
labels = []
for i, (cf_name, df) in enumerate(data.items()):
    color = colors[i]
    if cf_name == default_name:
        color = color_default_context
    ax = sns.lineplot(data=df, x=xname, y="ep_rew_mean", ax=ax, color=color)
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



