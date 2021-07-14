import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = Path("results/singlecontextfeature/box2d/MetaLunarLanderEnv")


progress_fname = "progress.csv"
# context_dirs = [Path(x[0]) for x in os.walk(path)]
context_dirs = [path / Path(p) for p in os.listdir(path)]
cf_names = [p.stem for p in context_dirs]

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

fig = plt.figure()
ax = fig.add_subplot(111)
for cf_name, df in data.items():
    ax = sns.lineplot(data=df, x="iteration", y="ep_rew_mean", ax=ax, label=cf_name)
title = f"Context feature performance"
ax.set_title(title)
ax.legend()
fig.set_tight_layout(True)
plt.show()



