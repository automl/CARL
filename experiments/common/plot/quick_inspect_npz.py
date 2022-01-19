import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fn = "tmp/test_logs/PPO_123456/evaluations.npz"
path = "results/singlecontextfeature_0.15_hidecontext/box2d/MetaBipedalWalkerEnv/None/PPO_2"
path = "results/singlecontextfeature_0.125_hidecontext/classic_control/MetaPendulumEnv/None/PPO_2"
path = "results/singlecontextfeature_0.075_hidecontext/classic_control/MetaPendulumEnv/None/PPO_3"
path = "results/singlecontextfeature_0.1_hidecontext/box2d/MetaLunarLanderEnv/None/PPO_3"
path = "results/singlecontextfeature_0.1_hidecontext/brax/MetaHalfcheetah/None/PPO_3"
fn = os.path.join(path, "evaluations.npz")
data = np.load(fn)

k_steps = "timesteps"
k_results = "results"
k_eplen = "ep_lengths"

steps = data[k_steps]
results = data[k_results]
ep_lengths = np.mean(data[k_eplen], axis=1)
y = np.mean(results, axis=1)
y_std = np.std(results, axis=1)

fig = plt.figure(figsize=(6, 8))
axes = fig.subplots(nrows=2, ncols=1)
# ax = sns.lineplot(steps, y, ax=ax, marker='o')
ax = axes[0]
ax.plot(steps, y, marker='.', label="ep reward")
ax.set_ylabel("mean reward across instances")

ax = axes[1]
ax.plot(steps, ep_lengths, marker='.', color="orange", label="ep length")
ax.set_ylabel("mean episode length across instances")
ax.set_xlabel("step")
fig.set_tight_layout(True)
# ax.fill_between(steps, y-y_std, y+y_std, color="blue", alpha=0.5)
# ax.set_xlim(0, max(steps))
plt.show()
