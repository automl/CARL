import matplotlib.pyplot as plt
import seaborn as sns
import json
from policy_transfer import gravities
from carl.experiments import plot_gravities_vlines

fname = "experiments/lunarLander_contexts_train_2intervals.json"
fname = "experiments/lunarLander_contexts_train_Gaussian.json"
context_feature_key = "GRAVITY_Y"
with open(fname, 'r') as file:
    contexts = json.load(file)
sampled_gravities = [c[context_feature_key] for c in contexts.values()]
mean = gravities["Mars"]
std = 1.45
n_contexts = 100000
gravities_plot = gravities.copy()
del gravities_plot["Jupiter"]
del gravities_plot["Neptune"]

fig = plt.figure(figsize=(4, 3), dpi=200)
ax = fig.add_subplot(111)
ax = sns.histplot(sampled_gravities, ax=ax, bins=1000, cumulative=True)
ax.set_ylabel("count")
ax.set_xlabel("gravity [m/s²]")
ylims = ax.get_ylim()
ax = plot_gravities_vlines(ax, gravities_plot, ylims[1], mean, std, 10, shortanno=True)
fig.set_tight_layout(True)
fig.savefig("experiments/gravity_sampled_gravities.png", bbox_inches="tight", dpi=200)
plt.show()
