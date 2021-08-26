import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from policy_transfer import gravities


def plot_gravities_vlines(ax, gravities, yvalue, mean, std):
    yvals = np.linspace(0.01, yvalue, len(gravities))
    keys = np.array(list(gravities.keys()))
    values = np.array(list(gravities.values()))
    ids = np.argsort(values)
    values = values[ids]
    keys = keys[ids]
    for i in range(len(gravities)):
        key = keys[i]
        value = values[i]
        ax.vlines(value, 0, yvals[i], color="mediumvioletred")
        leq = True
        if value <= mean or True:
            P = norm.cdf(value, mean, std)
        else:
            leq = False
            P = 1 - norm.cdf(value, mean, std)
        leq_str = "\leq " if leq else ">"
        ax.text(
            value,
            yvals[i],
            f"$g_{{{key}}}={value:.2f}, P(X{leq_str}g_{{{key}}})={P:.4f}$",
            bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 1, 'linewidth': 0}
        )


mean = gravities["Mars"]
std = 1.45
n_contexts = 1000
sampled_gravities = norm.rvs(loc=mean, scale=std, size=n_contexts)
figname = "gravity_distribution.png"

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.hist(sampled_gravities, bins=100)
gravities_l0 = sampled_gravities[sampled_gravities <= 0]
gravities_g0 = sampled_gravities[sampled_gravities > 0]
ax = sns.histplot(x=sampled_gravities, kde=True, ax=ax, stat="probability", cumulative=False, fill=False)
# ax = sns.kdeplot(x=sampled_gravities, cumulative=True)
ylims = ax.get_ylim()
title = f"$\mu = g_{{Mars}} = {gravities['Mars']:.2f}, \sigma = {std}$"
ax.set_title(title)

plot_gravities_vlines(ax, gravities, ylims[1], mean, std)
ax.set_xlabel("gravity [m/s²]")
fig.set_tight_layout(True)
fig.savefig(figname, bbox_inches="tight")
plt.show()

