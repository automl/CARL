import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from policy_transfer import gravities


def plot_gravities_vlines(ax, gravities, yvalue, mean, std, fontsize, shortanno: bool = False):
    yvals = np.linspace(0.01, yvalue, len(gravities))
    keys = np.array(list(gravities.keys()))
    values = np.array(list(gravities.values()))
    ids = np.argsort(values)
    values = values[ids]
    keys = keys[ids]
    colorline = "mediumvioletred"
    for i in range(len(gravities)):
        key = keys[i]
        value = values[i]
        lw = 1.25
        ax.vlines(value, 0, yvals[i], color=colorline, linewidths=lw)
        leq = True
        if value <= mean:
            P = norm.cdf(value, mean, std)
        else:
            leq = False
            P = 1 - norm.cdf(value, mean, std)
        leq_str = "\leq " if leq else ">"
        text = f"$g_{{{key}}}={value:.2f}$"
        if not shortanno:
            text += f", $P(X{leq_str}g_{{{key}}})={P:.4f}$"
        ax.text(
            value + 1,
            yvals[i] + 0.005,
            text,
            bbox={
                'facecolor': 'white',
                'alpha': 0.85,
                'pad': 1,
                'linewidth': 0.25,
                "edgecolor": colorline
            },
            fontsize=fontsize,
            horizontalalignment='right'
        )
    return ax


def mark_interval(ax, mean, std, fontsize, ci=0.95):
    conf_interval = norm.interval(ci, loc=mean, scale=std)
    X = np.linspace(*conf_interval, 1000)
    Y = norm.pdf(X, loc=mean, scale=std)
    ax.fill_between(X, Y, alpha=0.5)

    y = norm.pdf(mean, loc=mean, scale=std)

    text = f"{int(ci*100)}%-interval\n(in-distribution)"
    xy = (mean - 0.5, 0.1)
    xytext = (mean - 7, y - 0.075)
    ax.annotate(
        text=text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        horizontalalignment='center',
        arrowprops=dict(
            facecolor='black',
            arrowstyle="simple",
            linewidth=0.001,
            alpha=0.75
            # head_width=0,
            # head_length=0,
            # shrink=0.05
        ),
    )

    return ax


mean = gravities["Mars"]
std = 1.45
n_contexts = 100000
sampled_gravities = norm.rvs(loc=mean, scale=std, size=n_contexts)
figname = "gravity_distribution.png"

figsize = (5, 3)
sns.set_context("paper")
annofontsize = 8
shortanno = True
dpi = 250
fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_subplot(111)
# ax.hist(sampled_gravities, bins=100)
gravities_l0 = sampled_gravities[sampled_gravities <= 0]
gravities_g0 = sampled_gravities[sampled_gravities > 0]
# ax = sns.histplot(x=sampled_gravities, kde=True, ax=ax, stat="probability", cumulative=False, fill=False)
# ax = sns.kdeplot(x=sampled_gravities, cumulative=False)
X = np.linspace(-25, 5, 1000)
Y = norm.pdf(X, loc=mean, scale=std)

ax = mark_interval(ax, mean, std, annofontsize, ci=0.95)
ax.plot(X, Y)
ylims = ax.get_ylim()
ax = plot_gravities_vlines(ax, gravities, ylims[1], mean, std, annofontsize, shortanno)

title = f"$\mu = g_{{Mars}} = {gravities['Mars']:.2f}, \sigma = {std}$"
ax.set_title(title)
ylims = (0, ylims[1])
ax.set_ylim(*ylims)
xlims = ax.get_xlim()
xlims = (xlims[0], 0)
ax.set_xlim(*xlims)
ax.set_xlabel("gravity [m/s²]")
ax.set_ylabel("probability density")
fig.set_tight_layout(True)
fig.savefig(figname, bbox_inches="tight")
plt.show()

