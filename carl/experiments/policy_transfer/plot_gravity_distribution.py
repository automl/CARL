import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from policy_transfer import gravities
import json
from carl.experiments.policy_transfer.sample_lunarlander_contexts import fname as context_train_fname


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
    ax.fill_between(X, Y, alpha=0.3)

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


def mark_planets(ax, n_planets):
    ylims = ax.get_ylim()
    ymax = ylims[1]
    X = np.linspace(-25, 0, n_planets)
    X[-1] = -1e-3
    colorline = "mediumvioletred"
    for i in range(n_planets):
        lw = 3
        x = X[i]
        ax.vlines(x, 0, ymax, color=colorline, linewidths=lw, label="test")

    return ax


if __name__ == '__main__':
    mean = gravities["Mars"]
    std = 1.45
    n_contexts = 100000
    sampled_gravities = norm.rvs(loc=mean, scale=std, size=100)

    plot_exp_0 = True
    exp_id = "" if plot_exp_0 else "_exp1"
    figname = f"gravity_distribution{exp_id}.png"
    # experiment 0: Gaussian distribution centered around Mars
    # experiment 1: uniform intervals (-20, -15), (-5, 1e-3)

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

    if plot_exp_0:
        X = np.linspace(-25, 5, 1000)
        Y = norm.pdf(X, loc=mean, scale=std)

        ax = mark_interval(ax, mean, std, annofontsize, ci=0.95)
        ax.plot(X, Y)
    else:
        with open(context_train_fname, 'r') as file:
            contexts_train = json.load(file)
        Y = [c["GRAVITY_Y"] for c in contexts_train.values()]
        ax = sns.histplot(Y, ax=ax, bins=len(Y) // 2, label="train")

    # ax = sns.histplot(x=sampled_gravities,kde=False,ax=ax,stat="density",cumulative=False,fill=True,color="black",bins=100)
    ylims = ax.get_ylim()

    # mark planets
    if plot_exp_0:
        ax = plot_gravities_vlines(ax, gravities, ylims[1], mean, std, annofontsize, shortanno)
    else:
        ax = mark_planets(ax, n_planets=10)

    if plot_exp_0:
        title = f"$\mu = g_{{Mars}} = {gravities['Mars']:.2f}, \sigma = {std}$"
        ax.set_title(title)
    else:
        #handle_test =
        #legend_handles = [handle_train, handle_test]
        ax.legend()
        handles = ax.get_legend().legendHandles
        new_handles = handles[-2:]
        ax.legend(handles=new_handles, labels=["test", "train"])  # TODO dont hardcode
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

