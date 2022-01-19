import seaborn as sns
from matplotlib import pyplot as plt, colors as mplc
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from pathlib import Path

from experiments.common.plot.plotting_style import set_rc_params

from experiments.evaluation_protocol.evaluation_protocol import EvaluationProtocol, ContextFeature


def plot_evaluation_protocol(context_features, seed, n_contexts):
    set_rc_params()

    modes = ["A", "B", "C"]
    n_protocols = len(modes)
    fig = plt.figure(figsize=(6, 2), dpi=300)
    axes = fig.subplots(nrows=1, ncols=n_protocols + 1, sharex=True, sharey=True)
    for i in range(n_protocols):
        ax = axes[i]
        mode = modes[i]
        ep = EvaluationProtocol(context_features=context_features, mode=mode, seed=seed)
        cfs = ep.context_features
        cf0, cf1 = cfs

        xlim = (cf0.lower, cf0.upper)
        ylim = (cf1.lower, cf1.upper)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        contexts_train = ep.create_train_contexts(n=n_contexts)
        contexts_ES = ep.create_contexts_extrapolation_single(n=n_contexts)  # covers two quadrants
        contexts_EA = ep.create_contexts_extrapolation_all(n=n_contexts)
        contexts_I = ep.create_contexts_interpolation(n=n_contexts, contexts_forbidden=contexts_train)
        contexts_IC = ep.create_contexts_interpolation_combinatorial(n=n_contexts, contexts_forbidden=contexts_train)

        # Draw Quadrants
        patches = []

        colors = sns.color_palette("colorblind")
        color_T = colors[0]
        color_I = colors[1]
        color_ES = colors[2]
        color_EB = colors[3]
        color_IC = colors[4]
        # color_T = "cornflowerblue"
        # color_I = "red"
        # color_ES = "green"
        # color_EB = "blue"
        # color_IC = "yellow"
        ec_test = "black"
        markerfacecolor_alpha = 0.
        markersize = 10

        patch_kwargs = dict(zorder=0, linewidth=0., )

        # Extrapolation along single factors, Q_ES
        xy = (cf0.mid, cf1.lower)
        width = cf0.upper - cf0.mid
        height = cf1.mid - cf1.lower
        Q_ES = Rectangle(xy=xy, width=width, height=height, color=color_ES, **patch_kwargs)
        patches.append(Q_ES)

        xy = (cf0.lower, cf1.mid)
        height = cf1.upper - cf1.mid
        width = cf0.mid - cf0.lower
        Q_ES = Rectangle(xy=xy, width=width, height=height, color=color_ES, **patch_kwargs)
        patches.append(Q_ES)

        # Extrapolation along both factors
        xy = (cf0.mid, cf1.mid)
        height = cf1.upper - cf1.mid
        width = cf0.upper - cf0.mid
        Q_EB = Rectangle(xy=xy, width=width, height=height, color=color_EB, **patch_kwargs)
        patches.append(Q_EB)

        # Interpolation
        if mode == "A":
            xy = (cf0.lower, cf1.lower)
            height = cf1.mid - cf1.lower
            width = cf0.mid - cf0.lower
            Q_I = Rectangle(xy=xy, width=width, height=height, color=color_I, **patch_kwargs)
            patches.append(Q_I)
        elif mode == "B":
            xy = (cf0.lower, cf1.lower)
            width = cf0.mid - cf0.lower
            height = cf1.lower_constraint - cf1.lower
            Q_I = Rectangle(xy=xy, width=width, height=height, color=color_I, **patch_kwargs)
            patches.append(Q_I)

            width = cf0.lower_constraint - cf0.lower
            height = cf1.mid - cf1.lower
            Q_I = Rectangle(xy=xy, width=width, height=height, color=color_I, **patch_kwargs)
            patches.append(Q_I)

        # Combinatorial Interpolation
        if mode == "B":
            xy = (cf0.lower_constraint, cf1.lower_constraint)
            height = cf1.mid - cf1.lower_constraint
            width = cf0.mid - cf0.lower_constraint
            Q_IC = Rectangle(xy=xy, width=width, height=height, color=color_IC, **patch_kwargs)
            patches.append(Q_IC)
        elif mode == "C":
            xy = (cf0.lower, cf1.lower)
            height = cf1.mid - cf1.lower
            width = cf0.mid - cf0.lower
            Q_IC = Rectangle(xy=xy, width=width, height=height, color=color_IC, **patch_kwargs)
            patches.append(Q_IC)

        for patch in patches:
            ax.add_patch(patch)

        # Plot train context
        ax = sns.scatterplot(data=contexts_train, x=cf0.name, y=cf1.name, color=color_T, ax=ax, edgecolor=color_T,
                             s=markersize)

        # Extrapolation single
        ax = sns.scatterplot(data=contexts_ES, x=cf0.name, y=cf1.name,
                             color=mplc.to_rgba(color_ES, markerfacecolor_alpha), ax=ax, edgecolor=ec_test,
                             s=markersize)

        # Extrapolation all factors
        ax = sns.scatterplot(data=contexts_EA, x=cf0.name, y=cf1.name,
                             color=mplc.to_rgba(color_EB, markerfacecolor_alpha), ax=ax, edgecolor=ec_test,
                             s=markersize)

        # Interpolation (Train Distribution)
        if len(contexts_I) > 0:
            ax = sns.scatterplot(data=contexts_I, x=cf0.name, y=cf1.name,
                                 color=mplc.to_rgba(color_I, markerfacecolor_alpha), ax=ax, edgecolor=ec_test,
                                 s=markersize)

        # Combinatorial Interpolation
        if len(contexts_IC) > 0:
            ax = sns.scatterplot(data=contexts_IC, x=cf0.name, y=cf1.name,
                                 color=mplc.to_rgba(color_IC, markerfacecolor_alpha), ax=ax, edgecolor=ec_test,
                                 s=markersize)

        # Add axis descriptions
        ax.set_xlabel(cf0.name)
        if i == 0:
            ax.set_ylabel(cf1.name)
        ax.set_title(mode)
        unit_x = cf0.mid - cf0.lower
        unit_y = cf1.mid - cf1.lower
        aspect = unit_x / unit_y
        ax.set_aspect(aspect=aspect)
        yticks = [cf1.lower, cf1.mid, cf1.upper]
        ax.set_yticks(yticks)

    # Legend
    legend_elements = [
        Line2D([0], [0], label='Train Contexts', marker='o', color='w', markerfacecolor=color_T,
               markeredgecolor=color_T, markersize=8, linewidth=0),
        Line2D([0], [0], label='Test Contexts', marker='o', color='w', markerfacecolor='w', markeredgecolor=ec_test,
               markersize=8, linewidth=0),
        Patch(label="Interpolation", facecolor=color_I),
        Patch(label="Combinatorial Interpolation", facecolor=color_IC),
        Patch(label="Extrapolation (Single Factor)", facecolor=color_ES),
        Patch(label="Extrapolation (Both Factors)", facecolor=color_EB),
    ]
    ax = axes[-1]
    ax.set_axis_off()
    ax.legend(handles=legend_elements, loc="center", fontsize=8)

    # Adjust spaces for savin plot
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(Path("figures") / f"evaluation_protocol_traintest_distributions_seed{seed}.png", dpi=300, bbox_inches="tight")

    # Set tight layout for viewing in IDE
    fig.set_tight_layout(True)
    plt.show()


if __name__ == '__main__':
    cf0 = ContextFeature("gravity",  9., 9.5, 10., 11.)
    cf1 = ContextFeature("pole_length", 0.4, 0.5, 0.6, 0.8)
    seed = 1
    n_contexts = 100
    context_features = [cf0, cf1]
    plot_evaluation_protocol(context_features=context_features, seed=seed, n_contexts=n_contexts)