import tabulate
import numpy as np
from typing import List
from src.envs import *

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import pandas as pd


def plot_context_feature_freq(context_feature_names: List[str], fname: str =""):
    filter_cf_names = True
    if filter_cf_names:
        aliases = {
            "mass": ["mass", "m"],
            "geometry": ["length", "l", "height", "h", "width", "w", "radius"],
            "power": ["power"],
            "gravity": ["gravity", "g"],
            "force": ["force"],
            "position": ["position", "distance"],
            "velocity": ["velocity", "speed"],
            "torque": ["torque"],
            "damping": ["damping"],
            "friction": ["friction"]
        }
        cfs = []
        for cf in context_feature_names:
            cf = cf.lower()
            for alias, alias_values in aliases.items():
                longs = [a for a in alias_values if len(a) > 2]
                shorts = [a for a in alias_values if len(a) <= 2]
                if np.any([a in cf for a in longs]) \
                        or np.any([cf == a or cf[-len(a):] == a or cf[:len(a)+1] == a + "_" or cf == a for a in shorts]):
                    cf = alias
            cfs.append(cf)

        context_feature_names = cfs

    cf_names_unique, counts = np.unique(context_feature_names, return_counts=True)
    counts_orig = counts
    ids = np.argsort(counts)
    cf_names_unique = cf_names_unique[ids]
    counts = counts[ids]

    filter_single_occurrences = True
    cf_names_single_occurrences = []
    if filter_single_occurrences:
        ids = counts == 1
        cf_names_single_occurrences = cf_names_unique[ids]
        cf_names_single_occurrences.sort()
        cf_names_unique = cf_names_unique[~ids]
        counts = counts[~ids]

    # context feature frequency
    fig = plt.figure(figsize=(5, 7), dpi=200)
    ax = fig.add_subplot(111)
    ax.barh(cf_names_unique, counts)
    ax.set_yticklabels(cf_names_unique, ha='right', fontsize=8)
    ax.set_title(f"Context Feature Frequency (lazily filtered, $n = {np.sum(counts_orig)}$)", fontsize=10)
    ax.grid(axis="x", which="both")

    if filter_single_occurrences:
        text = "Single occurrences:\n" + "\n".join([f"{cf}" for cf in cf_names_single_occurrences])
        at2 = AnchoredText(text,
                           loc='lower right', prop=dict(size=8), frameon=False,
                           # bbox_to_anchor=(0., 1.),
                           # bbox_transform=ax.transAxes
                           )
        at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at2)

    fig.set_tight_layout(True)

    if fname:
        fig.savefig(fname, bbox_inches="tight")

    plt.show()


global_vars = vars()
vars = {k: v for k, v in global_vars.items() if "Env" in k or "Meta" in k}
env_names = [n for n in vars.keys() if "bounds" not in n and "defaults" not in n]
env_context_feature_names = {}

context_feature_names = []
dfs = []
for env_name in env_names:
    defaults = pd.Series(vars[env_name + "_defaults"])
    bounds = vars[env_name + "_bounds"]
    bounds = {k: (v[0], v[1], v[2].__name__) for k, v in bounds.items()}
    bounds = pd.Series(bounds)

    df = pd.DataFrame()
    df["Default"] = defaults
    df["Bounds"] = bounds
    rows = df.index
    context_feature_names.extend(rows)
    index = [r.lower().replace("_", " ") for r in rows]
    tuples = [(env_name, ind) for ind in index]
    index = pd.MultiIndex.from_tuples(tuples, names=["Environment", "Context Feature"])
    df.index = index
    dfs.append(df)

    env_context_feature_names[env_name] = defaults.keys()

df_cf_defbounds = pd.concat(dfs)

# Requires latex \usepackage{booktabs}
table_str = df_cf_defbounds.to_latex(
    header=True,
    index=True,
    index_names=True,
    float_format="{:0.2f}".format,
    bold_rows=True,
    caption=("Context Features: Defaults and Bounds", "Context Features: Defaults and bounds for each environment."),
    label="tab:context_features_defaults_bounds",
    # position="c",?
)
df_cf_defbounds_fname = "utils/context_features_defaults_bounds.tex"
with open(df_cf_defbounds_fname, 'w') as file:
    file.write(table_str)

for env_name, df in zip(env_names, dfs):
    index = [ind[1] for ind in df.index]  # no multi-index anymore
    df.index = index
    table_str = df.to_latex(
        header=True,
        index=True,
        index_names=True,
        float_format="{:0.2f}".format,
        bold_rows=True,
        caption=(
            f"Context Features: Defaults and Bounds ({env_name})",
            f"Context Features: Defaults and bounds for {env_name}."
        ),
        label=f"tab:context_features_defaults_bounds_{env_name}",
        # position="c",?
    )
    fname = f"utils/context_features_defaults_bounds_{env_name}.tex"
    with open(fname, 'w') as file:
        file.write(table_str)

plot_context_feature_freq(context_feature_names=context_feature_names, fname="utils/context_feature_freq.png")


