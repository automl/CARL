"""
Boxplot
- number of context features
- percentage of continuous CFs
- number of CFs changing the dynamics
- number of CFs changing the reward
"""
if __name__ == "__main__":
    from typing import List

    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    import carl.envs

    global_vars = vars(carl.envs)
    vars = {
        k: v for k, v in global_vars.items() if "Env" in k or "Meta" in k or "CARL" in k
    }
    env_names = [n for n in vars.keys() if "bounds" not in n and "defaults" not in n]

    env_context_feature_names = {}

    context_feature_names = []  # type: List[str]
    dfs = []  # type: List[pd.DataFrame]
    n_context_features_per_env = []
    n_float_cfs = 0
    for env_name in env_names:
        defaults = pd.Series(vars[env_name + "_defaults"])
        n_context_features_per_env.append(len(defaults))
        bounds = vars[env_name + "_bounds"]
        bounds_vals = list(bounds.values())
        n_float_cfs += np.sum([1 for v in bounds_vals if v[2] == float])
        env_context_feature_names[env_name] = defaults.keys()

    n_context_features = np.sum(n_context_features_per_env)

    n_reward_changing = 7
    n_dynami_changing = 129

    env_names.extend(["CARLMarioEnv", "CARLRnaDesignEnv"])
    n_context_features += 3 + 5
    n_float_cfs += 0 + 0  # integers == continuous?

    percentage_float_cfs = n_float_cfs / n_context_features

    dfp = pd.Series(
        {
            "$n_{{total}}$": n_context_features,
            "$n_{{dynamics}}$": n_dynami_changing,
            "$n_{{reward}}$": n_reward_changing,
            "$n_{{continuous}}$": n_float_cfs,
        }
    )
    dfp.name = "Context Features"
    vals = dfp.to_numpy()
    labels = dfp.index

    fontsize = 15
    sns.set_style("whitegrid")
    figsize = (2.5, 2)
    dpi = 200
    fname = "plots/context_feature_statistics.png"
    p = Path(fname)
    p.parent.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax = sns.barplot(x=vals, y=labels, ax=ax, palette="colorblind", orient="h")
    xmin = 0
    xmax = max(vals)
    ax.set_xlim(xmin, xmax)
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)

    for i, label in enumerate(labels):
        x = 10
        y = i + 0.2
        text = f"{label} = {vals[i]:.0f}"
        ax.text(x, y, text, fontsize=fontsize)

    fig.set_tight_layout(True)
    plt.show()

    fig.savefig(fname, bbox_inches="tight")
    dfp.to_csv(Path(fname).parent / "context_feature_statistics.csv")
