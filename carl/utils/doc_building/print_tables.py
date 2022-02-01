if __name__ == "__main__":
    from typing import List

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.offsetbox import AnchoredText

    import carl.envs

    def plot_context_feature_freq(
        context_feature_names: List[str], fname: str = ""
    ) -> None:
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
                "friction": ["friction"],
            }
            cfs = []
            for cf in context_feature_names:
                cf = cf.lower()
                for alias, alias_values in aliases.items():
                    longs = [a for a in alias_values if len(a) > 2]
                    shorts = [a for a in alias_values if len(a) <= 2]
                    if np.any([a in cf for a in longs]) or np.any(
                        [
                            cf == a
                            or cf[-len(a) :] == a
                            or cf[: len(a) + 1] == a + "_"
                            or cf == a
                            for a in shorts
                        ]
                    ):
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
        ax.set_yticklabels(cf_names_unique, ha="right", fontsize=8)
        ax.set_title(
            f"Context Feature Frequency (lazily filtered, $n = {np.sum(counts_orig)}$)",
            fontsize=10,
        )
        ax.grid(axis="x", which="both")

        if filter_single_occurrences:
            text = "Single occurrences:\n" + "\n".join(
                [f"{cf}" for cf in cf_names_single_occurrences]
            )
            at2 = AnchoredText(
                text,
                loc="lower right",
                prop=dict(size=8),
                frameon=False,
                # bbox_to_anchor=(0., 1.),
                # bbox_transform=ax.transAxes
            )
            at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at2)

        fig.set_tight_layout(True)

        if fname:
            fig.savefig(fname, bbox_inches="tight")

        plt.show()

    global_vars = vars(carl.envs)
    vars = {
        k: v for k, v in global_vars.items() if "Env" in k or "Meta" in k or "CARL" in k
    }
    env_names = [n for n in vars.keys() if "bounds" not in n and "defaults" not in n]
    env_context_feature_names = {}

    context_feature_names = []
    dfs = []
    n_context_features = []
    for env_name in env_names:
        defaults = pd.Series(vars[env_name + "_defaults"])
        n_context_features.append(len(defaults))
        bounds = vars[env_name + "_bounds"]
        print_bounds = {}
        for k, v in bounds.items():
            lower = v[0]
            upper = v[1]
            datatype = v[2]
            if datatype == "categorical":
                pass
            else:
                datatype = datatype.__name__
            print_bounds[k] = (lower, upper, datatype)
        # print_bounds = {k: (v[0], v[1], v[2].__name__) for k, v in bounds.items()}
        bounds = pd.Series(print_bounds)

        defaults.sort_index(inplace=True)
        bounds.sort_index(inplace=True)

        df = pd.DataFrame()
        df["Default"] = defaults
        df["Bounds"] = [b[:2] for b in bounds]
        df["Bounds"].replace(to_replace=(None, None), value="-", inplace=True)
        df["Bounds"][df["Bounds"] == (None, None)] = "-"
        df["Type"] = [b[2] for b in bounds]
        rows = df.index
        context_feature_names.extend(rows)

        # if "Acro" in env_name:
        #     special_format_cols = ["max_velocity_1", "max_velocity_2"]
        #     for b in bounds:
        #         print(f"({b[0]}, {b[1]})", type(b[0]))

        # index = [r.lower().replace("_", " ") for r in rows]
        # tuples = [(env_name, ind) for ind in index]
        # index = pd.MultiIndex.from_tuples(tuples, names=["Environment", "Context Feature"])
        # df.index = index
        dfs.append(df)
        env_context_feature_names[env_name] = list(defaults.keys())

    df_cf_defbounds = pd.concat(dfs)

    # Requires latex \usepackage{booktabs}
    bold_rows = False
    table_str = df_cf_defbounds.to_latex(
        header=True,
        index=True,
        index_names=True,
        float_format="{:0.2f}".format,
        bold_rows=bold_rows,
        caption=(
            "Context Features: Defaults and Bounds",
            "Context Features: Defaults and bounds for each environment.",
        ),
        label="tab:context_features_defaults_bounds",
        # position="c",?
    )
    df_cf_defbounds_fname = "utils/context_features_defaults_bounds.tex"
    with open(df_cf_defbounds_fname, "w") as file:
        file.write(table_str)

    for env_name, df in zip(env_names, dfs):
        # index = [ind[1] for ind in df.index]  # no multi-index anymore
        # df.index = index
        df.index.name = "Context Feature"
        df.reset_index(level=0, inplace=True)
        table_str = df.to_latex(
            header=True,
            index=False,
            index_names=True,
            float_format="{:0.2f}".format,
            bold_rows=bold_rows,
            caption=(
                f"{env_name}",  #: Context Features with Defaults, Bounds and Types",
                f"{env_name}",  #: Context Features with Defaults, Bounds and Types"
            ),
            label=f"tab:context_features_defaults_bounds_{env_name}",
            # position="c",?
        )
        table_str = table_str.replace("{table}", "{subtable}")
        table_str = table_str.replace(
            r"\begin{subtable}", r"\begin{subtable}{0.4\textwidth}"
        )
        # print(table_str)
        fname = f"utils/context_features_defaults_bounds_{env_name}.tex"
        with open(fname, "w") as file:
            file.write(table_str)

    # plot_context_feature_freq(context_feature_names=context_feature_names, fname="utils/context_feature_freq.png")
    def plot_statistics(
        env_names: List[str], n_context_features: int, fname: str = ""
    ) -> None:
        fig = plt.figure(figsize=(5, 7), dpi=200)
        ax = fig.add_subplot(111)
        ax.barh(env_names, n_context_features)
        ax.set_yticklabels(env_names, ha="right", fontsize=8)
        ax.set_title("TODO", fontsize=10)
        ax.grid(axis="x", which="both")

        fig.set_tight_layout(True)

        if fname:
            fig.savefig(fname, bbox_inches="tight")

        plt.show()

    # collect size of state space
    calc_new = False
    if calc_new:
        state_space_sizes = []
        action_space_sizes = []
        for env_name in env_names:
            env = eval(env_name)(hide_context=True)
            state = env.observation_space
            action_space = env.action_space
            env.close()
            state_space_sizes.append(state.shape)
            action_space_sizes.append(action_space.shape)
            print(env_name, state.shape)
    else:
        env_names = [
            "CARLMountainCarEnv",
            "CARLPendulumEnv",
            "CARLAcrobotEnv",
            "CARLCartPoleEnv",
            "CARLMountainCarContinuousEnv",
            "CARLLunarLanderEnv",
            "CARLVehicleRacingEnv",
            "CARLBipedalWalkerEnv",
            "CARLAnt",
            "CARLHalfcheetah",
            "CARLHumanoid",
            "CARLFetch",
            "CARLGrasp",
            "CARLUr5e",
        ]

        # hide_context = False
        state_space_sizes = [
            (13,),
            (8,),
            (15,),
            (10,),
            (12,),
            (24,),
            (96, 96, 3),
            (44,),
            (94,),
            (29,),
            (304,),
            (110,),
            (141,),
            (75,),
        ]
        # hide_context = True
        state_space_sizes = [
            (2,),
            (3,),
            (6,),
            (4,),
            (2,),
            (8,),
            (96, 96, 3),
            (24,),
            (87,),
            (23,),
            (299,),
            (101,),
            (132,),
            (66,),
        ]  # , (11,), (64, 64, 3)]

        action_space_sizes = [
            (3,),
            (1,),
            (3,),
            (2,),
            (1,),
            (4,),
            (3,),
            (4,),
            (8,),
            (6,),
            (17,),
            (10,),
            (19,),
            (6,),
        ]

    fname = "utils/env_statistics.png"

    env_names.append("CARLRnaDesignEnv")
    n_context_features.append(5)
    state_space_sizes.append((11,))
    action_space_sizes.append((8,))  # 2 types with 4 actions
    env_context_feature_names["CARLRnaDesignEnv"] = [
        "mutation_threshold",
        "reward_exponent",
        "state_radius",
        "dataset",
        "target_structure_ids",
    ]

    env_names.append("CARLMarioEnv")
    n_context_features.append(3)
    state_space_sizes.append((64, 64, 3))
    action_space_sizes.append((10,))
    env_context_feature_names["CARLMarioEnv"] = ["level_index", "noise", "mario_state"]
    # plot_statistics(env_names, n_context_features, fname=fname)

    s_sizes = [s[0] for s in state_space_sizes if len(s) == 1]
    # plot_statistics(env_names, s_sizes)
    a_sizes = [s[0] for s in action_space_sizes]
    # plot_statistics(env_names, a_sizes)

    sns.set_style("darkgrid")
    sns.set_context("paper")
    n_envs = len(env_names)
    n_axes = 2
    figsize = (3, 2 * n_axes)
    dpi = 200
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(nrows=n_axes, ncols=1)

    ax = axes[0]
    ax = sns.histplot(x=s_sizes, ax=ax, bins=n_envs)
    ax.set_xlabel("State Space Size")
    ax.set_ylabel("$n_{{envs}}$")

    ax = axes[1]
    ax = sns.histplot(x=a_sizes, ax=ax, bins=n_envs)
    ax.set_xlabel("Action Space Size")
    ax.set_ylabel("$n_{{envs}}$")

    fig.set_tight_layout(True)

    plt.show()
