"""
Following the evaluation protocols proposed in Kirk et al., 2021.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import matplotlib.colors as mplc
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from typing import Union, Dict, Optional, List


class ContextFeature(object):
    """
    Only continuous context features.
    """
    def __init__(
            self,
            name: str,
            lower: float,
            lower_constraint: float,
            mid: float,
            upper: float,
    ):
        self.name = name
        self.lower = lower
        self.lower_constraint = lower_constraint
        self.mid = mid
        self.upper = upper

        assert lower < lower_constraint < mid < upper


class EvaluationProtocol2D(object):
    """
    Only for continuous context features (so far).
    """
    def __init__(
            self,
            context_features: List[ContextFeature],
            mode: str,
            seed: Optional[int] = None
    ):
        if len(context_features) > 2:
            raise ValueError("Evaluation protocol only supports 2 context features so far.")
        self.context_features = context_features
        self.mode = self.check_mode(mode=mode)
        self._seed = None
        self._rng = None
        self.seed = seed

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        self._seed = seed
        self._rng = np.random.default_rng(seed=self.seed)

    @staticmethod
    def check_mode(mode):
        available_modes = ["A", "B", "C"]
        if mode not in available_modes:
            raise ValueError(f"Mode {mode} not in available modes {available_modes}.")
        return mode

    def create_train_contexts(self, n: int) -> pd.DataFrame:
        contexts = None
        n_cfs = len(self.context_features)
        if self.mode == "A":
            contexts = self.sample_train_A(n=n)
        else:
            sample_function = None
            if self.mode == "B":
                sample_function = self.sample_train_B
            elif self.mode == "C":
                sample_function = self.sample_train_C

            n_cfs = len(self.context_features)  # number of context feautures
            if n % n_cfs != 0:
                warnings.warn(f"Requested number of contexts {n} is not divisible by number of contexts "
                              f"{n_cfs} without rest. You will get less contexts than {n}.")

            n_i = n // n_cfs
            contexts = []
            for i in range(n_cfs):
                C = sample_function(i, n_i)
                contexts.append(C)
            contexts = pd.concat(contexts)
            contexts.reset_index(inplace=True, drop=True)

        return contexts

    def sample_train_A(self, n):
        contexts = {}
        for cf in self.context_features:
            C = self._rng.uniform(cf.lower, cf.mid, size=n)
            contexts[cf.name] = C
        contexts = pd.DataFrame(contexts)
        return contexts

    def sample_train_B(self, index_cf, n) -> pd.DataFrame:
        contexts = {}
        for j in range(len(self.context_features)):
            cf = self.context_features[j]
            if j != index_cf:
                samples = self._rng.uniform(cf.lower, cf.mid, size=n)
            else:
                samples = self._rng.uniform(cf.lower, cf.lower_constraint, size=n)
            contexts[cf.name] = samples
        contexts = pd.DataFrame(contexts)
        return contexts

    def sample_train_C(self, index_cf, n) -> pd.DataFrame:
        contexts = {}
        for j in range(len(self.context_features)):
            cf = self.context_features[j]
            if j != index_cf:
                samples = self._rng.uniform(cf.lower, cf.mid, size=n)
            else:
                samples = np.array([cf.lower] * n)  # TODO: or cf.default?
            contexts[cf.name] = samples
        contexts = pd.DataFrame(contexts)
        return contexts

    def create_contexts_extrapolation_single(self, n) -> pd.DataFrame:
        def sample_function(index_cf, n) -> pd.DataFrame:
            contexts = {}
            for j in range(len(self.context_features)):
                cf = self.context_features[j]
                if j != index_cf:
                    samples = self._rng.uniform(cf.lower, cf.mid, size=n)
                else:
                    samples = self._rng.uniform(cf.mid, cf.upper, size=n)
                contexts[cf.name] = samples
            contexts = pd.DataFrame(contexts)
            return contexts

        n_cfs = len(self.context_features)  # number of context feautures
        if n % n_cfs != 0:
            warnings.warn(f"Requested number of contexts {n} is not divisible by number of contexts "
                          f"{n_cfs} without rest. You will get less contexts than {n}.")
        n_i = n // n_cfs
        contexts = []
        for i in range(n_cfs):
            C = sample_function(i, n_i)
            contexts.append(C)
        contexts = pd.concat(contexts)
        contexts.reset_index(inplace=True, drop=True)
        return contexts

    def create_contexts_extrapolation_all(self, n):
        def sample_function(n) -> pd.DataFrame:
            contexts = {}
            for j in range(len(self.context_features)):
                cf = self.context_features[j]
                samples = self._rng.uniform(cf.mid, cf.upper, size=n)
                contexts[cf.name] = samples
            contexts = pd.DataFrame(contexts)
            return contexts

        n_cfs = len(self.context_features)  # number of context features
        if n % n_cfs != 0:
            warnings.warn(f"Requested number of contexts {n} is not divisible by number of contexts "
                          f"{n_cfs} without rest. You will get less contexts than {n}.")
        n_i = n // n_cfs
        contexts = [sample_function(n_i)]
        contexts = pd.concat(contexts)
        contexts.reset_index(inplace=True, drop=True)
        return contexts

    def create_contexts_interpolation(self, n, contexts_forbidden: Optional[pd.DataFrame] = None):
        contexts = pd.DataFrame()
        if self.mode in ["A", "B"]:
            contexts = self.create_train_contexts(n=n)
            if contexts_forbidden is not None:
                contexts = self.recreate(
                    contexts=contexts, contexts_forbidden=contexts_forbidden, create_function=self.create_train_contexts,
                    max_resample=10
                )
        return contexts

    def recreate(self, contexts, contexts_forbidden, create_function, max_resample: int = 10):
        n_new = None
        counter = 0
        while n_new != 0 or counter > max_resample:
            resample_ids = []
            for i in range(len(contexts)):
                for j in range(len(contexts_forbidden)):
                    c = contexts.iloc[i].to_numpy()  # because of the order of the list of context features the columns are in same order
                    c_f = contexts_forbidden.iloc[j].to_numpy()
                    is_duplicated = np.any(c == c_f)
                    if is_duplicated:
                        resample_ids.append(i)
            # resample_ids = np.any(contexts == contexts_forbidden, axis=1)
            n_new = len(resample_ids)
            if n_new > 0:
                print(f"Resample! n_new = {n_new}, counter = {counter}")
                contexts_new = create_function(n=n_new)
                contexts.iloc[resample_ids] = contexts_new
            counter += 1

        return contexts

    def create_contexts_interpolation_combinatorial(self, n, contexts_forbidden: Optional[pd.DataFrame] = None):
        def create(n):
            contexts = pd.DataFrame()
            if self.mode == "B":
                def sample_function(n) -> pd.DataFrame:
                    contexts = {}
                    for j in range(len(self.context_features)):
                        cf = self.context_features[j]
                        samples = self._rng.uniform(cf.lower_constraint, cf.mid, size=n)
                        contexts[cf.name] = samples
                    contexts = pd.DataFrame(contexts)
                    return contexts
                n_cfs = len(self.context_features)  # number of context features
                n_i = n // n_cfs
                contexts = [sample_function(n_i)]
                contexts = pd.concat(contexts)
                contexts.reset_index(inplace=True, drop=True)

            elif self.mode == "C":
                sample_function = self.sample_train_A
                n_cfs = len(self.context_features)  # number of context features
                n_i = n // n_cfs
                contexts = [sample_function(n_i)]
                contexts = pd.concat(contexts)
                contexts.reset_index(inplace=True, drop=True)
            return contexts

        contexts = create(n)
        if len(contexts) > 0 and contexts_forbidden is not None:
            contexts = self.recreate(
                contexts=contexts, contexts_forbidden=contexts_forbidden,
                create_function=create,
                max_resample=10
            )

        return contexts


if __name__ == '__main__':
    cf0 = ContextFeature("g",  9., 9.5, 10., 11.)
    cf1 = ContextFeature("l", 0.4, 0.5, 0.6, 0.8)
    # ep = EvaluationProtocol2D(
    #     context_features=[cf0, cf1],
    #     mode="B"
    # )
    # contexts = ep.create_train_contexts(n=10)

    n_contexts = 100
    modes = ["A", "B", "C"]
    n_protocols = len(modes)
    fig = plt.figure(figsize=(8, 6), dpi=300)
    axes = fig.subplots(nrows=1, ncols=n_protocols, sharex=True, sharey=True)
    for i in range(n_protocols):
        ax = axes[i]
        mode = modes[i]
        ep = EvaluationProtocol2D(context_features=[cf0, cf1], mode=mode)
        cfs = ep.context_features
        cf0, cf1 = cfs

        xlim = (cf0.lower, cf0.upper)
        ylim = (cf1.lower, cf1.upper)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        contexts_train = ep.create_train_contexts(n=n_contexts)
        contexts_ES = ep.create_contexts_extrapolation_single(n=2 * n_contexts)  # covers two quadrants
        contexts_EA = ep.create_contexts_extrapolation_all(n=n_contexts)
        contexts_I = ep.create_contexts_interpolation(n=n_contexts, contexts_forbidden=contexts_train)
        contexts_IC = ep.create_contexts_interpolation_combinatorial(n=n_contexts, contexts_forbidden=contexts_train)

        # Draw Quadrants
        patches = []

        color_T = "cornflowerblue"
        color_I = "red"
        color_ES = "green"
        color_EB = "blue"
        color_IC = "yellow"
        ec_test = "black"
        markerfacecolor_alpha = 0.

        patch_kwargs = dict(zorder=0, linewidth=0.,)

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
        ax = sns.scatterplot(data=contexts_train, x=cf0.name, y=cf1.name, color=color_T, ax=ax, edgecolor=color_T)

        # Extrapolation single
        ax = sns.scatterplot(data=contexts_ES, x=cf0.name, y=cf1.name, color=mplc.to_rgba(color_ES, markerfacecolor_alpha), ax=ax, edgecolor=ec_test)

        # Extrapolation all factors
        ax = sns.scatterplot(data=contexts_EA, x=cf0.name, y=cf1.name, color=mplc.to_rgba(color_EB, markerfacecolor_alpha), ax=ax, edgecolor=ec_test)

        # Interpolation (Train Distribution)
        if len(contexts_I) > 0:
            ax = sns.scatterplot(data=contexts_I, x=cf0.name, y=cf1.name, color=mplc.to_rgba(color_I, markerfacecolor_alpha), ax=ax, edgecolor=ec_test)

        # Combinatorial Interpolation
        if len(contexts_IC) > 0:
            ax = sns.scatterplot(data=contexts_IC, x=cf0.name, y=cf1.name, color=mplc.to_rgba(color_IC, markerfacecolor_alpha), ax=ax, edgecolor=ec_test)

        # Add axis descriptions
        ax.set_xlabel(cf0.name)
        if i == 0:
            ax.set_ylabel(cf1.name)
        ax.set_title(mode)

        # Legend
        legend_elements = [
            Line2D([0], [0], label='Train Contexts', marker='o', color='w', markerfacecolor=color_T,
                   markeredgecolor=color_T, markersize=10, linewidth=0),
            Line2D([0], [0], label='Test Contexts', marker='o', color='w', markerfacecolor='w', markeredgecolor=ec_test,
                   markersize=10, linewidth=0),
            Patch(label="Interpolation", facecolor=color_I),
            Patch(label="Combinatorial Interpolation", facecolor=color_IC),
            Patch(label="Extrapolation (Single Factor)", facecolor=color_ES),
            Patch(label="Extrapolation (Both Factors)", facecolor=color_EB),
        ]

        if i == n_protocols - 1:
            ax.legend(handles=legend_elements)

    fig.set_tight_layout(True)
    plt.show()




