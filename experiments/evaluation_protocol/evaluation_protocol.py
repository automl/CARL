"""
Following the evaluation protocols proposed in Kirk et al., 2021.
"""
import pandas as pd
import numpy as np
import warnings
from typing import Optional, List


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

        assert lower < lower_constraint < mid < upper or lower > lower_constraint > mid > upper


class EvaluationProtocol(object):
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
        contexts = [sample_function(n)]
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
                contexts = [sample_function(n)]
                contexts = pd.concat(contexts)
                contexts.reset_index(inplace=True, drop=True)

            elif self.mode == "C":
                sample_function = self.sample_train_A
                contexts = [sample_function(n)]
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
