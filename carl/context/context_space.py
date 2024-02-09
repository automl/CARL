from __future__ import annotations

from typing import List

import warnings

import gymnasium.spaces as spaces
import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Hyperparameter,
    NormalFloatHyperparameter,
    NumericalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from typing_extensions import TypeAlias

from carl.utils.types import Context, Contexts

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

ContextFeature: TypeAlias = Hyperparameter
NumericalContextFeature: TypeAlias = NumericalHyperparameter
NormalFloatContextFeature: TypeAlias = NormalFloatHyperparameter
UniformFloatContextFeature: TypeAlias = UniformFloatHyperparameter
UniformIntegerContextFeature: TypeAlias = UniformIntegerHyperparameter
CategoricalContextFeature: TypeAlias = CategoricalHyperparameter


class ContextSpace(object):
    def __init__(self, context_space: dict[str, ContextFeature]) -> None:
        """Context space

        Parameters
        ----------
        context_space : dict[str, ContextFeature]
            Raw definition of the context space.
        """
        self.context_space = context_space

    @property
    def context_feature_names(self) -> list[str]:
        """
        Context feature names.

        Returns
        -------
        list[str]
            Context features names.
        """
        return list(self.context_space.keys())

    def insert_defaults(
        self, context: Context, context_keys: List[str] | None = None
    ) -> Context:
        """Insert default context if keys missing.

        Parameters
        ----------
        context : Context
            The context.
        context_keys : List[str] | None, optional
            Insert defaults only for certain keys, by default None.

        Returns
        -------
        Context
            The filled context with default values.
        """
        context_with_defaults = self.get_default_context()

        # insert defaults only for certain keys
        if context_keys:
            context_with_defaults = {
                key: context_with_defaults[key] for key in context_keys
            }

        context_with_defaults.update(context)
        return context_with_defaults

    def verify_context(self, context: Context) -> bool:
        """Verify context.

        Check if context feature names are correct and the
        values are in bounds.

        Parameters
        ----------
        context : Context
            The context to check.

        Returns
        -------
        bool
            True if valid, False if not.
        """
        is_valid = True
        cfs = self.context_feature_names
        for cfname, v in context.items():
            # Check if context feature exists in space
            # by checking name
            if cfname not in cfs:
                is_valid = False
                break

            # Check if context feature value is in bounds
            cf = self.context_space[cfname]
            if isinstance(cf, NumericalContextFeature):
                if not (cf.lower <= v <= cf.upper):
                    is_valid = False
                    break
        return is_valid

    def get_default_context(self) -> Context:
        """Get the default context from the context space.

        Returns
        -------
        Context
            Default context.
        """
        context = {cf.name: cf.default_value for cf in self.context_space.values()}
        return context

    def get_lower_and_upper_bound(
        self, context_feature_name: str
    ) -> tuple[float, float]:
        """Get lower and upper bounds for each context feature from the context space.

        Parameters
        ----------
        context_feature_name : str
            Name of context feature to get the bounds for.

        Returns
        -------
        tuple[float, float]
            Lower and upper bound as a tuple.
        """
        cf = self.context_space[context_feature_name]
        bounds = (cf.lower, cf.upper)
        return bounds

    def to_gymnasium_space(
        self, context_feature_names: List[str] | None = None, as_dict: bool = False
    ) -> spaces.Space:
        """Convert the context space to a gymnasium space (box).

        Parameters
        ----------
        context_feature_names : List[str] | None, optional
            The context features that should be included in the space, by default None.
            If it is None, then use all available context features.
        as_dict : bool, optional
            If True, create a dict gymnasium space, by default False. If False,
            context feature values as a vector.

        Returns
        -------
        spaces.Space
            Gymnasium space which can be used as an observation space.
        """
        if context_feature_names is None:
            context_feature_names = self.context_feature_names
        if as_dict:
            context_space = {}

            for cf_name in context_feature_names:
                context_feature = self.context_space[cf_name]
                if isinstance(context_feature, NumericalContextFeature):
                    context_space[context_feature.name] = spaces.Box(
                        low=context_feature.lower, high=context_feature.upper
                    )
                else:
                    context_space[context_feature.name] = spaces.Discrete(
                        len(context_feature.choices)
                    )
            return spaces.Dict(context_space)
        else:
            low = np.array(
                [self.context_space[cf].lower for cf in context_feature_names]
            )
            high = np.array(
                [self.context_space[cf].upper for cf in context_feature_names]
            )

            return spaces.Box(low=low, high=high, dtype=np.float32)

    def sample_contexts(
        self, context_keys: List[str] | None = None, size: int = 1
    ) -> Context | List[Contexts]:
        """Sample a number of contexts from the space.

        Parameters
        ----------
        context_keys : List[str] | None, optional
            The context feature names to sample for, by default None
        size : int, optional
            The number of contexts to sample, by default 1

        Returns
        -------
        Context | List[Contexts]
            A context or list of contexts. Always filled with defaults
            if context features missing.

        Raises
        ------
        ValueError
            When elements of context_keys are not valid.
        """
        if context_keys is None:
            context_keys = self.context_space.keys()
        else:
            for key in context_keys:
                if key not in self.context_space.keys():
                    raise ValueError(f"Invalid context feature name: {key}")

        contexts = []
        for _ in range(size):
            context = {cf.name: cf.rvs() for cf in self.context_space.values()}
            context = self.insert_defaults(context, context_keys)
            contexts += [context]

        if size == 1:
            return contexts[0]
        else:
            return contexts
