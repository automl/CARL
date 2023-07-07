from __future__ import annotations

from typing import Any, List, Union

import gymnasium.spaces as spaces
import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Hyperparameter,
    NormalFloatHyperparameter,
    NumericalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from omegaconf import DictConfig
from typing_extensions import TypeAlias

from carl.context.search_space_encoding import search_space_to_config_space
from carl.utils.types import Context, Contexts

ContextFeature: TypeAlias = Hyperparameter
NumericalContextFeature: TypeAlias = NumericalHyperparameter
NormalFloatContextFeature: TypeAlias = NormalFloatHyperparameter
UniformFloatContextFeature: TypeAlias = UniformFloatHyperparameter
UniformIntegerContextFeature: TypeAlias = UniformIntegerHyperparameter
CategoricalContextFeature: TypeAlias = CategoricalHyperparameter


class ContextSpace(object):
    def __init__(self, context_space: dict[str, ContextFeature]) -> None:
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
        context_with_defaults = self.get_default_context()

        # insert defaults only for certain keys
        if context_keys:
            context_with_defaults = {
                key: context_with_defaults[key] for key in context_keys
            }

        context_with_defaults.update(context)
        return context_with_defaults

    def verify_context(self, context: Context) -> bool:
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
        context = {cf.name: cf.default_value for cf in self.context_space.values()}
        return context

    def get_lower_and_upper_bound(
        self, context_feature_name: str
    ) -> tuple[float, float]:
        cf = self.context_space[context_feature_name]
        bounds = (cf.lower, cf.upper)
        return bounds

    def to_gymnasium_space(
        self, context_feature_names: List[str] | None = None, as_dict: bool = False
    ) -> spaces.Space:
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
                    raise ValueError(
                        f"Context features must be of type NumericalContextFeature."
                        f"Got {type(context_feature)}."
                    )
            print(context_space)
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
        if context_keys is None:
            context_keys = self.context_space.keys()
        else:
            for key in context_keys:
                if key not in self.context_space.keys():
                    raise ValueError(f"Invalid context feature name: {key}")

        contexts = []
        for _ in range(size):
            context = {cf.name: cf.sample() for cf in self.context_space.values()}
            context = self.insert_defaults(context, context_keys)
            contexts += [context]

        if size == 1:
            return contexts[0]
        else:
            return contexts