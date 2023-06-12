from __future__ import annotations

from typing import Any, List, Union

import gymnasium.spaces as spaces
import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    Hyperparameter,
    NormalFloatHyperparameter,
    NumericalHyperparameter,
    UniformFloatHyperparameter,
)
from omegaconf import DictConfig
from carl.context.search_space_encoding import search_space_to_config_space
from typing_extensions import TypeAlias
from carl.utils.types import Context, Contexts

ContextFeature: TypeAlias = Hyperparameter
NumericalContextFeature: TypeAlias = NumericalHyperparameter
NormalFloatContextFeature: TypeAlias = NormalFloatHyperparameter
UniformFloatContextFeature: TypeAlias = UniformFloatHyperparameter


class ContextSpace(object):
    def __init__(
            self,
            context_space: dict[str, ContextFeature]
        ) -> None:
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
    
    def insert_defaults(self, context: Context) -> Context:
        context_with_defaults = self.get_default_context()
        context_with_defaults.update(context)
        return context_with_defaults
    
    def get_default_context(self) -> Context:
        context = {cf.name: cf.default_value for cf in self.context_space.values()}
        return context
    
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
                    context_space[context_feature.name] = spaces.Box(low=context_feature.lower, high=context_feature.upper)
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


class ContextSampler(ConfigurationSpace):
    def __init__(
        self,
        context_distributions: list[ContextFeature] | str | DictConfig,
        context_space: ContextSpace,
        seed: int,
        name: str | None = None,
    ):
        self.context_distributions = context_distributions
        super().__init__(name=name, seed=seed)

        if isinstance(context_distributions, list):
            self.add_context_features(context_distributions)
        elif type(context_distributions) in [str, DictConfig]:
            cs = search_space_to_config_space(context_distributions)
            self.add_context_features(cs.get_hyperparameters())
        else:
            raise ValueError(f"Unknown type `{type(context_distributions)}` for `context_distributions`.")
        
        self.context_feature_names = [cf.name for cf in context_distributions]
        self.context_space = context_space

        self.verify_distributions(context_distributions=context_distributions, context_space=context_space)

    def add_context_features(self, context_features: list[ContextFeature]) -> None:
        self.add_hyperparameters(context_features)

    @staticmethod
    def verify_distributions(context_distributions: list[ContextFeature], context_space: ContextSpace) -> bool:
        # TODO verify distributions or context?
        return True

    def sample_contexts(self, n_contexts: int) -> Contexts:
        contexts = self._sample_contexts(size=n_contexts)

        # Convert to dict
        contexts = {i: C for i, C in enumerate(contexts)}
        
        return contexts
        
    def _sample_contexts(
        self, size: int = 1
    ) -> list[Context]:
        contexts = self.sample_configuration(size=size)
        if size == 1:
            contexts = [contexts]
        contexts = [C.get_dictionary() for C in contexts]
        return contexts
  