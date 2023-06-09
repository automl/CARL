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

ContextFeature: TypeAlias = Hyperparameter
NumericalContextFeature: TypeAlias = NumericalHyperparameter
NormalFloatContextFeature: TypeAlias = NormalFloatHyperparameter
UniformFloatContextFeature: TypeAlias = UniformFloatHyperparameter


class ContextSpace(object):
    def __init__(
            self,
            context_features: list[ContextFeature]
        ) -> None:
        self.context_space = {
            context_feature.name: context_feature for context_feature in context_features
        }

    @property
    def context_feature_names(self) -> List[str]:
        """
        Context feature names.

        Returns
        -------
        List[str]
            Context features names.
        """
        return list(self.context_space.keys())
    
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
        context_distributions: Union[
            List[Hyperparameter], str, DictConfig, ConfigurationSpace
        ],
        seed: int,
        name: str | None = None,
    ):
        self.context_distributions = context_distributions
        # TODO: pass seed
        super().__init__(name=name, seed=seed)

        if isinstance(context_distributions, list):
            self.add_context_features(context_distributions)
        else:
            cs = search_space_to_config_space(context_distributions)
            self.add_context_features(cs.get_hyperparameters())

        self._context_features: list[str] = context_features if context_features else []

    def add_context_features(self, context_features: list[ContextFeature]) -> None:
        self.add_hyperparameters(context_features)

    def add_defaults(context_space: ContextSpace):
        """Add defaults from env context space

        Parameters
        ----------
        context_space : ContextSpace
            _description_
        """
        ...

    def sample_contexts(self, n_contexts: int) -> dict[int, dict[str, Any]]:
        contexts = self.sample_configuration(size=n_contexts)
        contexts = {i: C for i, C in enumerate(contexts)}
        return contexts
    
    def sample_configuration(
        self, size: int = 1
    ) -> Union[Configuration, List[Configuration]]:
        """
        Samples values for all active context features. For all other values default is used.

        Parameters
        ----------
        size : int = 1
            Number of contexts to sample.

        Returns
        -------
        Union[Configuration, List[Configuration]]
            Sampled contexts.
        """

        def insert_defaults(cfg: Configuration) -> Configuration:
            values = cfg.get_dictionary()
            for feature in values.keys():
                if feature not in self._context_features:
                    values[feature] = self.get_hyperparameter(feature).default_value

            return Configuration(self, values=values)

        configurations = super().sample_configuration(size=size)

        if size == 1:
            configurations = insert_defaults(configurations)
        else:
            for i, configuration in enumerate(configurations):
                configurations[i] = insert_defaults(configuration)

        return configurations
