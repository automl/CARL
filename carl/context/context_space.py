from __future__ import annotations

from typing import List, Union

import gymnasium.spaces as spaces
import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    Hyperparameter,
    NumericalHyperparameter,
    UniformFloatHyperparameter,
)
from omegaconf import DictConfig
from search_space_encoding import search_space_to_config_space
from typing_extensions import TypeAlias

ContextFeature: TypeAlias = Hyperparameter


class ContextSpace(ConfigurationSpace):
    """
    Baseclass for contexts of environments implemented in CARL. DO NOT USE DIRECTLY. USE SUBCLASSES!
    """

    def __init__(
        self,
        context_space: Union[List[Hyperparameter], str, DictConfig, ConfigurationSpace]
        | None = None,
        context_features: List[str] | None = None,
        name: str = "context_space",
        seed: int = None,
    ):
        super().__init__(name=name, seed=seed)

        if isinstance(context_space, list):
            self.add_hyperparameters(context_space)
        else:
            cs = search_space_to_config_space(context_space)
            self.add_hyperparameters(cs.get_hyperparameters())

        self._context_features: list[str] = context_features if context_features else []

    def to_gym_space(
        self, context_features: List[str] = None, as_dict: bool = False
    ) -> spaces.Space:
        if context_features is None:
            context_features = [hp.name for hp in self.get_hyperparameters()]

        if as_dict:
            context_space = {}

            for context_feature in context_features:
                hp = self.get_hyperparameter(context_feature)
                if isinstance(hp, NumericalHyperparameter):
                    context_space[hp.name] = spaces.Box(low=hp.lower, high=hp.upper)
                else:
                    raise ValueError(
                        f"Context features must be of type NumericalHyperparameter."
                        f"Got {type(hp)}."
                    )
            return spaces.Dict(context_space)
        else:
            low = np.array(
                [self.get_hyperparameter(cf).lower for cf in context_features]
            )
            high = np.array(
                [self.get_hyperparameter(cf).upper for cf in context_features]
            )

            return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def context_features(self) -> List[str]:
        """
        Context features that are sampled.

        Returns
        -------
        List[str]
            Context features.
        """
        return self._context_features

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


class CartPoleContextSpace(ContextSpace):
    """
    Context space for CARLCartPoleEnv.

    Parameters
    ----------
    context_features : List[str]
        Context features to vary, i.e. sample.

    seed : int = None
        Seed for PRNG.
    """

    def __init__(
        self,
        context_space: Union[List[Hyperparameter], str, DictConfig, ConfigurationSpace]
        | None = None,
        context_features: List[str] | None = None,
        seed: int = None,
    ):
        if context_space is None:
            context_space = CartPoleContextSpace.get_default()
        super().__init__(
            context_space, context_features, name=self.__class__.__name__, seed=seed
        )

    @staticmethod
    def get_default() -> list[ContextFeature]:
        """Get context features with default values

        Returns
        -------
        list[ContextFeature]
            List of context features
        """
        return [
            UniformFloatHyperparameter(
                "gravity", lower=0.1, upper=np.inf, default_value=9.8
            ),
            UniformFloatHyperparameter(
                "masscart", lower=0.1, upper=10, default_value=1.0
            ),
            UniformFloatHyperparameter(
                "masspole", lower=0.01, upper=1, default_value=0.1
            ),
            UniformFloatHyperparameter(
                "pole_length", lower=0.05, upper=5, default_value=0.5
            ),
            UniformFloatHyperparameter(
                "force_magnifier", lower=1, upper=100, default_value=10.0
            ),
            UniformFloatHyperparameter(
                "update_interval", lower=0.002, upper=0.2, default_value=0.02
            ),
            UniformFloatHyperparameter(
                "initial_state_lower", lower=-10, upper=10, default_value=-0.1
            ),
            UniformFloatHyperparameter(
                "initial_state_upper", lower=-10, upper=10, default_value=0.1
            ),  # TODO: We can't have inf. How to handle/annotate when transforming to obs space?
        ]


if __name__ == "__main__":
    print("hello world")
    context_space = CartPoleContextSpace()
    print(context_space)
    context = context_space.sample_configuration()
    print(context)
