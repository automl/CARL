from __future__ import annotations

from ConfigSpace import ConfigurationSpace
from carl.context.context_space import ContextFeature, ContextSpace
from omegaconf import DictConfig
from carl.context.search_space_encoding import search_space_to_config_space
from carl.utils.types import Contexts, Context


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
  