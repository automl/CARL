from __future__ import annotations

from ConfigSpace import ConfigurationSpace
from omegaconf import DictConfig

from carl.context.context_space import ContextFeature, ContextSpace
from carl.context.search_space_encoding import search_space_to_config_space
from carl.utils.types import Context, Contexts


class ContextSampler(ConfigurationSpace):
    def __init__(
        self,
        context_distributions: (
            list[ContextFeature] | dict[str, ContextFeature] | str | DictConfig
        ),
        context_space: ContextSpace,
        seed: int,
        name: str | None = None,
    ):
        self.context_distributions = context_distributions
        super().__init__(name=name, seed=seed)

        if isinstance(context_distributions, list):
            self.add_context_features(context_distributions)
        elif isinstance(context_distributions, dict):
            self.add_context_features(context_distributions.values())
        elif type(context_distributions) in [str, DictConfig]:
            cs = search_space_to_config_space(context_distributions)
            self.add_context_features(cs.get_hyperparameters())
        else:
            raise ValueError(
                f"Unknown type `{type(context_distributions)}` for `context_distributions`."
            )

        self.context_feature_names = [cf.name for cf in self.get_context_features()]
        self.context_space = context_space

    def add_context_features(self, context_features: list[ContextFeature]) -> None:
        self.add_hyperparameters(context_features)

    def get_context_features(self) -> list[ContextFeature]:
        return list(self.values())

    def sample_contexts(self, n_contexts: int) -> Contexts:
        contexts = self._sample_contexts(size=n_contexts)

        # Convert to dict
        contexts = {i: C for i, C in enumerate(contexts)}

        return contexts

    def _sample_contexts(self, size: int = 1) -> list[Context]:
        contexts = self.sample_configuration(size=size)
        default_context = self.context_space.get_default_context()

        if size == 1:
            contexts = [contexts]
        contexts = [dict(default_context | dict(C)) for C in contexts]

        return contexts
