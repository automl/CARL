from typing import Union, Optional, List
from enum import Enum
from typing import Dict, Any
import warnings

from carl.context.sampling import get_default_context_and_bounds, sample_contexts


class ContextDifficulty(Enum):
    easy = 0.1
    medium = 0.25
    hard = 0.5


class NContexts(Enum):
    small = 100
    medium = 1000
    large = 10000


class AbstractContextSampler(object):
    pass


class ContextSampler(AbstractContextSampler):
    def __init__(
        self,
        env_name: str,
        context_feature_names: Optional[Union[List[str], str]] = None,
        seed: int = 842,
        difficulty: str = "easy",
        n_samples: Union[str, int] = 100,
        sigma_rel: Optional[float] = None,  # overrides difficulty
    ):
        """
        Sample contexts for training or evaluation.

        Parameters
        ----------
        env_name : str
            Name of the environment class, e.g. `CARLPendulumEnv`.
        context_feature_names : Optional[Union[List[str], str]]
            If None, then return default context `n_samples` times.
            If `all`, then return contexts where all features are varied.
            If List[str] (list of context feature names), return context features varied
            as specified in the list, for the rest use default values.
        seed : Optional[int]
            Seed for sampling contexts.
        difficulty : str = "easy"
            Difficulty setting as expressed via different relative standard deviations.
            Can be overriden by argument `sigma_rel`.
        n_samples : Union[str, int] = 100
            Number of contexts to draw. Can be integer or `small`, `medium` and `large`.
        sigma_rel : Optional[float]
            Relative standard deviation, overrides argument `difficulty`.
        """
        super(ContextSampler, self).__init__()
        self.seed = seed
        self.contexts = None
        self.env_name = env_name
        self.C_def, self.C_bounds = get_default_context_and_bounds(
            env_name=self.env_name
        )
        if context_feature_names is None:
            context_feature_names = []
        elif type(context_feature_names) == str and context_feature_names == "all":
            context_feature_names = list(self.C_def.keys())
        self.context_feature_names = context_feature_names

        if sigma_rel is None:
            if difficulty not in ContextDifficulty.__members__:
                raise ValueError(
                    "Please specify a valid difficulty. Valid options are: ",
                    list(ContextDifficulty.__members__.keys()),
                )
            self.sigma_rel = ContextDifficulty[difficulty].value
        else:
            self.sigma_rel = sigma_rel

        if type(n_samples) == str:
            if n_samples not in NContexts.__members__:
                raise ValueError(
                    "Please specify a valid size. Valid options are: ",
                    list(NContexts.__members__.keys()),
                )
            self.n_samples = NContexts[n_samples]
        elif type(n_samples) == int or type(n_samples) == float:
            self.n_samples = int(n_samples)
        else:
            raise ValueError(
                f"`n_samples` must be of type str " f"or int, got {type(n_samples)}."
            )

        self.contexts = None

    def sample_contexts(self) -> Dict[Any, Dict[str, Any]]:
        if self.contexts is not None:
            warnings.warn("Return already sampled contexts.")
            return self.contexts

        context_feature_args = self.context_feature_names
        self.contexts = sample_contexts(
            env_name=self.env_name,
            num_contexts=self.n_samples,
            default_sample_std_percentage=self.sigma_rel,
            context_feature_args=context_feature_args,
            seed=self.seed,
        )
        return self.contexts


if __name__ == "__main__":
    env_name = "CARLPendulumEnv"
    cs = ContextSampler(
        env_name=env_name,
        difficulty="easy",
        n_samples=1,
        context_feature_names=["m", "l", "g"],
        seed=455,
    )
    contexts = cs.sample_contexts()
    print(contexts)
