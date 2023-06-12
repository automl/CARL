from __future__ import annotations

import inspect
from typing import Any, SupportsFloat

from gymnasium import Wrapper, spaces
import gymnasium
from gymnasium.core import Env

import jax
from jax import numpy as jp
import numpy as np
import abc

from carl.utils.types import Contexts, Context
from carl.context.selection import AbstractSelector, RoundRobinSelector
from carl.context.context_space import (
    ContextFeature,
    ContextSpace,
    UniformFloatContextFeature,
    ContextSampler,
    NormalFloatContextFeature,
)


class CARLEnv(Wrapper, abc.ABC):
    def __init__(
        self,
        env: Env,
        contexts: Contexts | None = None,
        state_context_features: list[str]
        | None = None,  # list the context features which should be added to the state # TODO rename to obs_context_features?
        obs_context_as_dict: bool = True,  # TODO discuss default
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(env)

        self.state_observation_space: gymnasium.spaces.Space = env.observation_space
        self.obs_context_as_dict = obs_context_as_dict

        if contexts is None:
            contexts = {0: self.get_default_context(self)}
        self.contexts = contexts
        self.state_context_features = state_context_features

        # Context Selector
        self.context_selector: type[AbstractSelector]
        if context_selector is None:
            self.context_selector = RoundRobinSelector(contexts=contexts)  # type: ignore [assignment]
        elif isinstance(context_selector, AbstractSelector):
            self.context_selector = context_selector  # type: ignore [assignment]
        elif inspect.isclass(context_selector) and issubclass(
            context_selector, AbstractSelector
        ):
            if context_selector_kwargs is None:
                context_selector_kwargs = {}
            _context_selector_kwargs = {"contexts": contexts}
            context_selector_kwargs.update(_context_selector_kwargs)
            self.context_selector = context_selector(**context_selector_kwargs)  # type: ignore [assignment]
        else:
            raise ValueError(
                f"Context selector must be None or an AbstractSelector class or instance. "
                f"Got type {type(context_selector)}."
            )
        self._progress_instance()
        self._update_context()
        self.observation_space = self.get_observation_space(
            obs_context_feature_names=self.state_context_features
        )

    @property
    def contexts(self) -> Contexts:
        return self._contexts
    
    @contexts.setter
    def contexts(self, contexts: Contexts) -> None:
        """Set `contexts` property

        For each context maybe fill with default context values.
        This is only necessary whenever we update the contexts,
        so here is the right place.

        Parameters
        ----------
        contexts : Contexts
            Contexts to set
        """
        context_space = self.get_context_space()
        contexts = {k: context_space.insert_defaults(v) for k, v in contexts.items()}
        self._contexts = contexts

    def get_observation_space(
        self, obs_context_feature_names: list[str] | None = None
    ) -> gymnasium.spaces.Dict:
        context_space = self.get_context_space()
        obs_space_context = context_space.to_gymnasium_space(
            context_feature_names=obs_context_feature_names,
            as_dict=self.obs_context_as_dict,
        )
        obs_space = spaces.Dict(
            {
                "state": self.state_observation_space,
                "context": obs_space_context,
            }
        )
        return obs_space

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        """Get the context features

        Defined per environment.

        Returns
        -------
        dict[str, ContextFeature]
            Context feature definitions
        """
        raise NotImplementedError

    @classmethod
    def get_context_space(cls) -> ContextSpace:
        """Get context space

        Returns
        -------
        ContextSpace
            Context space with utility methods holding
            information about defaults, types, bounds, etc.
        """
        return ContextSpace(cls.get_context_features())

    @classmethod
    def get_default_context(cls) -> Context:
        """Get the default context

        Returns
        -------
        Context
            Default context.
        """
        default_context = cls.get_context_space().get_default_context()
        return default_context

    def _progress_instance(self) -> None:
        """
        Progress instance.

        In this case instance is a specific context.
        1. Select instance with the instance_mode. If the instance_mode is random, randomly select
        the next instance from the set of contexts. If instance_mode is rr or roundrobin, select
        the next instance.
        2. If Gaussian noise should be added to whitelisted context features, do so.

        Returns
        -------
        None

        """
        context = self.context_selector.select()  # type: ignore [call-arg]
        self.context = context

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._progress_instance()
        self._update_context()
        state = super().reset(seed=seed, options=options)
        state = self._add_context_to_state(state)
        return state

    def _add_context_to_state(self, state: Any) -> dict[str, Any]:
        """Add context observation to the state

        The state is the observation from the underlying environment
        and we add the context information to it. We return a dictionary
        of the state and context, and the context is maybe represented
        as a dictionary itself (controlled via `self.obs_context_as_dict`).

        Parameters
        ----------
        state : Any
            State from the environment

        Returns
        -------
        dict[str, Any]
            State context observation dict
        """
        context = self.context
        if not self.obs_context_as_dict:
            context = [self.context[k] for k in self.state_context_features]
        else:
            context = {
                k: v
                for k, v in self.context.items()
                if k in self.state_context_features
            }
        state_context_dict = {
            "state": state,
            "context": context,
        }
        return state_context_dict

    @abc.abstractmethod
    def _update_context(self) -> None:
        """
        Update the context feature values of the environment.

        Returns
        -------
        None

        """
        ...

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        state = super().step(action)
        state = self._add_context_to_state(state)
        return state


class CARLClassicControlEnv(CARLEnv):
    env_name: str

    def __init__(
        self,
        env: Env | None = None,
        contexts: Contexts | None = None,
        state_context_features: list[str]
        | None = None,  # list the context features which should be added to the state # TODO rename to obs_context_features?
        # context_mask: list[str] | None = None,
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        **kwargs,
    ):
        if env is None:
            env = gymnasium.make(id=self.env_name)
        super().__init__(
            env=env,
            contexts=contexts,
            state_context_features=state_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )


class CARLCartPole(CARLClassicControlEnv):
    env_name: str = "CartPole-v1"

    # TODO do we want to modify the initial state distribution bounds like before?

    def _update_context(self) -> None:
        for k, v in self.context.items():
            setattr(self.env, k, v)

    def get_context_features() -> list[ContextFeature]:
        return {
            "gravity": 
            UniformFloatContextFeature(
                "gravity", lower=0.1, upper=np.inf, default_value=9.8
            ),
            "masscart": 
            UniformFloatContextFeature(
                "masscart", lower=0.1, upper=10, default_value=1.0
            ),
            "masspole":
            UniformFloatContextFeature(
                "masspole", lower=0.01, upper=1, default_value=0.1
            ),
            "length":
            UniformFloatContextFeature(
                "length", lower=0.05, upper=5, default_value=0.5
            ),
            "force_mag":
            UniformFloatContextFeature(
                "force_mag", lower=1, upper=100, default_value=10.0
            ),
            "tau":
            UniformFloatContextFeature(
                "tau", lower=0.002, upper=0.2, default_value=0.02
            ),
        }


if __name__ == "__main__":
    from rich import print as printr

    seed = 0
    # Sampling demo
    context_distributions = [NormalFloatContextFeature("gravity", mu=9.8, sigma=1)]
    context_sampler = ContextSampler(
        context_distributions=context_distributions,
        context_space=CARLCartPole.get_context_space(),
        seed=seed,
    )
    contexts = context_sampler.sample_contexts(n_contexts=5)

    # Env demo

    printr(CARLCartPole.get_context_space())

    printr(contexts)


    state_context_features = list(CARLCartPole.get_default_context().keys())[:2]

    env = CARLCartPole(contexts=contexts, state_context_features=state_context_features)

    state = env.reset()

    printr(state)
