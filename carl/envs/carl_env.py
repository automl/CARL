from __future__ import annotations

import abc
from typing import Any, SupportsFloat

import inspect

import gymnasium
import jax
import numpy as np
from gymnasium import Wrapper, spaces
from gymnasium.core import Env
from jax import numpy as jp

from carl.context.context_space import ContextFeature, ContextSpace
from carl.context.sampler import ContextSampler
from carl.context.selection import AbstractSelector, RoundRobinSelector
from carl.utils.types import Context, Contexts


class CARLEnv(Wrapper, abc.ABC):
    def __init__(
        self,
        env: Env,
        contexts: Contexts | None = None,
        obs_context_features: list[str]
        | None = None,  # list the context features which should be added to the state # TODO rename to obs_context_features?
        obs_context_as_dict: bool = True,  # TODO discuss default
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        **kwargs,
    ):
        """Base CARL wrapper.

        Good to know:

        - The observation always is a dictionary of {"state": ..., "context": ...}. Use
            an observation flattening wrapper if you need a different format.
        - After each env reset, a new context is selected by the context selector.
        - The context set is always filled with defaults if missing.

        Attributes
        ----------
        base_observation_space: gymnasium.spaces.Space
            The observation space from the wrapped environment.
        obs_context_as_dict: bool, optional
            Whether to pass the context as a vector or a dict in the observations.
            The default is True.
        observation_space: gymnasium.spaces.Dict
            The observation space of the CARL environment which is a dictionary of
            "state" and "context".
        contexts: Contexts
            The context set.
        context_selector: ContextSelector.
            The context selector selecting a new context after each env reset.


        Parameters
        ----------
        env : Env
            Environment adhering to the gymnasium API.
        contexts : Contexts | None, optional
            Context set, by default None. If it is None, we build the
            context set with the default context.
        obs_context_features : list[str] | None, optional
            Context features which should be included in the observation, by default None.
            If they are None, add all context features.
        obs_context_as_dict: bool, optional
            Whether to pass the context as a vector or a dict in the observations.
            The default is True.
        context_selector: AbstractSelector | type[AbstractSelector] | None, optional
            The context selector (class), after each reset selects a new context to use.
             If None, use a round robin selector.
        context_selector_kwargs : dict, optional
            Optional keyword arguments for the context selector, by default None.
            Only used when `context_selector` is not None.

        Raises
        ------
        ValueError
            If the type of `context_selector` is invalid.
        """
        super().__init__(env)

        self.base_observation_space: gymnasium.spaces.Space = env.observation_space
        self.obs_context_as_dict = obs_context_as_dict

        if contexts is None:
            contexts = {0: self.get_default_context()}   # was self.get_default_context(self) before
        self.contexts = contexts
        self.obs_context_features = obs_context_features

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
        self.observation_space: gymnasium.spaces.Dict = self.get_observation_space(
            obs_context_feature_names=self.obs_context_features
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
        """Get the observation space for the context.

        Parameters
        ----------
        obs_context_feature_names : list[str] | None, optional
            Name of the context features to be included in the observation, by default None.
            If it is None, we add all context features.

        Returns
        -------
        gymnasium.spaces.Dict
            Gymnasium observation space which contains the observation space of the
            underlying environment ("state") and for the context ("context").
        """
        context_space = self.get_context_space()
        obs_space_context = context_space.to_gymnasium_space(
            context_feature_names=obs_context_feature_names,
            as_dict=self.obs_context_as_dict,
        )
        obs_space = spaces.Dict(
            {
                # TODO should we rename "state" to "obs"?
                "state": self.base_observation_space,
                "context": obs_space_context,
            }
        )
        return obs_space

    @staticmethod
    @abc.abstractmethod
    def get_context_features() -> dict[str, ContextFeature]:
        """Get the context features

        Defined per environment.

        Returns
        -------
        dict[str, ContextFeature]
            Context feature definitions
        """
        ...

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
        """Reset the environment.

        First, we progress the instance, i.e. select a new context with the context
        selector. Then we update the context in the wrapped environment.
        Finally, we reset the underlying environment and add context information
        to the observation.

        Parameters
        ----------
        seed : int | None, optional
            Seed, by default None
        options : dict[str, Any] | None, optional
            Options, by default None

        Returns
        -------
        tuple[Any, dict[str, Any]]
            Observation, info.
        """
        self._progress_instance()
        self._update_context()
        state, info = super().reset(seed=seed, options=options)
        state = self._add_context_to_state(state)
        # TODO: Add context id or sth similar to info?
        return state, info

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
            context = [self.context[k] for k in self.obs_context_features]
        else:
            context = {
                k: v for k, v in self.context.items() if k in self.obs_context_features
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
        """Step the environment.

        The context is added to the observation returned by the
        wrapped environment.

        Parameters
        ----------
        action : Any
            Action

        Returns
        -------
        tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]
            Observation, rewar, terminated, truncated, info.
        """
        state = super().step(action)
        state = self._add_context_to_state(state)
        return state
