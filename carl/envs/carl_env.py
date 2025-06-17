from __future__ import annotations

import abc
from typing import Any, SupportsFloat, TypeVar

import inspect

import gymnasium
from gymnasium import Wrapper, spaces
from gymnasium.core import Env

from carl.context.context_space import ContextFeature, ContextSpace
from carl.context.selection import AbstractSelector, RoundRobinSelector
from carl.utils.types import Context, Contexts

ObsType = TypeVar("ObsType")


class CARLEnv(Wrapper, abc.ABC):
    def __init__(
        self,
        env: Env,
        contexts: Contexts | None = None,
        obs_context_features: list[str] | None = None,
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict | None = None,
        **kwargs,
    ):
        """Base CARL wrapper.

        Good to know:

        - The observation always is a dictionary of {"state": ..., "context": ...}. Use
            an observation flattening wrapper if you need a different format.
        - After each env reset, a new context is selected by the context selector.
        - The context set is always filled with defaults if missing.

        Parameters
        ----------
        env : Env
            Environment adhering to gymnasium API.
        contexts : Contexts, optional
            The context set, by default None.
        obs_context_features : list[str], optional
            The context features which should be added to the state, by default None. If None,
            add all available context features.
        obs_context_as_dict : bool, optional
            Whether to pass the context as a vector or a dict in the observations.
            The default is True.
        context_selector : AbstractSelector | type[AbstractSelector] | None
            The context selector selecting a new context after each env reset, by default None.
            If None, use a round robin selector. Can be an object or class. For the latter,
            you can pass kwargs.
        context_selector_kwargs : dict, optional
            Keyword arguments for the context selector if it is passed as a class.


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

        """
        super().__init__(env)

        self.base_observation_space: gymnasium.spaces.Space = env.observation_space
        self.obs_context_as_dict = obs_context_as_dict

        if contexts is None:
            contexts = {
                0: self.get_default_context()
            }  # was self.get_default_context(self) before
        self.contexts = contexts
        self.context: Context | None = None  # Set by `_progress_instance`
        if obs_context_features is None:
            obs_context_features = list(list(self.contexts.values())[0].keys())
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

        self.observation_space: gymnasium.spaces.Dict = self.get_observation_space(
            obs_context_feature_names=self.obs_context_features
        )

    @property
    def contexts(self) -> Contexts:
        return self._contexts

    @property
    def context_id(self):
        return self.context_selector.context_id

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

    @context_id.setter
    def context_id(self, new_id) -> None:
        """Set `context_id` property

        This will switch the context ID of the context selector.
        Realistically you'll want to only use this if your selector does not automaticall progress the instances.

        Parameters
        ----------
        new_id :
            ID to set the context to
        """
        assert new_id in self.context_selector.context_ids, (
            "Unknown ID, this context does not exist in the context set."
        )
        self.context_selector.context_id = new_id
        self.context_selector.context = self.context_selector.contexts[new_id]
        self.context = self.context_selector.context
        self._update_context()

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
                "obs": self.base_observation_space,
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
        last_context_id = self.context_id
        self._progress_instance()
        if self.context_id != last_context_id:
            self._update_context()
        state, info = super().reset(seed=seed, options=options)
        state = self._add_context_to_state(state)
        info["context_id"] = self.context_id
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

        if not self.obs_context_as_dict:
            context = [self.context[k] for k in self.obs_context_features]
        else:
            context = {
                k: v for k, v in self.context.items() if k in self.obs_context_features
            }
        state_context_dict = {
            "obs": state,
            "context": context,
        }
        return state_context_dict

    @abc.abstractmethod
    def _update_context(self) -> None:
        """
        Update the context feature values of the environment.

        `self._progress_instance` must be called at least once to se(lec)t a valid context.

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
        state, reward, terminated, truncated, info = super().step(action)
        state = self._add_context_to_state(state)
        info["context_id"] = self.context_id
        return state, reward, terminated, truncated, info
