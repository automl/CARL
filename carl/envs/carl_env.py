from typing import Any, Dict, List, Optional, Tuple, Union

import json
import os
import inspect

import gym
import numpy as np
from gym import Wrapper, spaces

from carl.context.augmentation import add_gaussian_noise
from carl.context.utils import get_context_bounds
from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector, RoundRobinSelector

import importlib

brax_spec = importlib.util.find_spec("brax")
if brax_spec is not None:
    import jaxlib
    import jax.numpy as jnp

import torch as th


class CARLEnv(Wrapper):
    """
    Meta-environment formulating the original environments as cMDPs.

    Here, a context feature can be anything defining the behavior of the
    environment. An instance is the environment with a specific context.

    Can change the context after each episode.

    If not all keys are present in the provided context(s) the contexts will be filled
    with the default context values in the init of the class.

    Parameters
    ----------
    env: gym.Env
        Environment which context features are made visible / which is turned into a cMDP.
    contexts: Dict[Any, Dict[Any, Any]]
        Dict of contexts/instances. Key are context id, values are contexts as
        Dict[context feature id, context feature value].
    hide_context: bool = False
        If False, the context will be appended to the original environment's state.
    add_gaussian_noise_to_context: bool = False
        Wether to add Gaussian noise to the context with the relative standard deviation
        'gaussian_noise_std_percentage'.
    gaussian_noise_std_percentage: float = 0.01
        The relative standard deviation for the Gaussian noise. The actual standard deviation
        is calculated by 'gaussian_noise_std_percentage' * context feature value.
    logger: TrialLogger, optional
        Optional TrialLogger which takes care of setting up logging directories and handles
        custom logging.
    max_episode_length: int = 1e6
        Maximum length of episode in (time)steps. Cutoff.
    scale_context_features: str = "no"
        Wether to scale context features. Available modes are 'no', 'by_mean' and 'by_default'.
        'by_mean' scales the context features by their mean over all passed instances and
        'by_default' scales the context features by their default values ('default_context').
    default_context: Dict
        The default context of the environment. Used for scaling the context features if applicable. Used for filling
        incomplete contexts.
    state_context_features: Optional[List[str]] = None
        If the context is visible to the agent (hide_context=False), the context features are appended to the state.
        state_context_features specifies which of the context features are appended to the state. The default is
        appending all context features.
    context_mask: Optional[List[str]]
        Name of context features to be ignored when appending context features to the state.
    context_selector: Optional[Union[AbstractSelector, type(AbstractSelector)]]
        Context selector (object of) class, e.g., can be RoundRobinSelector (default) or RandomSelector.
        Should subclass AbstractSelector.
    context_selector_kwargs: Optional[Dict]
        Optional kwargs for context selector class.

    Raises
    ------
    ValueError
        If the choice of instance_mode is not available.
    ValueError
        If the choice of scale_context_features is not available.

    """

    available_scale_methods = ["by_default", "by_mean", "no"]
    available_instance_modes = ["random", "rr", "roundrobin"]

    def __init__(
        self,
        env: gym.Env,
        n_envs: int = 1,
        contexts: Dict[Any, Dict[Any, Any]] = {},
        hide_context: bool = True,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.01,
        logger: Optional[TrialLogger] = None,
        max_episode_length: int = int(1e6),
        scale_context_features: str = "no",
        default_context: Optional[Dict] = None,
        state_context_features: Optional[List[str]] = None,
        context_mask: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[
            Union[AbstractSelector, type(AbstractSelector)]
        ] = None,
        context_selector_kwargs: Optional[Dict] = None,
    ):
        super().__init__(env=env)
        # Gather args
        self._context: Optional[Dict] = None  # init for property
        self._contexts: Optional[Dict[Any, Dict[Any, Any]]] = None  # init for property
        self.default_context = default_context
        self.contexts = contexts
        self.context_mask = context_mask
        self.hide_context = hide_context
        self.dict_observation_space = dict_observation_space
        self.cutoff = max_episode_length
        self.logger = logger
        self.add_gaussian_noise_to_context = add_gaussian_noise_to_context
        self.gaussian_noise_std_percentage = gaussian_noise_std_percentage
        if context_selector is None:
            self.context_selector = RoundRobinSelector(contexts=contexts)
        elif isinstance(context_selector, AbstractSelector):
            self.context_selector = context_selector
        elif inspect.isclass(context_selector) and issubclass(
            context_selector, AbstractSelector
        ):
            if context_selector_kwargs is None:
                context_selector_kwargs = {}
            _context_selector_kwargs = {"contexts": contexts}
            context_selector_kwargs.update(_context_selector_kwargs)
            self.context_selector = context_selector(**context_selector_kwargs)
        else:
            raise ValueError(
                f"Context selector must be None or an AbstractSelector class or instance. "
                f"Got type {type(context_selector)}."
            )
        if state_context_features is not None:
            if (
                state_context_features == "changing_context_features"
                or state_context_features[0] == "changing_context_features"
            ):
                # if we have only one context the context features do not change during training
                if len(self.contexts) > 1:
                    # detect which context feature changes
                    context_array = np.array(
                        [np.array(list(c.values())) for c in self.contexts.values()]
                    )
                    which_cf_changes = ~np.all(
                        context_array == context_array[0, :], axis=0
                    )
                    context_keys = np.array(
                        list(self.contexts[list(self.contexts.keys())[0]].keys())
                    )
                    state_context_features = context_keys[which_cf_changes]
                    # TODO properly record which are appended to state
                    if logger is not None:
                        fname = os.path.join(logger.logdir, "env_info.json")
                        if state_context_features is not None:
                            save_val = list(state_context_features)  # please json
                        else:
                            save_val = state_context_features
                        with open(fname, "w") as file:
                            data = {"state_context_features": save_val}
                            json.dump(data, file, indent="\t")
                else:
                    state_context_features = []
        else:
            state_context_features = list(
                self.contexts[list(self.contexts.keys())[0]].keys()
            )
        self.state_context_features: List[str] = state_context_features
        # state_context_features contains the names of the context features that should be appended to the state
        # However, if context_mask is set, we want to update staet_context_feature_names so that the context features
        # in context_mask are not appended to the state anymore.
        if self.context_mask:
            self.state_context_features = [
                s for s in self.state_context_features if s not in self.context_mask
            ]

        self.step_counter = 0  # type: int # increased in/after step
        self.total_timestep_counter = 0  # type: int
        self.episode_counter = -1  # type: int # increased during reset
        self.whitelist_gaussian_noise = (
            None
        )  # type: Optional[List[str]] # holds names of context features
        # where it is allowed to add gaussian noise

        # Set initial context
        self.context_index = 0  # type: int
        context_keys = list(self.contexts.keys())
        self.context = self.contexts[context_keys[self.context_index]]

        # Scale context features
        if scale_context_features not in self.available_scale_methods:
            raise ValueError(
                f"{scale_context_features} not in {self.available_scale_methods}."
            )
        self.scale_context_features = scale_context_features
        self.context_feature_scale_factors = None
        if self.scale_context_features == "by_mean":
            cfs_vals = np.concatenate(
                [np.array(list(v.values()))[:, None] for v in self.contexts.values()],
                axis=-1,
            )
            self.context_feature_scale_factors = np.mean(cfs_vals, axis=-1)
            self.context_feature_scale_factors[
                self.context_feature_scale_factors == 0
            ] = 1  # otherwise value / scale_factor = nan
        elif self.scale_context_features == "by_default":
            if self.default_context is None:
                raise ValueError(
                    "Please set default_context for scale_context_features='by_default'."
                )
            self.context_feature_scale_factors = np.array(
                list(self.default_context.values())
            )
            self.context_feature_scale_factors[
                self.context_feature_scale_factors == 0
            ] = 1  # otherwise value / scale_factor = nan

        self.vectorized = n_envs > 1
        self.build_observation_space()

    @property
    def context(self) -> Dict:
        return self._context

    @context.setter
    def context(self, context: Dict):
        self._context = self.fill_context_with_default(context=context)

    @property
    def contexts(self) -> Dict[Any, Dict[Any, Any]]:
        return self._contexts

    @contexts.setter
    def contexts(self, contexts: Dict[Any, Dict[Any, Any]]):
        self._contexts = {
            k: self.fill_context_with_default(context=v) for k, v in contexts.items()
        }

    def reset(self, **kwargs: Dict) -> Any:
        """
        Reset environment.

        Parameters
        ----------
        kwargs: Dict
            Any keyword arguments passed to env.reset().

        Returns
        -------
        state
            State of environment after reset.

        """
        self.episode_counter += 1
        self.step_counter = 0
        self._progress_instance()
        self._update_context()
        self._log_context()
        state = self.env.reset(**kwargs)
        state = self.build_context_adaptive_state(state)
        return state

    def build_context_adaptive_state(
        self, state: List[float], context_feature_values: Optional[List[float]] = None
    ) -> List[float]:
        """
        Add context to state if visible

        Parameters
        ----------
        state                   : state of environment
        context_feature_values  : list of context feature values
        Returns
        -------
        State with context or original state
        """
        tnp = np
        if brax_spec is not None:
            if type(state) == jaxlib.xla_extension.DeviceArray:
                tnp = jnp
        if not self.hide_context:
            if context_feature_values is None:
                # use current context
                context_values = tnp.array(list(self.context.values()))
            else:
                # use potentially modified context
                context_values = context_feature_values
            # Append context to state
            if self.state_context_features is not None:
                # if self.state_context_features is an empty list, the context values will also be empty and we
                # get the correct state
                context_keys = list(self.context.keys())
                context_values = tnp.array(
                    [
                        context_values[context_keys.index(k)]
                        for k in self.state_context_features
                    ]
                )

            # Pass the context features to the encoder to extract decomposed representation
            if self.context_encoder:
                # Encode context
                context_values = self.encode_contexts(context_values)

            if self.dict_observation_space:
                state = dict(state=state, context=context_values)
            elif self.vectorized:
                state = tnp.array([np.concatenate((s, context_values)) for s in state])
            else:
                state = tnp.concatenate((state, context_values))
        return state

    def step(self, action: Any) -> Tuple[Any, Any, bool, Dict]:
        """
        Step the environment.

        1. Step
        2. Add (potentially scaled) context features to state if hide_context = False.

        Emits done if the environment has taken more steps than cutoff (max_episode_length).

        Parameters
        ----------
        action:
            Action to pass to env.step.

        Returns
        -------
        state, reward, done, info: Any, Any, bool, Dict
            Standard signature.

        """
        # Step the environment
        state, reward, done, info = self.env.step(action)

        if not self.hide_context:
            # Scale context features
            context_feature_values = np.array(list(self.context.values()))
            if self.scale_context_features == "by_default":
                context_feature_values /= self.context_feature_scale_factors
            elif self.scale_context_features == "by_mean":
                context_feature_values /= self.context_feature_scale_factors
            elif self.scale_context_features == "no":
                pass
            else:
                raise ValueError(
                    f"{self.scale_context_features} not in {self.available_scale_methods}."
                )

            # Add context features to state
            state = self.build_context_adaptive_state(state, context_feature_values)

        self.total_timestep_counter += 1
        self.step_counter += 1
        if self.step_counter >= self.cutoff:
            done = True
        return state, reward, done, info

    def __getattr__(self, name: str) -> Any:
        # TODO: does this work with activated noise? I think we need to update it
        # We need this because our CARLEnv has underscore class methods which would
        # throw an error otherwise
        if name in ["_progress_instance", "_update_context", "_log_context"]:
            return getattr(self, name)
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    def fill_context_with_default(self, context: Dict) -> Dict:
        """
        Fill the context with the default values if entries are missing

        Parameters
        ----------
        context

        Returns
        -------
        context

        """
        if self.default_context:
            context_def = self.default_context.copy()
            context_def.update(context)
            context = context_def
        return context

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
        context = self.context_selector.select()

        if self.add_gaussian_noise_to_context and self.whitelist_gaussian_noise:
            context_augmented = {}
            for key, value in context.items():
                if key in self.whitelist_gaussian_noise:
                    context_augmented[key] = add_gaussian_noise(
                        default_value=value,
                        percentage_std=self.gaussian_noise_std_percentage,
                        random_generator=None,  # self.np_random TODO discuss this
                    )
                else:
                    context_augmented[key] = context[key]
            context = context_augmented
        self.context = context

    def build_observation_space(
        self,
        env_lower_bounds: Optional[Union[List, np.array]] = None,
        env_upper_bounds: Optional[Union[List, np.array]] = None,
        context_bounds: Optional[Dict[str, Tuple[float]]] = None,
    ) -> None:
        """
        Build observation space of environment.

        If the hide_context = False, add correct bounds for the context features to the
        observation space.

        Parameters
        ----------
        env_lower_bounds: Optional[Union[List, np.array]], default=None
            Lower bounds for environment observation space. If env_lower_bounds and env_upper_bounds
            both are None, (re-)create bounds (low=-inf, high=inf) with correct dimension.
        env_upper_bounds: Optional[Union[List, np.array]], default=None
            Upper bounds for environment observation space.
        context_bounds: Optional[Dict[str, Tuple[float]]], default=None
            Lower and upper bounds for context features.
            The bounds are provided as a Dict containing the context feature names/ids as keys and the
            bounds per feature as a tuple (low, high).
            If None and the context should not be hidden,
            creates default bounds with (low=-inf, high=inf) with correct dimension.

        Raises
        ------
        ValueError:
            If (env.)observation space is not gym.spaces.Box and the context should not be hidden
            (hide_context = False).

        Returns
        -------
        None

        """
        self.observation_space: gym.spaces.Space
        if (
            not self.dict_observation_space
            and not isinstance(self.observation_space, spaces.Box)
            and not self.hide_context
        ):
            raise ValueError(
                "This environment does not yet support non-hidden contexts. Only supports "
                "Box observation spaces."
            )

        obs_space = (
            self.env.observation_space.spaces["state"].low
            if isinstance(self.env.observation_space, spaces.Dict)
            else self.env.observation_space.low
        )
        obs_shape = obs_space.shape
        if len(obs_shape) == 3 and self.hide_context:
            # do not touch pixel state
            pass
        else:
            if self.context is None:
                context_keys = list(list(self.contexts.values())[0].keys())
            else:
                context_keys = list(self.context.keys())

            if env_lower_bounds is None and env_upper_bounds is None:
                obs_dim = obs_shape[0]
                env_lower_bounds = -np.inf * np.ones(obs_dim)
                env_upper_bounds = np.inf * np.ones(obs_dim)

            if self.hide_context or (
                self.state_context_features is not None
                and len(self.state_context_features) == 0
            ):
                # TODO get rid of the spaghetti please
                if self.dict_observation_space:
                    context_lower_bounds, context_upper_bounds = get_context_bounds(
                        context_keys, context_bounds
                    )
                    self.env.observation_space = spaces.Dict(
                        {
                            "state": spaces.Box(
                                low=env_lower_bounds,
                                high=env_upper_bounds,
                                dtype=np.float32,
                            ),
                            "context": spaces.Box(
                                low=context_lower_bounds,
                                high=context_upper_bounds,
                                dtype=np.float32,
                            ),
                        }
                    )

                else:
                    self.env.observation_space = spaces.Box(
                        env_lower_bounds,
                        env_upper_bounds,
                        dtype=np.float32,
                    )
            else:
                context_keys = list(self.context.keys())
                if context_bounds is None:
                    context_dim = len(list(self.context.keys()))
                    context_lower_bounds = -np.inf * np.ones(context_dim)
                    context_upper_bounds = np.inf * np.ones(context_dim)
                else:
                    context_lower_bounds, context_upper_bounds = get_context_bounds(
                        context_keys, context_bounds
                    )
                if self.state_context_features is not None:
                    ids = np.array(
                        [context_keys.index(k) for k in self.state_context_features]
                    )
                    context_lower_bounds = context_lower_bounds[ids]
                    context_upper_bounds = context_upper_bounds[ids]

                if self.dict_observation_space:
                    self.env.observation_space = spaces.Dict(
                        {
                            "state": spaces.Box(
                                low=env_lower_bounds,
                                high=env_upper_bounds,
                                dtype=np.float32,
                            ),
                            "context": spaces.Box(
                                low=context_lower_bounds,
                                high=context_upper_bounds,
                                dtype=np.float32,
                            ),
                        }
                    )
                else:
                    low = np.concatenate((env_lower_bounds, context_lower_bounds))
                    high = np.concatenate((env_upper_bounds, context_upper_bounds))
                    self.env.observation_space = spaces.Box(
                        low=low, high=high, dtype=np.float32
                    )
            self.observation_space = (
                self.env.observation_space
            )  # make sure it is the same object

    def encode_contexts(
        self,
        context_vals: Optional[Union[List, np.array]],
    ) -> np.array:
        """
        Pass the context values through an encoder
        Parameters
        ----------
        context_vals: np.array
            Context values to encode.

        Returns
        -------
        np.array
            Encoded context values.
        """

        # Make contexts into a tensor
        context_tensor = th.tensor(context_vals)
        context_reps = self.context_encoder.get_representation(context_tensor)

        # Post-processing to get an array from tensors, irrespective
        # of the latent size
        context_reps = np.array([t.detach().numpy() for t in context_reps])

        return context_reps

    def _update_context(self) -> None:
        """
        Update the context feature values of the environment.

        Returns
        -------
        None

        """
        raise NotImplementedError

    def _log_context(self) -> None:
        """
        Log context.

        Returns
        -------
        None

        """
        if self.logger:
            self.logger.write_context(
                self.episode_counter, self.total_timestep_counter, self.context
            )
