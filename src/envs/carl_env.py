import gym
from gym import Wrapper
from gym import spaces
import numpy as np
import os
import json
from typing import Dict, Tuple, Union, List, Optional, Any
from src.context_changer import add_gaussian_noise
from src.context_utils import get_context_bounds
from src.trial_logger import TrialLogger


class CARLEnv(Wrapper):
    """
    Meta-environment formulating the original environments as cMDPs.

    Here, a context feature can be anything defining the behavior of the
    environment. An instance is the environment with a specific context.

    Can change the context after each episode.
    """
    available_scale_methods = ["by_default", "by_mean", "no"]
    available_instance_modes = ["random", "rr", "roundrobin"]

    def __init__(
            self,
            env: gym.Env,
            contexts: Dict[Any, Dict[Any, Any]],
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = False,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None,
            max_episode_length: int = 1e6,
            scale_context_features: str = "no",
            default_context: Optional[Dict] = None,
            state_context_features: Optional[List[str]] = None,
    ):
        """

        Parameters
        ----------
        env: gym.Env
            Environment which context features are made visible / which is turned into a cMDP.
        contexts: Dict[Any, Dict[Any, Any]]
            Dict of contexts/instances. Key are context id, values are contexts as
            Dict[context feature id, context feature value].
        instance_mode: str, default="rr"
            Instance sampling mode. Available modes are 'random' or 'rr'/'roundrobin'.
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
            The default context of the environment. Used for scaling the context features if applicable.
        state_context_features: Optional[List[str]] = None
            If the context is visible to the agent (hide_context=False), the context features are appended to the state.
            state_context_features specifies which of the context features are appended to the state. The default is
            appending all context features.

        Raises
        ------
        ValueError
            If the choice of instance_mode is not available.
        ValueError
            If the choice of scale_context_features is not available.

        """
        super().__init__(env=env)
        # Gather args
        self.contexts = contexts
        if instance_mode not in self.available_instance_modes:
            raise ValueError(f"instance_mode '{instance_mode}' not in '{self.available_instance_modes}'.")
        self.instance_mode = instance_mode
        self.hide_context = hide_context
        self.cutoff = max_episode_length
        self.logger = logger
        self.add_gaussian_noise_to_context = add_gaussian_noise_to_context
        self.gaussian_noise_std_percentage = gaussian_noise_std_percentage
        if state_context_features is not None:
            if state_context_features == "changing_context_features" or state_context_features[0] == "changing_context_features":
                # detect which context feature changes
                context_array = np.array([np.array(list(c.values())) for c in self.contexts.values()])
                which_cf_changes = ~np.all(context_array == context_array[0, :], axis=0)
                context_keys = np.array(list(self.contexts[list(self.contexts.keys())[0]].keys()))
                state_context_features = context_keys[which_cf_changes]
                # print(which_cf_changes, state_context_features)
                if len(state_context_features) == 0:
                    state_context_features = None
                # TODO properly record which are appended to state
                if logger is not None:
                    fname = os.path.join(logger.logdir, "env_info.json")
                    if state_context_features is not None:
                        save_val = list(state_context_features)  # please json
                    else:
                        save_val = state_context_features
                    with open(fname, 'w') as file:
                        data = {
                            "state_context_features": save_val
                        }
                        json.dump(data, file, indent="\t")
        self.state_context_features = state_context_features

        self.step_counter = 0  # type: int # increased in/after step
        self.total_timestep_counter = 0  # type: int
        self.episode_counter = -1  # type: int # increased during reset
        self.whitelist_gaussian_noise = None    # type: Optional[List[str]] # holds names of context features
                                                # where it is allowed to add gaussian noise

        # Set initial context
        self.context_index = 0  # type: int
        context_keys = list(contexts.keys())
        self.context = contexts[context_keys[self.context_index]]

        # Scale context features
        if scale_context_features not in self.available_scale_methods:
            raise ValueError(f"{scale_context_features} not in {self.available_scale_methods}.")
        self.scale_context_features = scale_context_features
        self.default_context = default_context
        self.context_feature_scale_factors = None
        if self.scale_context_features == "by_mean":
            cfs_vals = np.concatenate([np.array(list(v.values()))[:, None] for v in self.contexts.values()], axis=-1)
            self.context_feature_scale_factors = np.mean(cfs_vals, axis=-1)
            self.context_feature_scale_factors[self.context_feature_scale_factors == 0] = 1  # otherwise value / scale_factor = nan
        elif self.scale_context_features == "by_default":
            if self.default_context is None:
                raise ValueError("Please set default_context for scale_context_features='by_default'.")
            self.context_feature_scale_factors = np.array(list(self.default_context.values()))
            self.context_feature_scale_factors[self.context_feature_scale_factors == 0] = 1  # otherwise value / scale_factor = nan

        self.build_observation_space()

    def reset(self, **kwargs):
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

    def build_context_adaptive_state(self, state, context_feature_values=None):
        if not self.hide_context:
            if context_feature_values is None:
                # use current context
                context_values = np.array(list(self.context.values()))
            else:
                # use potentially modified context
                context_values = context_feature_values
            # Append context to state
            if self.state_context_features is not None:
                context_keys = list(self.context.keys())
                context_values = np.array([context_values[context_keys.index(k)] for k in self.state_context_features])
            state = np.concatenate((state, context_values))
        return state

    def step(self, action) -> (Any, Any, bool, Dict):
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
                raise ValueError(f"{self.scale_context_features} not in {self.available_scale_methods}.")

            # Add context features to state
            state = self.build_context_adaptive_state(state, context_feature_values)

        self.total_timestep_counter += 1
        self.step_counter += 1  # TODO do we need to reset the step counter? yes, we do
        if self.step_counter >= self.cutoff:
            done = True
        return state, reward, done, info

    def __getattr__(self, name):
        # TODO: does this work with activated noise? I think we need to update it
        # We need this because our CARLEnv has underscore class methods which would
        # through an error otherwise
        if name in ["_progress_instance", "_update_context", "_log_context"]:
            return getattr(self, name)
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def _progress_instance(self):
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
        if self.instance_mode == "random":
            # TODO pass seed?
            self.context_index = np.random.choice(np.arange(len(self.contexts.keys())))
        elif self.instance_mode in ["rr", "roundrobin"]:
            self.context_index = (self.context_index + 1) % len(self.contexts.keys())
        else:
            raise ValueError(f"Instance mode '{self.instance_mode}' not a valid choice.")
        # TODO add the case that instance_mode is a function or a class
        contexts_keys = list(self.contexts.keys())
        context = self.contexts[contexts_keys[self.context_index]]

        # TODO use class for context changing / value augmentation
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
                    context_augmented[key] = context[key] # TODO(frederik): sample from categorical?
            context = context_augmented
        self.context = context

    def build_observation_space(
            self,
            env_lower_bounds: Optional[Union[List, np.array]] = None,
            env_upper_bounds: Optional[Union[List, np.array]] = None,
            context_bounds: Optional[Dict[str, Tuple[float]]] = None
    ):
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
            If (env.)observation space is not gym.spaces.Box and the context should not be hidden (hide_context = False).

        Returns
        -------
        None

        """
        if not isinstance(self.observation_space, spaces.Box) and not self.hide_context:
            raise ValueError("This environment does not yet support non-hidden contexts. Only supports "
                             "Box observation spaces.")

        obs_shape = self.env.observation_space.low.shape
        if len(obs_shape) == 3 and self.hide_context:
            # do not touch pixel state
            pass
        else:
            if env_lower_bounds is None and env_upper_bounds is None:
                obs_dim = self.env.observation_space.low.shape[0]
                env_lower_bounds = - np.inf * np.ones(obs_dim)
                env_upper_bounds = np.inf * np.ones(obs_dim)

            if self.hide_context:
                self.env.observation_space = spaces.Box(
                    env_lower_bounds, env_upper_bounds, dtype=np.float32,
                )
            else:
                context_keys = list(self.context.keys())
                if context_bounds is None:
                    context_dim = len(list(self.context.keys()))
                    context_lower_bounds = - np.inf * np.ones(context_dim)
                    context_upper_bounds = np.inf * np.ones(context_dim)
                else:
                    context_lower_bounds, context_upper_bounds = get_context_bounds(context_keys, context_bounds)
                if self.state_context_features is not None:
                    ids = np.array([context_keys.index(k) for k in self.state_context_features])
                    context_lower_bounds = context_lower_bounds[ids]
                    context_upper_bounds = context_upper_bounds[ids]
                low = np.concatenate((env_lower_bounds, context_lower_bounds))
                high = np.concatenate((env_upper_bounds, context_upper_bounds))
                self.env.observation_space = spaces.Box(
                    low=low,
                    high=high,
                    dtype=np.float32
                )
            self.observation_space = self.env.observation_space  # make sure it is the same object

    def _update_context(self):
        """
        Update the context feature values of the environment.

        Returns
        -------
        None

        """
        raise NotImplementedError

    def _log_context(self):
        """
        Log context.

        Returns
        -------
        None

        """
        if self.logger:
            self.logger.write_context(self.episode_counter, self.total_timestep_counter, self.context)


