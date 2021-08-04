import gym
from gym import Wrapper
from gym import spaces
import numpy as np
from typing import Dict, Tuple, Union, List, Optional
from src.context_changer import add_gaussian_noise
from src.context_utils import get_context_bounds
from src.trial_logger import TrialLogger


class MetaEnv(Wrapper):
    available_scale_methods = ["by_default", "by_mean", "no"]

    def __init__(
            self,
            env,
            contexts,
            instance_mode="rr",
            hide_context=False,
            add_gaussian_noise_to_context: bool = False,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None,
            max_episode_length: int = 1e6,
            scale_context_features: str = "no",
            default_context: Optional[Dict] = None,
    ):
        super().__init__(env=env)
        self.contexts = contexts
        self.instance_mode = instance_mode
        self.hide_context = hide_context
        self.context_index = 0
        context_keys = list(contexts.keys())
        self.context = contexts[context_keys[self.context_index]]
        self.cutoff = max_episode_length

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

        self.logger = logger
        self.step_counter = 0  # type: int # increased in/after step
        self.episode_counter = -1  # type: int # increased during reset

        self.add_gaussian_noise_to_context = add_gaussian_noise_to_context
        self.gaussian_noise_std_percentage = gaussian_noise_std_percentage
        self.whitelist_gaussian_noise = None  # type: list[str]
        
        if not self.hide_context:
            context_dim = len(list(self.context.values()))
            #TODO: extend this to non-Box obs spaces somehow
            if not isinstance(self.observation_space, gym.spaces.Box):
                raise ValueError("This environment does not yet support non-hidden contexts")
            obs_dim = self.env.observation_space.low.shape[0]
            self.env.observation_space = gym.spaces.Box(low=-np.inf*np.ones(context_dim+obs_dim), high=np.inf*np.ones(context_dim+obs_dim), dtype=np.float32)
            self.observation_space = self.env.observation_space

    def reset(self):
        self.episode_counter += 1
        self._progress_instance()
        self._update_context()
        self._log_context()
        state = self.env.reset()
        if not self.hide_context:
            state = np.concatenate((state, np.array(list(self.context.values()))))
        return state

    def step(self, action):
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
            state = np.concatenate((state, context_feature_values))

        self.step_counter += 1
        if self.step_counter >= self.cutoff:
            done = True
        return state, reward, done, info

    def __getattr__(self, name):
        # TODO: does this work with activated noise? I think we need to update it
        if name in ["_progress_instance", "_update_context"]:
            return getattr(self, name)
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def _progress_instance(self):
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
            context = context_augmented
        self.context = context

    def build_observation_space(
            self,
            env_lower_bounds: Union[List, np.array],
            env_upper_bounds: Union[List, np.array],
            context_bounds: Dict[str, Tuple[float]]
    ):
        if self.hide_context:
            self.env.observation_space = spaces.Box(
                env_lower_bounds, env_upper_bounds, dtype=np.float32,
            )
        else:
            context_keys = list(self.context.keys())
            context_lower_bounds, context_upper_bounds = get_context_bounds(context_keys, context_bounds)
            low = np.concatenate((env_lower_bounds, context_lower_bounds))
            high = np.concatenate((env_upper_bounds, context_upper_bounds))
            self.env.observation_space = spaces.Box(
                low=low,
                high=high
            )
        self.observation_space = self.env.observation_space  # make sure it is the same object

    def _update_context(self):
        raise NotImplementedError

    def _log_context(self):
        if self.logger:
            self.logger.write_context(self.episode_counter, self.step_counter, self.context)


