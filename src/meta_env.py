import gym
from gym import Wrapper
from gym import spaces
import numpy as np
from src.context_changer import add_gaussian_noise
from src.context_utils import get_context_bounds


class MetaEnv(Wrapper):
    def __init__(
            self,
            env,
            contexts,
            instance_mode="rr",
            hide_context=False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01
    ):
        super().__init__(env=env)
        self.contexts = contexts
        self.instance_mode = instance_mode
        self.hide_context = hide_context
        self.context = contexts[0]
        self.context_index = 0

        self.add_gaussian_noise_to_context = add_gaussian_noise_to_context
        self.gaussian_noise_std_percentage = gaussian_noise_std_percentage
        self.whitelist_gaussian_noise = None  # type: list[str]

        if not self.hide_context:
            context_dim = len(list(self.context.values()))
            #TODO: extend this to non-Box obs spaces somehow
            # TODO check if observation space of env is a box or not
            obs_dim = self.env.observation_space.low.shape[0]
            self.env.observation_space = gym.spaces.Box(low=-np.inf*np.ones(context_dim+obs_dim), high=np.inf*np.ones(context_dim+obs_dim))

    def reset(self):
        self._progress_instance()
        self._update_context()
        state = self.env.reset()
        if not self.hide_context:
            state = np.concatenate((state, np.array(list(self.context.values()))))  # TODO test if this has the correct shape
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if not self.hide_context:
            state = np.concatenate((state, np.array(list(self.context.values()))))  # TODO test if this has the correct shape
        return state, reward, done, info

    def __getattr__(self, name):
        if name in ["_progress_instance", "_update_context"]:
            return getattr(self, name)
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def _progress_instance(self):
        if self.instance_mode == "random":
            self.context_index = np.random_choice(np.arange(len(self.contexts.keys())))
        else:
            self.context_index = (self.context_index + 1) % len(self.contexts.keys())
        self.context = self.contexts[self.context_index]

        # TODO use class for context changing / value augmentation
        if self.add_gaussian_noise_to_context and self.whitelist_gaussian_noise:
            for key, value in self.context.items():
                if key in self.whitelist_gaussian_noise:
                    self.context[key] = add_gaussian_noise(
                        default_value=value,
                        percentage_std=self.gaussian_noise_std_percentage,
                        random_generator=self.np_random
                    )

    def build_observation_space(self, env_lower_bounds, env_upper_bounds, context_bounds):
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


