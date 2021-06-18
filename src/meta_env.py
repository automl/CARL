from gym import Wrapper
import numpy as np


class MetaEnv(Wrapper):
    def __init__(self, env, contexts, instance_mode="rr"):
        super().__init__(env=env)
        self.contexts = contexts
        self.instance_mode = instance_mode
        self.context = contexts[0]
        self.context_index = 0

    def reset(self):
        self._progress_instance()
        self._update_context()
        return self.env.reset()

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

    def _update_context(self):
        raise NotImplementedError


