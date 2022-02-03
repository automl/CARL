from abc import abstractmethod
import numpy as np
from carl.utils.types import Context
from typing import Dict, Any


class AbstractSelector(object):
    def __init__(self, contexts: Dict[Any, Context]):
        self.context_ids = np.arange(len(contexts))
        self.contexts_keys = list(contexts.keys())
        self.contexts = contexts
        self.n_calls = 0  # type: int

    @abstractmethod
    def _select(self) -> Context:
        ...

    def select(self) -> Context:
        context = self._select()
        self.n_calls += 1
        return context


class RandomSelector(AbstractSelector):
    def _select(self):
        # TODO seed?
        context_id = np.random.choice(self.context_ids)
        context = self.contexts[self.contexts_keys[context_id]]
        return context


class RoundRobinSelector(AbstractSelector):
    def __init__(self, contexts: Dict[Any, Context]):
        super().__init__(contexts=contexts)
        self.context_id = -1  # type: int

    def _select(self):
        self.context_id = (self.context_id + 1) % len(self.contexts)
        context = self.contexts[self.contexts_keys[self.context_id]]
        return context
