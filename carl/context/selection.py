from abc import abstractmethod
import numpy as np
from carl.utils.types import Context
from typing import Dict, Any, Optional, Tuple, List, Callable


class AbstractSelector(object):
    """
    Base class for context selectors.

    Context is selected when calling `select`, not in `__init__`.


    Parameters
    ----------
    contexts: Dict[Any, Context]
        Context set. A `Context` is a Dict[str, Any].


    Attributes
    ----------
    contexts : Dict[Any, Context]
        Context set.
    context_ids : List[int]
        Integer index for contexts.
    contexts_keys : List[Any]
        Keys of contexts dictionary.
    n_calls : int
        Number of times `select` has been called.
    context_id : Optional[int]
        Context id of current selected context. Is None at first.

    """

    def __init__(self, contexts: Dict[Any, Context]):
        self.contexts: Dict[Any, Context] = contexts
        self.context_ids: List[int] = list(np.arange(len(contexts)))
        self.contexts_keys: List[Any] = list(contexts.keys())
        self.n_calls: int = 0
        self.context_id: Optional[
            int
        ] = None  # holds index of current context (integer index of context keys)

    @abstractmethod
    def _select(self) -> Tuple[Context, int]:
        """
        Select next context (internal).

        Should be implemented in child class, internal use.

        Returns
        -------
        context : Context
            Selected context.
        context_id : int
            Integer id of selected context.
        """
        ...

    def select(self) -> Context:
        """
        Select next context (API).

        Returns
        -------
        context : Context
            Selected context.
        """
        context, context_id = self._select()
        self.context_id = context_id
        self.n_calls += 1
        return context


class RandomSelector(AbstractSelector):
    """
    Random Context Selector.
    """

    def _select(self):
        # TODO seed?
        context_id = np.random.choice(self.context_ids)
        context = self.contexts[self.contexts_keys[context_id]]
        return context, context_id


class RoundRobinSelector(AbstractSelector):
    """
    Round robin context selector.

    Iterate through all contexts and then start at the first again.
    """

    def _select(self):
        if self.context_id is None:
            self.context_id = -1
        self.context_id = (self.context_id + 1) % len(self.contexts)
        context = self.contexts[self.contexts_keys[self.context_id]]
        return context, self.context_id


class CustomSelector(AbstractSelector):
    """
    Custom selector.

    Pass an individual function implementing selection logic. Could also be implemented by subclassing
    `AbstractSelector`.

    Parameters
    ----------
    contexts: Dict[Any, Context]
        Set of contexts.
    selector_function: callable
        Function receiving a pointer to the selector implementing selection logic.
        See example below.

    Examples
    --------
    >>> def selector_function(inst: AbstractSelector) -> Tuple[Context, int]:
    >>>     if inst.n_calls == 0:
    >>>         context_id = 1
    >>>     else:
    >>>         context_id = 0
    >>>     return inst.contexts[inst.contexts_keys[context_id]], context_id
    >>> contexts = ...
    >>> selector = CustomSelector(contexts=contexts, selector_function=selector_function)

    This custom selector selects a context id based on the number of times `select` has been called.

    """

    def __init__(
        self,
        contexts: Dict[Any, Context],
        selector_function: Callable[[AbstractSelector], Tuple[Context, int]],
    ):
        super().__init__(contexts=contexts)
        self.selector_function = selector_function

    def _select(self):
        context, context_id = self.selector_function(inst=self)
        self.context_id = context_id
        return context, context_id
