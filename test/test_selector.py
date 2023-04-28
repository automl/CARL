from typing import Any, Dict

import unittest
from unittest.mock import patch

from carl.context.selection import (
    AbstractSelector,
    CustomSelector,
    RandomSelector,
    RoundRobinSelector,
)
from carl.utils.types import Context


def dummy_select(dummy):
    return None, None


class TestSelectors(unittest.TestCase):
    @staticmethod
    def generate_contexts() -> Dict[Any, Context]:
        n_context_features = 5
        keys = "abc"
        values = {str(i): i for i in range(n_context_features)}
        contexts = {k: v for k, v in zip(keys, values)}
        return contexts

    @patch.object(AbstractSelector, "_select", dummy_select)
    def test_abstract_selector(self):
        contexts = self.generate_contexts()
        selector = AbstractSelector(contexts=contexts)
        selector.select()
        selector.select()
        selector.select()
        selector.select()
        self.assertEqual(selector.n_calls, 4)

    def test_random_selector(self):
        contexts = self.generate_contexts()
        selector = RandomSelector(contexts=contexts)
        selector.select()
        selector.select()
        selector.select()

    def test_roundrobin_selector(self):
        contexts = self.generate_contexts()
        selector = RoundRobinSelector(contexts=contexts)

        self.assertEqual(None, selector.context_id)

        selector.select()
        self.assertEqual(selector.context_id, 0)
        self.assertEqual(selector.contexts_keys[selector.context_id], "a")

        selector.select()
        self.assertEqual(selector.context_id, 1)
        self.assertEqual(selector.contexts_keys[selector.context_id], "b")

        selector.select()
        self.assertEqual(selector.context_id, 2)
        self.assertEqual(selector.contexts_keys[selector.context_id], "c")

        selector.select()
        self.assertEqual(selector.context_id, 0)
        self.assertEqual(selector.contexts_keys[selector.context_id], "a")

    def test_custom_selector(self):
        def selector_function(inst: AbstractSelector):
            if inst.n_calls == 0:
                context_id = 1
            else:
                context_id = 0
            return inst.contexts[inst.contexts_keys[context_id]], context_id

        contexts = self.generate_contexts()
        selector = CustomSelector(
            contexts=contexts, selector_function=selector_function
        )

        selector.select()
        self.assertEqual(selector.context_id, 1)
        self.assertEqual(selector.contexts_keys[selector.context_id], "b")

        selector.select()
        self.assertEqual(selector.context_id, 0)
        self.assertEqual(selector.contexts_keys[selector.context_id], "a")
