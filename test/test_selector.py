import unittest

from typing import Dict, Any
from carl.utils.types import Context
from carl.context.selection import RoundRobinSelector, RandomSelector, AbstractSelector


class TestSelectors(unittest.TestCase):
    @staticmethod
    def generate_contexts() -> Dict[Any, Context]:
        n_context_features = 5
        keys = "abc"
        values = {str(i): i for i in range(n_context_features)}
        contexts = {k: v for k, v in zip(keys, values)}
        return contexts

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

        self.assertEqual(selector.context_id, -1)

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