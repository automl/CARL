from typing import Any, Dict

import pytest

from carl.envs.gymnasium.classic_control.carl_pendulum import CARLPendulum
from carl.utils.types import Context

CARLPendulum.render_mode = "rgb_array"


class TestContextSelection:
    @staticmethod
    def generate_contexts() -> Dict[Any, Context]:
        keys = "abc"
        context = {"dt": 0.03, "m": 1.0, "l": 1.8}
        contexts = {k: context for k in keys}
        return contexts

    def test_default_selector(self):
        from carl.context.selection import RoundRobinSelector

        contexts = self.generate_contexts()
        env = CARLPendulum(contexts=contexts)

        env.reset()
        assert isinstance(env.context_selector, RoundRobinSelector)
        assert env.context_selector.n_calls == 1

        env.reset()
        assert env.context_selector.n_calls == 2

    def test_roundrobin_selector_init(self):
        from carl.context.selection import RoundRobinSelector

        contexts = self.generate_contexts()
        env = CARLPendulum(
            contexts=contexts, context_selector=RoundRobinSelector(contexts=contexts)
        )
        self.assertEqual(type(env.context_selector), RoundRobinSelector)

    def test_random_selector_init(self):
        from carl.context.selection import RandomSelector

        contexts = self.generate_contexts()
        env = CARLPendulum(
            contexts=contexts, context_selector=RandomSelector(contexts=contexts)
        )
        assert isinstance(env.context_selector, RandomSelector)

    def test_random_selectorclass_init(self):
        from carl.context.selection import RandomSelector

        contexts = self.generate_contexts()
        env = CARLPendulum(contexts=contexts, context_selector=RandomSelector)
        assert isinstance(env.context_selector, RandomSelector)

    def test_unknown_selector_init(self):
        with pytest.raises(ValueError):
            contexts = self.generate_contexts()
            _ = CARLPendulum(contexts=contexts, context_selector="bork")
