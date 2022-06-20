import unittest
from carl.context.sampling import get_default_context_and_bounds


class TestSampling(unittest.TestCase):
    def test_get_default_context_and_bounds(self):
        env_name = "CARLPendulumEnv"
        env_defaults, env_bounds = get_default_context_and_bounds(env_name=env_name)
        defaults = {'max_speed': 8.0, 'dt': 0.05, 'g': 10.0, 'm': 1.0, 'l': 1.0}
        self.assertDictEqual(env_defaults, defaults)



