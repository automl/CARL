import unittest
from carl.context.sampling import get_default_context_and_bounds
from experiments.carlbench.context_sampling import ContextSampler


class TestSampling(unittest.TestCase):
    def test_get_default_context_and_bounds(self):
        env_name = "CARLPendulumEnv"
        env_defaults, env_bounds = get_default_context_and_bounds(env_name=env_name)
        defaults = {"max_speed": 8.0, "dt": 0.05, "g": 10.0, "m": 1.0, "l": 1.0}
        self.assertDictEqual(env_defaults, defaults)

    def test_context_sampler(self):
        env_name = "CARLPendulumEnv"
        cs = ContextSampler(
            env_name=env_name,
            difficulty="easy",
            n_samples=1,
            context_feature_names=["m", "l", "g"],
            seed=455,
        )
        contexts = cs.sample_contexts()
        true_dict = {
            0: {
                "max_speed": 8.0,
                "dt": 0.05,
                "g": 9.748400206554313,
                "m": 0.8727822986909317,
                "l": 0.9215523401261485,
            }
        }
        for context_sampled, context_true in zip(contexts.values(), true_dict.values()):
            for k in context_sampled.keys():
                self.assertAlmostEqual(context_sampled[k], context_true[k])
