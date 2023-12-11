import unittest

from carl.context.context_space import (
    ContextSpace,
    NormalFloatContextFeature,
    UniformFloatContextFeature,
)
from carl.context.sampler import ContextSampler

context_space_dict = {
    "gravity": UniformFloatContextFeature(
        "gravity", lower=1, upper=10, default_value=9.8
    )
}
sample_dist = {
    "gravity": NormalFloatContextFeature(
        "gravity", mu=9.8, sigma=0.0, default_value=9.8
    )
}


class TestContextSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.cspace = ContextSpace(context_space_dict)
        self.sampler = ContextSampler(
            context_distributions=sample_dist,
            context_space=ContextSpace(context_space_dict),
            seed=0,
            name="TestSampler",
        )
        return super().setUp()

    def test_init(self):
        ContextSampler(
            context_distributions=sample_dist,  # as dict
            context_space=self.cspace,
            seed=0,
            name="TestSampler",
        )
        ContextSampler(
            context_distributions=list(sample_dist.values()),  # as list/iterable
            context_space=self.cspace,
            seed=0,
            name="TestSampler",
        )

        with self.assertRaises(ValueError):
            ContextSampler(
                context_distributions=0,
                context_space=self.cspace,
                seed=0,
                name="TestSampler",
            )

    def test_sample_contexts(self):
        contexts = self.sampler.sample_contexts(n_contexts=3)
        self.assertEqual(len(contexts), 3)
        self.assertEqual(contexts[0]["gravity"], 9.8)

        contexts = self.sampler.sample_contexts(n_contexts=1)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["gravity"], 9.8)


if __name__ == "__main__":
    unittest.main()
