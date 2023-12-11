import unittest

import gymnasium
import numpy as np

from carl.context.context_space import (
    ContextSpace,
    UniformFloatContextFeature,
    UniformIntegerContextFeature,
)

context_space_dict = {
    "gravity": UniformFloatContextFeature(
        "gravity", lower=0.1, upper=np.inf, default_value=9.8
    ),
    "masscart": UniformFloatContextFeature(
        "masscart", lower=0.1, upper=10, default_value=1.0
    ),
    "masspole": UniformFloatContextFeature(
        "masspole", lower=0.01, upper=1, default_value=0.1
    ),
    "length": UniformFloatContextFeature(
        "length", lower=0.05, upper=5, default_value=0.5
    ),
    "force_mag": UniformFloatContextFeature(
        "force_mag", lower=1, upper=100, default_value=10.0
    ),
    "tau": UniformFloatContextFeature(
        "tau", lower=0.002, upper=0.2, default_value=0.02
    ),
}

context_space_dict_othertypes = {
    "gravity": UniformFloatContextFeature(
        "gravity", lower=0.1, upper=np.inf, default_value=9.8
    ),
    "masscart": UniformIntegerContextFeature(
        "masscart", lower=1, upper=10, default_value=1
    ),
}


class TestContextSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.default_context = {
            "gravity": 9.8,
            "masscart": 1,
            "masspole": 0.1,
            "length": 0.5,
            "force_mag": 10,
            "tau": 0.02,
        }
        self.context_space = ContextSpace(context_space=context_space_dict)
        return super().setUp()

    def test_insert_defaults(self):
        context_with_defaults = self.context_space.insert_defaults({})
        self.assertDictEqual(context_with_defaults, self.default_context)

    def test_get_default_context(self):
        default_context = self.context_space.get_default_context()
        self.assertDictEqual(default_context, self.default_context)

    def test_get_lower_and_upper_bound(self):
        bounds_gt = (0.05, 5)
        bounds = self.context_space.get_lower_and_upper_bound("length")
        self.assertTupleEqual(bounds_gt, bounds)

    def test_to_gymnasium_space_type(self):
        space = self.context_space.to_gymnasium_space(as_dict=False)
        self.assertEqual(type(space), gymnasium.spaces.Box)

        space = self.context_space.to_gymnasium_space(as_dict=True)
        self.assertEqual(type(space), gymnasium.spaces.Dict)

    def test_to_gynasium_space(self):
        cspace = ContextSpace(context_space_dict_othertypes)
        cspace.to_gymnasium_space()

    def test_verify_context(self):
        # Unknown context feature name
        context = {"hihi": 39, "gravity": 3}
        is_valid = self.context_space.verify_context(context)
        self.assertEqual(is_valid, False)

        # Out of bounds
        context = {"masscart": -10}
        is_valid = self.context_space.verify_context(context)
        self.assertEqual(is_valid, False)

    def test_sample(self):
        context = self.context_space.sample_contexts(["gravity"], size=1)
        is_valid = self.context_space.verify_context(context)
        self.assertEqual(is_valid, True)

        contexts = self.context_space.sample_contexts(["gravity"], size=10)
        self.assertTrue(len(contexts) == 10)
        for context in contexts:
            is_valid = self.context_space.verify_context(context)
            self.assertEqual(is_valid, True)

        contexts = self.context_space.sample_contexts(None, size=10)
        self.assertTrue(len(contexts) == 10)
        for context in contexts:
            is_valid = self.context_space.verify_context(context)
            self.assertEqual(is_valid, True)

        with self.assertRaises(ValueError):
            self.context_space.sample_contexts(["false_feature"], size=0)


if __name__ == "__main__":
    unittest.main()
