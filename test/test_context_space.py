import gymnasium
import numpy as np
import pytest

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


class TestContextSpace:
    @staticmethod
    def generate_context_space() -> None:
        default_context = {
            "gravity": 9.8,
            "masscart": 1,
            "masspole": 0.1,
            "length": 0.5,
            "force_mag": 10,
            "tau": 0.02,
        }
        context_space = ContextSpace(context_space=context_space_dict)
        return default_context, context_space

    def test_insert_defaults(self):
        default_context, context_space = self.generate_context_space()
        context_with_defaults = context_space.insert_defaults({})
        assert len(context_with_defaults) == len(default_context)
        for key in default_context:
            assert context_with_defaults[key] == default_context[key]

    def test_get_default_context(self):
        default_context, _ = self.generate_context_space()
        assert len(default_context) == len(default_context)
        for key in default_context:
            assert default_context[key] == default_context[key]

    def test_get_lower_and_upper_bound(self):
        _, context_space = self.generate_context_space()
        bounds_gt = (0.05, 5)
        bounds = context_space.get_lower_and_upper_bound("length")
        assert bounds == bounds_gt

    def test_to_gymnasium_space_type(self):
        _, context_space = self.generate_context_space()
        space = context_space.to_gymnasium_space(as_dict=False)
        assert isinstance(space, gymnasium.spaces.Box)

        space = context_space.to_gymnasium_space(as_dict=True)
        assert isinstance(space, gymnasium.spaces.Dict)

    def test_to_gynasium_space(self):
        cspace = ContextSpace(context_space_dict_othertypes)
        cspace.to_gymnasium_space()

    def test_verify_context(self):
        _, context_space = self.generate_context_space()
        # Unknown context feature name
        context = {"hihi": 39, "gravity": 3}
        is_valid = context_space.verify_context(context)
        assert not is_valid

        # Out of bounds
        context = {"masscart": -10}
        is_valid = context_space.verify_context(context)
        assert not is_valid

    def test_sample(self):
        _, context_space = self.generate_context_space()
        context = context_space.sample_contexts(["gravity"], size=1)
        is_valid = context_space.verify_context(context)
        assert is_valid

        contexts = context_space.sample_contexts(["gravity"], size=10)
        assert len(contexts) == 10
        for context in contexts:
            is_valid = context_space.verify_context(context)
            assert is_valid

        contexts = context_space.sample_contexts(None, size=10)
        assert len(contexts) == 10
        for context in contexts:
            is_valid = context_space.verify_context(context)
            assert is_valid

        with pytest.raises(ValueError):
            context_space.sample_contexts(["false_feature"], size=0)
