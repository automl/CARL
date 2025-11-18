import pytest

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
        "gravity", mu=9.8, sigma=0.0, default_value=9.8, upper=20, lower=1
    )
}


class TestContextSampler:
    def test_init(self):
        cspace = ContextSpace(context_space_dict)
        ContextSampler(
            context_distributions=sample_dist,  # as dict
            context_space=cspace,
            seed=0,
            name="TestSampler",
        )
        ContextSampler(
            context_distributions=list(sample_dist.values()),  # as list/iterable
            context_space=cspace,
            seed=0,
            name="TestSampler",
        )

        with pytest.raises(ValueError):
            ContextSampler(
                context_distributions=0,
                context_space=cspace,
                seed=0,
                name="TestSampler",
            )

    def test_sample_contexts(self):
        cspace = ContextSpace(context_space_dict)
        sampler = ContextSampler(
            context_distributions=sample_dist,
            context_space=cspace,
            seed=0,
            name="TestSampler",
        )
        contexts = sampler.sample_contexts(n_contexts=3)
        assert len(contexts) == 3
        assert contexts[0]["gravity"] == 9.8

        contexts = sampler.sample_contexts(n_contexts=1)
        assert len(contexts) == 1
        assert contexts[0]["gravity"] == 9.8
