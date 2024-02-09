import unittest

from carl.context.context_space import (
    CategoricalContextFeature,
    NormalFloatContextFeature,
)
from carl.context.sampler import ContextSampler
from carl.envs import CARLBraxAnt, CARLBraxHalfcheetah
from carl.envs.brax.brax_walker_goal_wrapper import (
    BraxLanguageWrapper,
    BraxWalkerGoalWrapper,
)

DIRECTIONS = [
    1,  # north
    3,  # south
    2,  # east
    4,  # west
    12,
    32,
    14,
    34,
    112,
    332,
    114,
    334,
    212,
    232,
    414,
    434,
]


class TestGoalSampling(unittest.TestCase):
    def test_uniform_sampling(self):
        context_distributions = [
            NormalFloatContextFeature("target_distance", mu=9.8, sigma=1),
            CategoricalContextFeature("target_direction", choices=DIRECTIONS),
        ]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        assert len(contexts.keys()) == 10
        assert "target_distance" in contexts[0].keys(), "target_distance not in context"
        assert (
            "target_direction" in contexts[0].keys()
        ), "target_direction not in context"
        assert all(
            [contexts[i]["target_direction"] in DIRECTIONS for i in range(10)]
        ), "Not all directions are valid."
        assert all(
            [contexts[i]["target_distance"] <= 200 for i in range(10)]
        ), "Not all distances are valid (too large)."
        assert all(
            [contexts[i]["target_distance"] >= 4 for i in range(10)]
        ), "Not all distances are valid (too small)."

    def test_normal_sampling(self):
        context_distributions = [
            NormalFloatContextFeature("target_distance", mu=9.8, sigma=1),
            CategoricalContextFeature("target_direction", choices=DIRECTIONS),
        ]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        assert (
            len(contexts.keys()) == 10
        ), "Number of sampled contexts does not match the requested number."
        assert "target_distance" in contexts[0].keys(), "target_distance not in context"
        assert (
            "target_direction" in contexts[0].keys()
        ), "target_direction not in context"
        assert all(
            [contexts[i]["target_direction"] in DIRECTIONS for i in range(10)]
        ), "Not all directions are valid."
        assert all(
            [contexts[i]["target_distance"] <= 200 for i in range(10)]
        ), "Not all distances are valid (too large)."
        assert all(
            [contexts[i]["target_distance"] >= 4 for i in range(10)]
        ), "Not all distances are valid (too small)."


class TestGoalWrapper(unittest.TestCase):
    def test_reset(self):
        context_distributions = [
            NormalFloatContextFeature("target_distance", mu=9.8, sigma=1),
            CategoricalContextFeature("target_direction", choices=DIRECTIONS),
        ]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxAnt(contexts=contexts)

        assert isinstance(env.env, BraxWalkerGoalWrapper)
        assert env.position is None, "Position set before reset."

        state, info = env.reset()
        assert state is not None, "No state returned."
        assert info is not None, "No info returned."
        assert env.position is not None, "Position not set."

        context_distributions = [
            NormalFloatContextFeature("target_distance", mu=9.8, sigma=1),
            CategoricalContextFeature("target_direction", choices=DIRECTIONS),
        ]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxHalfcheetah.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxHalfcheetah(contexts=contexts, use_language_goals=True)

        assert isinstance(env.env, BraxLanguageWrapper), "Language wrapper not used."
        assert env.position is None, "Position set before reset."

        state, info = env.reset()
        assert state is not None, "No state returned."
        assert info is not None, "No info returned."
        assert env.position is not None, "Position not set."

    def test_reward_scale(self):
        context_distributions = [
            NormalFloatContextFeature("target_distance", mu=9.8, sigma=1),
            CategoricalContextFeature("target_direction", choices=DIRECTIONS),
        ]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxAnt(contexts=contexts)

        for _ in range(10):
            env.reset()
            for _ in range(10):
                action = env.action_space.sample()
                _, wrapped_reward, _, _, _ = env.step(action)
                assert wrapped_reward >= 0, "Negative reward."

        context_distributions = [
            NormalFloatContextFeature("target_distance", mu=9.8, sigma=1),
            CategoricalContextFeature("target_direction", choices=DIRECTIONS),
        ]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxHalfcheetah.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxHalfcheetah(contexts=contexts)

        for _ in range(10):
            env.reset()
            for _ in range(10):
                action = env.action_space.sample()
                _, wrapped_reward, _, _, _ = env.step(action)
                assert wrapped_reward >= 0, "Negative reward."


class TestLanguageWrapper(unittest.TestCase):
    def test_reset(self) -> None:
        context_distributions = [
            NormalFloatContextFeature("target_distance", mu=9.8, sigma=1),
            CategoricalContextFeature("target_direction", choices=DIRECTIONS),
        ]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxAnt(contexts=contexts, use_language_goals=True)
        state, info = env.reset()
        assert type(state) is dict, "State is not a dictionary."
        assert "obs" in state.keys(), "Observation not in state."
        assert "goal" in state["obs"].keys(), "Goal not in observation."
        assert type(state["obs"]["goal"]) is str, "Goal is not a string."
        assert (
            str(env.context["target_distance"]) in state["obs"]["goal"]
        ), "Distance not in goal."
        assert "north north east" in state["obs"]["goal"], "Direction not in goal."
        assert info is not None, "No info returned."

    def test_step(self):
        context_distributions = [
            NormalFloatContextFeature("target_distance", mu=9.8, sigma=1),
            CategoricalContextFeature("target_direction", choices=DIRECTIONS),
        ]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxHalfcheetah.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxHalfcheetah(contexts=contexts, use_language_goals=True)
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            state, _, _, _, _ = env.step(action)
            assert type(state) is dict, "State is not a dictionary."
            assert "obs" in state.keys(), "Observation not in state."
            assert "goal" in state["obs"].keys(), "Goal not in observation."
            assert type(state["obs"]["goal"]) is str, "Goal is not a string."
            assert "north north east" in state["obs"]["goal"], "Direction not in goal."
            assert (
                str(env.context["target_distance"]) in state["obs"]["goal"]
            ), "Distance not in goal."

        context_distributions = [
            NormalFloatContextFeature("target_distance", mu=9.8, sigma=1),
            CategoricalContextFeature("target_direction", choices=DIRECTIONS),
        ]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxAnt(contexts=contexts)
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            state, _, _, _, _ = env.step(action)
            assert type(state) is dict, "State is not a dictionary."
            assert "obs" in state.keys(), "Observation not in state."
            assert "goal" not in state.keys(), "Goal in observation."
