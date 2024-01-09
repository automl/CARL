import unittest

from carl.context.context_space import NormalFloatContextFeature, CategoricalContextFeature
from carl.context.sampler import ContextSampler
from carl.envs import CARLBraxAnt, CARLBraxHalfcheetah
from carl.envs.brax.brax_walker_goal_wrapper import BraxLanguageWrapper, BraxWalkerGoalWrapper

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
        context_distributions = [NormalFloatContextFeature("target_distance", mu=9.8, sigma=1), CategoricalContextFeature("target_direction", choices=DIRECTIONS)]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        assert len(contexts.keys()) == 10
        assert "target_distance" in contexts[0].keys()
        assert "target_direction" in contexts[0].keys()
        assert all([contexts[i]["target_direction"] in DIRECTIONS for i in range(10)])
        assert all([contexts[i]["target_distance"] <= 200 for i in range(10)])
        assert all([contexts[i]["target_distance"] >= 4 for i in range(10)])

    def test_normal_sampling(self):
        context_distributions = [NormalFloatContextFeature("target_distance", mu=9.8, sigma=1), CategoricalContextFeature("target_direction", choices=DIRECTIONS)]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        assert len(contexts.keys()) == 10
        assert "target_distance" in contexts[0].keys()
        assert "target_direction" in contexts[0].keys()
        assert all([contexts[i]["target_direction"] in DIRECTIONS for i in range(10)])
        assert all([contexts[i]["target_distance"] <= 200 for i in range(10)])
        assert all([contexts[i]["target_distance"] >= 4 for i in range(10)])


class TestGoalWrapper(unittest.TestCase):
    def test_reset(self):
        context_distributions = [NormalFloatContextFeature("target_distance", mu=9.8, sigma=1), CategoricalContextFeature("target_direction", choices=DIRECTIONS)]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxAnt(contexts=contexts)

        assert isinstance(env.env, BraxWalkerGoalWrapper)
        assert env.position is None

        state, info = env.reset()
        assert state is not None
        assert info is not None
        assert env.position is not None

        context_distributions = [NormalFloatContextFeature("target_distance", mu=9.8, sigma=1), CategoricalContextFeature("target_direction", choices=DIRECTIONS)]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxHalfcheetah.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxHalfcheetah(contexts=contexts, use_language_goals=True)

        assert isinstance(env.env, BraxLanguageWrapper)
        assert env.position is None

        state, info = env.reset()
        assert state is not None
        assert info is not None
        assert env.position is not None

    def test_reward_scale(self):
        context_distributions = [NormalFloatContextFeature("target_distance", mu=9.8, sigma=1), CategoricalContextFeature("target_direction", choices=DIRECTIONS)]
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
                assert wrapped_reward >= 0

        context_distributions = [NormalFloatContextFeature("target_distance", mu=9.8, sigma=1), CategoricalContextFeature("target_direction", choices=DIRECTIONS)]
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
                assert wrapped_reward >= 0


class TestLanguageWrapper(unittest.TestCase):
    def test_reset(self) -> None:
        context_distributions = [NormalFloatContextFeature("target_distance", mu=9.8, sigma=1), CategoricalContextFeature("target_direction", choices=DIRECTIONS)]
        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLBraxAnt.get_context_space(),
            seed=0,
        )
        contexts = context_sampler.sample_contexts(n_contexts=10)
        env = CARLBraxAnt(contexts=contexts, use_language_goals=True)
        state, info = env.reset()
        assert type(state) is dict
        assert "obs" in state.keys()
        assert "goal" in state["obs"].keys()
        assert type(state["obs"]["goal"]) is str
        assert str(env.context["target_distance"]) in state["obs"]["goal"]
        assert "north north east" in state["obs"]["goal"]
        assert info is not None

    def test_step(self):
        context_distributions = [NormalFloatContextFeature("target_distance", mu=9.8, sigma=1), CategoricalContextFeature("target_direction", choices=DIRECTIONS)]
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
            assert type(state) is dict
            assert "obs" in state.keys()
            assert "goal" in state["obs"].keys()
            assert type(state["obs"]["goal"]) is str
            assert "north north east" in state["obs"]["goal"]
            assert str(env.context["target_distance"]) in state["obs"]["goal"]

        context_distributions = [NormalFloatContextFeature("target_distance", mu=9.8, sigma=1), CategoricalContextFeature("target_direction", choices=DIRECTIONS)]
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
            assert type(state) is dict
            assert "obs" in state.keys()
            assert "goal" not in state.keys()
