import unittest

from carl.envs.brax import (
    CARLAnt,
    CARLHalfcheetah,
    CARLFetch,
    BraxWalkerGoalWrapper,
    BraxLanguageWrapper,
    sample_walker_language_goals,
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
        contexts = sample_walker_language_goals(10, low=4, high=200)
        assert len(contexts.keys()) == 10
        assert "target_distance" in contexts[0].keys()
        assert "target_direction" in contexts[0].keys()
        assert all([contexts[i]["target_direction"] in DIRECTIONS for i in range(10)])
        assert all([contexts[i]["target_distance"] <= 200 for i in range(10)])
        assert all([contexts[i]["target_distance"] >= 4 for i in range(10)])

    def test_normal_sampling(self):
        contexts = sample_walker_language_goals(10, normal=True, low=4, high=200)
        assert len(contexts.keys()) == 10
        assert "target_distance" in contexts[0].keys()
        assert "target_direction" in contexts[0].keys()
        assert all([contexts[i]["target_direction"] in DIRECTIONS for i in range(10)])
        assert all([contexts[i]["target_distance"] <= 200 for i in range(10)])
        assert all([contexts[i]["target_distance"] >= 4 for i in range(10)])


class TestGoalWrapper(unittest.TestCase):
    def test_reset(self):
        contexts = sample_walker_language_goals(10, low=4, high=200)
        env = CARLAnt(contexts=contexts)
        wrapped_env = BraxWalkerGoalWrapper(env)

        assert wrapped_env.position is None
        state = wrapped_env.reset()
        assert state is not None
        assert wrapped_env.position is not None

        state, info = wrapped_env.reset(return_info=True)
        assert state is not None
        assert info is not None

        env = CARLHalfcheetah(contexts=contexts)
        wrapped_env = BraxWalkerGoalWrapper(env)

        assert wrapped_env.position is None
        state = wrapped_env.reset()
        assert state is not None
        assert wrapped_env.position is not None

        state, info = wrapped_env.reset(return_info=True)
        assert state is not None
        assert info is not None

    def test_reward_scale(self):
        contexts = sample_walker_language_goals(10, low=4, high=200)
        env = CARLAnt(contexts=contexts)
        wrapped_env = BraxWalkerGoalWrapper(env)
        basic_env = CARLAnt()

        for _ in range(10):
            wrapped_env.reset()
            basic_env.reset()
            for _ in range(10):
                action = basic_env.action_space.sample()
                _, wrapped_reward, _, _ = wrapped_env.step(action)
                _, basic_reward, _, _ = basic_env.step(action)
                assert wrapped_reward >= basic_reward - 0.01

        contexts = sample_walker_language_goals(10, low=4, high=200)
        env = CARLHalfcheetah(contexts=contexts)
        wrapped_env = BraxWalkerGoalWrapper(env)
        basic_env = CARLHalfcheetah()

        for _ in range(10):
            wrapped_env.reset()
            basic_env.reset()
            for _ in range(10):
                action = basic_env.action_space.sample()
                _, wrapped_reward, _, _ = wrapped_env.step(action)
                _, basic_reward, _, _ = basic_env.step(action)
                assert wrapped_reward >= basic_reward - 0.01


class TestLanguageWrapper(unittest.TestCase):
    def test_reset(self) -> None:
        env = CARLFetch()
        wrapped_env = BraxLanguageWrapper(env)
        state = wrapped_env.reset()
        assert type(state) is dict
        assert "env_state" in state.keys()
        assert "goal" in state.keys()
        assert type(state["goal"]) is str
        assert str(wrapped_env.context["target_distance"]) in state["goal"]
        assert str(wrapped_env.context["target_radius"]) in state["goal"]
        state, info = wrapped_env.reset(return_info=True)
        assert info is not None
        assert type(state) is dict

        contexts = sample_walker_language_goals(10, low=4, high=200)
        env = CARLAnt(contexts=contexts)
        wrapped_env = BraxLanguageWrapper(env)
        state = wrapped_env.reset()
        assert type(state) is dict
        assert "env_state" in state.keys()
        assert "goal" in state.keys()
        assert type(state["goal"]) is str
        assert str(wrapped_env.context["target_distance"]) in state["goal"]
        assert str(wrapped_env.context["target_direction"]) in state["goal"]
        state, info = wrapped_env.reset(return_info=True)
        assert info is not None
        assert type(state) is dict

    def test_step(self):
        contexts = sample_walker_language_goals(10, low=4, high=200)
        env = CARLFetch(contexts=contexts)
        wrapped_env = BraxLanguageWrapper(env)
        wrapped_env.reset()
        for _ in range(10):
            action = wrapped_env.action_space.sample()
            state, _, _, _ = wrapped_env.step(action)
            assert type(state) is dict
            assert "env_state" in state.keys()
            assert "goal" in state.keys()
            assert type(state["goal"]) is str
            assert str(wrapped_env.context["target_distance"]) in state["goal"]
            assert str(wrapped_env.context["target_radius"]) in state["goal"]

        env = CARLAnt(contexts=contexts)
        wrapped_env = BraxLanguageWrapper(env)
        wrapped_env.reset()
        for _ in range(10):
            action = wrapped_env.action_space.sample()
            state, _, _, _ = wrapped_env.step(action)
            assert type(state) is dict
            assert "env_state" in state.keys()
            assert "goal" in state.keys()
            assert type(state["goal"]) is str
            assert str(wrapped_env.context["target_distance"]) in state["goal"]
            assert str(wrapped_env.context["target_direction"]) in state["goal"]
