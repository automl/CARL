from carl.envs.gymnasium.classic_control.carl_pendulum import CARLPendulum

CARLPendulum.render_mode = "rgb_array"


class TestStateConstruction:
    def test_observation(self):
        env = CARLPendulum()
        context = CARLPendulum.get_default_context()
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "obs" in obs
        assert "context" in obs
        assert len(obs["context"]) == len(context)

    def test_observation_emptycontext(self):
        env = CARLPendulum(obs_context_features=[])
        state, info = env.reset()
        assert len(state["context"]) == 0

    def test_observation_reducedcontext(self):
        n = 3
        context_keys = list(CARLPendulum.get_default_context().keys())[:n]
        env = CARLPendulum(obs_context_features=context_keys)
        state, info = env.reset()
        assert len(state["context"]) == n
