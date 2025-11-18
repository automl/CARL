import carl.envs


class TestCarlMarioEnv:
    def test_observation(self):
        env = carl.envs.CARLMarioEnv()
        context = env.get_default_context()
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "obs" in obs
        assert "context" in obs
        assert len(obs["context"]) == len(context)

    def test_observation_emptycontext(self):
        env = carl.envs.CARLMarioEnv(obs_context_features=[])
        state, info = env.reset()
        assert len(state["context"]) == 0
