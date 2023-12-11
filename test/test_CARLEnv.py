import unittest

from carl.envs.gymnasium.classic_control.carl_pendulum import CARLPendulum

CARLPendulum.render_mode = "rgb_array"


class TestStateConstruction(unittest.TestCase):
    def test_observation(self):
        env = CARLPendulum()
        context = CARLPendulum.get_default_context()
        obs, info = env.reset()
        self.assertEqual(type(obs), dict)
        self.assertTrue("obs" in obs, msg=str(obs))
        self.assertTrue("context" in obs)
        self.assertEqual(len(obs["context"]), len(context))

    def test_observation_emptycontext(self):
        env = CARLPendulum(obs_context_features=[])
        state, info = env.reset()
        self.assertEqual(len(state["context"]), 0)

    def test_observation_reducedcontext(self):
        n = 3
        context_keys = list(CARLPendulum.get_default_context().keys())[:n]
        env = CARLPendulum(obs_context_features=context_keys)
        state, info = env.reset()
        self.assertEqual(len(state["context"]), n)


if __name__ == "__main__":
    unittest.main()
