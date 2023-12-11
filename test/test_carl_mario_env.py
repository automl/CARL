import unittest

import carl.envs


class TestCarlMarioEnv(unittest.TestCase):
    def test_observation(self):
        env = carl.envs.CARLMarioEnv()
        context = env.get_default_context()
        obs, info = env.reset()
        self.assertEqual(type(obs), dict)
        self.assertTrue("obs" in obs, msg=str(obs))
        self.assertTrue("context" in obs)
        self.assertEqual(len(obs["context"]), len(context))

    def test_observation_emptycontext(self):
        env = carl.envs.CARLMarioEnv(obs_context_features=[])
        state, info = env.reset()
        self.assertEqual(len(state["context"]), 0)


if __name__ == "__main__":
    unittest.main()
