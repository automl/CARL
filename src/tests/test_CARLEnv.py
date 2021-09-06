import unittest
import numpy as np

from src.envs.classic_control.meta_pendulum import CARLPendulumEnv


class TestStateConstruction(unittest.TestCase):
    def test_hiddenstate(self):
        """
        Test if we set hide_context = True that we get the original, normal state.
        """
        env = CARLPendulumEnv(
            contexts={},
            hide_context=True,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=None,
        )
        env.reset()
        action = [0.01]  # torque
        state, reward, done, info = env.step(action=action)
        env.close()
        self.assertEqual(len(state), 3)

    def test_visiblestate(self):
        """
        Test if we set hide_context = False and state_context_features=None that we get the normal state extended by
        all context features.
        """
        env = CARLPendulumEnv(
            contexts={},
            hide_context=False,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=None,
        )
        env.reset()
        action = [0.01]  # torque
        state, reward, done, info = env.step(action=action)
        env.close()
        self.assertEqual(len(state), 8)

    def test_visiblestate_customnone(self):
        """
        Test if we set hide_context = False and state_context_features="changing_context_features" that we get the
        normal state, not extended by context features..
        """
        env = CARLPendulumEnv(
            contexts={},
            hide_context=False,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=["changing_context_features"],
        )
        env.reset()
        action = [0.01]  # torque
        state, reward, done, info = env.step(action=action)
        env.close()
        # Because we don't change any context features the state length should be 3
        self.assertEqual(len(state), 3)

    def test_visiblestate_custom(self):
        """
        Test if we set hide_context = False and state_context_features=["g", "m"] that we get the
        normal state, extended by the context feature values of g and m.
        """
        env = CARLPendulumEnv(
            contexts={},
            hide_context=False,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=["g", "m"],
        )
        env.reset()
        action = [0.01]  # torque
        state, reward, done, info = env.step(action=action)
        env.close()
        # state should be of length 5 because we add two context features
        self.assertEqual(len(state), 5)
        # last two state values should be g = 10 and m = 1
        self.assertTrue(np.all(np.equal(state[-2:], [10., 1.])))


if __name__ == '__main__':
    unittest.main()
