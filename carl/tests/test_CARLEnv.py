import unittest
import numpy as np

from carl.envs.classic_control.carl_pendulum import CARLPendulumEnv


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
        self.assertEqual(3, len(state))

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
        self.assertEqual(8, len(state))

    def test_visiblestate_customnone(self):
        """
        Test if we set hide_context = False and state_context_features="changing_context_features" that we get the
        normal state, not extended by context features.
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
        self.assertEqual(3, len(state))

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
        self.assertEqual(5, len(state))

    def test_visiblestate_changingcontextfeatures_nochange(self):
        """
        Test if we set hide_context = False and state_context_features="changing_context_features" that we get the
        normal state, extended by the context features which are changing in the set of contexts. Here: None are
        changing.
        """
        contexts = {
            "0": {"max_speed": 8., "dt":  0.05, "g": 10.0, "m": 1., "l": 1.},
            "1": {"max_speed": 8., "dt":  0.05, "g": 10.0, "m": 1., "l": 1.},
            "2": {"max_speed": 8., "dt":  0.05, "g": 10.0, "m": 1., "l": 1.},
            "3": {"max_speed": 8., "dt":  0.05, "g": 10.0, "m": 1., "l": 1.},
        }
        env = CARLPendulumEnv(
            contexts=contexts,
            hide_context=False,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=["changing_context_features"],
        )
        env.reset()
        action = [0.01]  # torque
        state, reward, done, info = env.step(action=action)
        env.close()
        # state should be of length 3 because all contexts are the same
        self.assertEqual(3, len(state))

    def test_visiblestate_changingcontextfeatures_change(self):
        """
        Test if we set hide_context = False and state_context_features="changing_context_features" that we get the
        normal state, extended by the context features which are changing in the set of contexts.
        Here: Two are changing.
        """
        contexts = {
            "0": {"max_speed": 8., "dt":  0.03, "g": 10.0, "m": 1., "l": 1.},
            "1": {"max_speed": 8., "dt":  0.05, "g": 10.0, "m": 1., "l": .95},
            "2": {"max_speed": 8., "dt":  0.05, "g": 10.0, "m": 1., "l": .3},
            "3": {"max_speed": 8., "dt":  0.05, "g": 10.0, "m": 1., "l": 1.3},
        }
        env = CARLPendulumEnv(
            contexts=contexts,
            hide_context=False,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=["changing_context_features"],
        )
        env.reset()
        action = [0.01]  # torque
        state, reward, done, info = env.step(action=action)
        env.close()
        # state should be of length 5 because two features are changing (dt and l)
        self.assertEqual(5, len(state))


if __name__ == '__main__':
    unittest.main()
