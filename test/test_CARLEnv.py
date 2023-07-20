import unittest

import numpy as np

from carl.envs.gymnasium.classic_control.carl_pendulum import CARLPendulum

CARLPendulum.render_mode = "rgb_array"


class TestStateConstruction(unittest.TestCase):
    def test_dict_observation_space(self):
        contexts = {"0": {"max_speed": 8.0, "dt": 0.03, "g": 10.0, "m": 1.0, "l": 1.0}}
        env = CARLPendulum(
            contexts=contexts,
            hide_context=False,
            dict_observation_space=True,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=["changing_context_features"],
        )
        obs, _ = env.reset()
        self.assertEqual(type(obs), dict)
        self.assertTrue("state" in obs)
        self.assertTrue("context" in obs)
        action = [0.01]  # torque
        next_obs, reward, terminated, truncated, info = env.step(action=action)
        env.close()

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

class TestEpisodeTermination(unittest.TestCase):
    def test_episode_termination(self):
        """
        Test if we set hide_context = True that we get the original, normal state.
        """
        ep_length = 100
        env = CARLPendulumEnv(
            contexts={},
            hide_context=True,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=None,
            max_episode_length=ep_length,
        )
        env.reset()
        action = [0.0]  # torque
        done = False
        counter = 0
        while not done:
            state, reward, terminated, truncated, info = env.step(action=action)
            counter += 1
            self.assertTrue(counter <= ep_length)
            if terminated or truncated:
                done = True

            if counter > ep_length:
                break
        env.close()


class TestContextSampler(unittest.TestCase):
    def test_get_defaults(self):
        from carl.context.sampling import get_default_context_and_bounds

    def test_context_feature_scaling_by_mean(self):
        contexts = {
            # order is important because context "0" is checked in the test
            # because of the reset context "0" must come seond
            "1": {"max_speed": 16.0, "dt": 0.06, "g": 20.0, "m": 2.0, "l": 3.6},
            "0": {"max_speed": 8.0, "dt": 0.03, "g": 10.0, "m": 1.0, "l": 1.8},
        }
        env = CARLPendulumEnv(
            contexts=contexts,
            hide_context=False,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=None,
            scale_context_features="by_mean",
        )
        env.reset()
        action = [0.0]
        state, reward, terminated, truncated, info = env.step(action=action)
        n_c = len(env.default_context)
        scaled_contexts = state[-n_c:]
        target = np.array(
            [16 / 12, 0.06 / 0.045, 20 / 15, 2 / 1.5, 3.6 / 2.7, 1, 1]
        )  # for context "1"
        self.assertTrue(
            np.all(target == scaled_contexts),
            f"target {target} != actual {scaled_contexts}",
        )

    def test_context_feature_scaling_by_default(self):
        default_context = {
            "max_speed": 8.0,
            "dt": 0.05,
            "g": 10.0,
            "m": 1.0,
            "l": 1.0,
            "initial_angle_max": np.pi,
            "initial_velocity_max": 1,
        }
        contexts = {
            "0": {"max_speed": 8.0, "dt": 0.03, "g": 10.0, "m": 1.0, "l": 1.8},
        }
        env = CARLPendulumEnv(
            contexts=contexts,
            hide_context=False,
            add_gaussian_noise_to_context=False,
            gaussian_noise_std_percentage=0.01,
            state_context_features=None,
            scale_context_features="by_default",
            default_context=default_context,
        )
        env.reset()
        action = [0.0]
        state, reward, terminated, truncated, info = env.step(action=action)
        n_c = len(default_context)
        scaled_contexts = state[-n_c:]
        self.assertTrue(
            np.all(np.array([1.0, 0.6, 1, 1, 1.8, 1, 1]) == scaled_contexts)
        )

    def test_sample_contexts(self):
        from carl.context.sampling import sample_contexts

        contexts = sample_contexts(
            env_name="CARLPendulumEnv",
            context_feature_args=["l"],
            num_contexts=1,
            default_sample_std_percentage=0.0,
        )
        self.assertEqual(contexts[0]["l"], 1)


class TestContextAugmentation(unittest.TestCase):
    def test_gaussian_noise(self):
        from carl.context.augmentation import add_gaussian_noise

    def test_default_selector(self):
        from carl.context.selection import RoundRobinSelector

        contexts = self.generate_contexts()
        env = CARLPendulumEnv(contexts=contexts)

        env.reset()
        self.assertEqual(type(env.context_selector), RoundRobinSelector)
        self.assertEqual(env.context_selector.n_calls, 1)

        env.reset()
        self.assertEqual(env.context_key, "b")

    def test_roundrobin_selector_init(self):
        from carl.context.selection import RoundRobinSelector

        contexts = self.generate_contexts()
        env = CARLPendulumEnv(
            contexts=contexts, context_selector=RoundRobinSelector(contexts=contexts)
        )
        self.assertEqual(type(env.context_selector), RoundRobinSelector)

    def test_random_selector_init(self):
        from carl.context.selection import RandomSelector

        contexts = self.generate_contexts()
        env = CARLPendulumEnv(
            contexts=contexts, context_selector=RandomSelector(contexts=contexts)
        )
        self.assertEqual(type(env.context_selector), RandomSelector)

    def test_random_selectorclass_init(self):
        from carl.context.selection import RandomSelector

        contexts = self.generate_contexts()
        env = CARLPendulumEnv(contexts=contexts, context_selector=RandomSelector)
        self.assertEqual(type(env.context_selector), RandomSelector)

    def test_unknown_selector_init(self):
        with self.assertRaises(ValueError):
            contexts = self.generate_contexts()
            _ = CARLPendulumEnv(contexts=contexts, context_selector="bork")

    def test_get_context_key(self):
        contexts = self.generate_contexts()
        env = CARLPendulumEnv(contexts=contexts)
        self.assertEqual(env.context_key, None)


class TestContextSampler(unittest.TestCase):
    def test_get_defaults(self):
        from carl.context.sampling import get_default_context_and_bounds

        defaults, bounds = get_default_context_and_bounds(env_name="CARLPendulumEnv")
        DEFAULT_CONTEXT = {
            "max_speed": 8.0,
            "dt": 0.05,
            "g": 10.0,
            "m": 1.0,
            "l": 1.0,
            "initial_angle_max": np.pi,
            "initial_velocity_max": 1,
        }
        self.assertDictEqual(defaults, DEFAULT_CONTEXT)

    def test_sample_contexts(self):
        from carl.context.sampling import sample_contexts

        contexts = sample_contexts(
            env_name="CARLPendulumEnv",
            context_feature_args=["l"],
            num_contexts=1,
            default_sample_std_percentage=0.0,
        )
        self.assertEqual(contexts[0]["l"], 1)


class TestContextAugmentation(unittest.TestCase):
    def test_gaussian_noise(self):
        from carl.context.augmentation import add_gaussian_noise

        c = add_gaussian_noise(default_value=1, percentage_std=0)
        self.assertEqual(c, 1)


if __name__ == "__main__":
    unittest.main()
