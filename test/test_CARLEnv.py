import unittest

from carl.envs.gymnasium.classic_control.carl_pendulum import CARLPendulum

CARLPendulum.render_mode = "rgb_array"


class TestStateConstruction(unittest.TestCase):
    def test_observation(self):
        env = CARLPendulum()
        context = CARLPendulum.get_default_context()
        obs, info = env.reset()
        self.assertEqual(type(obs), dict)
        self.assertTrue("state" in obs)
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
