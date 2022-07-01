import unittest

from carl.envs.rna.carl_rna import CARLRnaDesignEnv, RnaGymWrapper


class TestRNA(unittest.TestCase):
    def test_rna_init(self):
        env = CARLRnaDesignEnv()
        self.assertFalse(env is None)
        self.assertTrue(isinstance(env, CARLRnaDesignEnv))
        self.assertTrue(isinstance(env.env, RnaGymWrapper))
        self.assertFalse(env.data_location is None)

        self.assertTrue("mutation_threshold" in env.context.keys())
        self.assertTrue("reward_exponent" in env.context.keys())
        self.assertTrue("state_radius" in env.context.keys())
        self.assertTrue("dataset" in env.context.keys())
        self.assertTrue("target_structure_ids" in env.context.keys())

    def test_update_context(self):
        env = CARLRnaDesignEnv()
        self.assertTrue(
            env.context["mutation_threshold"] == env.env._env_config.mutation_threshold
        )
        self.assertTrue(
            env.context["reward_exponent"] == env.env._env_config.reward_exponent
        )
        self.assertTrue(env.context["state_radius"] == env.env._env_config.state_radius)

        env.context = {
            "mutation_threshold": 3,
            "reward_exponent": 2,
            "state_radius": 1,
            "dataset": env.context["dataset"],
            "target_structure_ids": env.context["target_structure_ids"],
        }
        env._update_context()
        self.assertTrue(env.env._env_config.mutation_threshold == 3)
        self.assertTrue(env.env._env_config.reward_exponent == 2)
        self.assertTrue(env.env._env_config.state_radius == 1)
