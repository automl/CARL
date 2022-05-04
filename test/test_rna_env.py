import unittest

import numpy as np

from carl.envs import CARLRnaDesignEnv
from carl.envs.rna.carl_rna import RnaGymWrapper


class TestRNA(unittest.TestCase):
    def test_rna_init(self):
        env = CARLRnaDesignEnv()
        self.assertFalse(env is None)
        self.assertTrue(isinstance(CARLEnv, RnaGymWrapper))
        self.assertFalse(env.data_location is None)

        self.assertTrue("mutation_threshold" in env.context[0].keys())
        self.assertTrue("reward_exponent" in env.context[0].keys())
        self.assertTrue("state_radius" in env.context[0].keys())
        self.assertTrue("dataset" in env.context[0].keys())
        self.assertTrue("target_structure_ids" in env.context[0].keys())

    def test_update_context(self):
        env = CARLRnaDesignEnv()
        self.assertTrue(env.context("mutation_threshold") == env.mutation_threshold)
        self.assertTrue(env.context("reward_exponent") == env.reward_exponent)
        self.assertTrue(env.context("state_radius") == env.state_radius)

        env.context = {"mutation_threshold": 3, "reward_exponent": 2, "state_radius": 1, "dataset": env.context["dataset"], "target_structure_ids": env.context["target_structure_ids"]}
        self.assertTrue(env.context("mutation_threshold") == 3)
        self.assertTrue(env.context("reward_exponent") == 2)
        self.assertTrue(env.context("state_radius") == 1)