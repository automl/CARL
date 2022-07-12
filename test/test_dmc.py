import unittest

from carl.envs.dmc.loader import load_dmc_env


class TestDMCLoader(unittest.TestCase):
    def test_load_classic_dmc_env(self):
        _ = load_dmc_env(
            domain_name="walker",
            task_name="walk",
        )

    def test_load_context_dmc_env(self):
        _ = load_dmc_env(
            domain_name="walker",
            task_name="walk_context",
        )

    def test_load_unknowntask_dmc_env(self):
        with self.assertRaises(ValueError):
            _ = load_dmc_env(
                domain_name="walker",
                task_name="walk_context_blub",
            )

    def test_load_unknowndomain_dmc_env(self):
        with self.assertRaises(ValueError):
            _ = load_dmc_env(
                domain_name="sdfsdf",
                task_name="walk",
            )
