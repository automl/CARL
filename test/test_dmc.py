import unittest

from carl.envs.dmc.loader import load_dmc_env
from carl.envs.dmc.dmc_tasks.finger import check_constraints, spin_context, turn_easy_context, turn_hard_context


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


class TestDmcEnvs(unittest.TestCase):
    def test_finger_constraints(self):
        # Finger can reach spinner?
        with self.assertRaises(ValueError):
            check_constraints(limb_length_0=0.17, limb_length_1=0.16, spinner_length=0.1)
        # Spinner collides with finger hinge?
        with self.assertRaises(ValueError):
            check_constraints(limb_length_0=0.17, limb_length_1=0.16, spinner_length=0.81)

    def test_finger_tasks(self):
        tasks = [spin_context, turn_hard_context, turn_easy_context]
        contexts = [{}, {"spinner_length": 0.2}]
        for context in contexts:
            for task in tasks:
                _ = task(context=context)
