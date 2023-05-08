import unittest

from carl.envs.dmc.dmc_tasks.finger import (
    check_constraints,
    spin_context,
    turn_easy_context,
    turn_hard_context,
)
from carl.envs.dmc.dmc_tasks.utils import adapt_context
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


class TestDmcEnvs(unittest.TestCase):
    def test_finger_constraints(self):
        # Finger can reach spinner?
        with self.assertRaises(ValueError):
            check_constraints(
                limb_length_0=0.17, limb_length_1=0.16, spinner_length=0.1
            )
        # Spinner collides with finger hinge?
        with self.assertRaises(ValueError):
            check_constraints(
                limb_length_0=0.17, limb_length_1=0.16, spinner_length=0.81
            )

    def test_finger_tasks(self):
        tasks = [spin_context, turn_hard_context, turn_easy_context]
        contexts = [{}, {"spinner_length": 0.2}]
        for context in contexts:
            for task in tasks:
                _ = task(context=context)


class TestDmcUtils(unittest.TestCase):
    def setUp(self) -> None:
        from carl.envs.dmc.carl_dm_finger import DEFAULT_CONTEXT
        from carl.envs.dmc.dmc_tasks.finger import get_model_and_assets

        self.xml_string, _ = get_model_and_assets()
        self.default_context = DEFAULT_CONTEXT

    def test_adapt_context_no_context(self):
        context = {}
        _ = adapt_context(xml_string=self.xml_string, context=context)

    def test_adapt_context_partialcontext(self):
        context = {"gravity": 10}
        _ = adapt_context(xml_string=self.xml_string, context=context)

    def test_adapt_context_fullcontext(self):
        # only continuous context features
        context = self.default_context
        context["gravity"] *= 1.25
        _ = adapt_context(xml_string=self.xml_string, context=context)

    def test_adapt_context_contextmask(self):
        # only continuous context features
        context = self.default_context
        context_mask = list(context.keys())
        _ = adapt_context(
            xml_string=self.xml_string, context=context, context_mask=context_mask
        )

    def test_adapt_context_wind(self):
        context = {"wind": 10}
        with self.assertRaises(KeyError):
            _ = adapt_context(xml_string=self.xml_string, context=context)

    def test_adapt_context_friction(self):
        from carl.envs.dmc.carl_dm_walker import DEFAULT_CONTEXT
        from carl.envs.dmc.dmc_tasks.walker import get_model_and_assets

        xml_string, _ = get_model_and_assets()
        context = DEFAULT_CONTEXT
        context["friction_tangential"] *= 1.3
        _ = adapt_context(xml_string=xml_string, context=context)


class TestQuadruped(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_make_model(self):
        from carl.envs.dmc.dmc_tasks.quadruped import make_model

        _ = make_model(floor_size=1)

    def test_instantiate_env_with_context(self):
        from carl.envs.dmc.carl_dm_quadruped import CARLDmcQuadrupedEnv

        tasks = ["escape_context", "run_context", "walk_context", "fetch_context"]
        for task in tasks:
            _ = CARLDmcQuadrupedEnv(
                contexts={
                    0: {
                        "gravity": -10,
                    }
                },
                task=task,
            )
