import pytest

from carl.envs.dmc import (
    CARLDmcFingerEnv,
    CARLDmcFishEnv,
    CARLDmcPointMassEnv,
    CARLDmcQuadrupedEnv,
    CARLDmcWalkerEnv,
)
from carl.envs.dmc.dmc_tasks.finger import check_constraints
from carl.envs.dmc.dmc_tasks.finger import (
    get_model_and_assets as get_finger_model_and_assets,
)
from carl.envs.dmc.dmc_tasks.finger import (
    spin_context,
    turn_easy_context,
    turn_hard_context,
)
from carl.envs.dmc.dmc_tasks.pointmass import (
    check_constraints as check_constraints_pointmass,
)
from carl.envs.dmc.dmc_tasks.pointmass import make_model as make_pointmass_model
from carl.envs.dmc.dmc_tasks.quadruped import make_model as make_quadruped_model
from carl.envs.dmc.dmc_tasks.utils import adapt_context
from carl.envs.dmc.dmc_tasks.walker import (
    get_model_and_assets as get_walker_model_and_assets,
)
from carl.envs.dmc.loader import load_dmc_env


class TestDMCLoader:
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
        with pytest.raises(ValueError):
            _ = load_dmc_env(
                domain_name="walker",
                task_name="walk_context_blub",
            )

    def test_load_unknowndomain_dmc_env(self):
        with pytest.raises(ValueError):
            _ = load_dmc_env(
                domain_name="sdfsdf",
                task_name="walk",
            )


class TestFinger:
    def test_finger_constraints(self):
        # Finger can reach spinner?
        with pytest.raises(ValueError):
            check_constraints(
                limb_length_0=0.17,
                limb_length_1=0.16,
                spinner_length=0.1,
                raise_error=True,
            )
        # Spinner collides with finger hinge?
        with pytest.raises(ValueError):
            check_constraints(
                limb_length_0=0.17,
                limb_length_1=0.16,
                spinner_length=0.81,
                raise_error=True,
            )

    def test_finger_tasks(self):
        tasks = [spin_context, turn_hard_context, turn_easy_context]
        contexts = [{}, {"spinner_length": 0.2}]
        for context in contexts:
            for task in tasks:
                _ = task(context=context)


class TestDmcUtils:
    def get_string_and_context(self):
        xml_string, _ = get_finger_model_and_assets()
        default_context = CARLDmcFingerEnv.get_default_context()
        return xml_string, default_context

    def test_adapt_context_no_context(self):
        context = {}
        xml_string, _ = self.get_string_and_context()
        _ = adapt_context(xml_string=xml_string, context=context)

    def test_adapt_context_partialcontext(self):
        context = {"gravity": 10}
        xml_string, _ = self.get_string_and_context()
        _ = adapt_context(xml_string=xml_string, context=context)

    def test_adapt_context_fullcontext(self):
        # only continuous context features
        xml_string, context = self.get_string_and_context()
        context["gravity"] *= 1.25
        _ = adapt_context(xml_string=xml_string, context=context)

    def test_adapt_context_friction(self):
        xml_string, _ = get_walker_model_and_assets()
        context = CARLDmcWalkerEnv.get_default_context()
        context["friction_tangential"] *= 1.3
        _ = adapt_context(xml_string=xml_string, context=context)


class TestQuadruped:
    def test_make_model(self):
        _ = make_quadruped_model(floor_size=1)

    def test_instantiate_env_with_context(self):
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


class TestFish:
    def test_make_model(self):
        _ = make_quadruped_model(floor_size=1)

    def test_instantiate_env_with_context(self):
        tasks = ["swim_context", "upright_context"]
        for task in tasks:
            _ = CARLDmcFishEnv(
                contexts={
                    0: {
                        "gravity": -10,
                    }
                },
                task=task,
            )


class TestPointmass:
    def test_make_model(self):
        _ = make_pointmass_model(floor_size=1)

    def test_instantiate_env_with_context(self):
        tasks = ["easy_pointmass", "hard_pointmass"]
        for task in tasks:
            _ = CARLDmcPointMassEnv(
                contexts={
                    0: {
                        "starting_x": 0.3,
                    }
                },
                task=task,
            )

    def test_constraints(self):
        # Is starting point inside grid?
        with pytest.raises(ValueError):
            check_constraints_pointmass(
                mass=0.3,
                starting_x=0.3,
                starting_y=0.3,
                target_x=0.0,
                target_y=0.0,
                area_size=0.6,
            )
        # Is target inside grid?
        with pytest.raises(ValueError):
            check_constraints_pointmass(
                mass=0.3,
                starting_x=0.0,
                starting_y=0.0,
                target_x=0.3,
                target_y=0.3,
                area_size=0.6,
            )
