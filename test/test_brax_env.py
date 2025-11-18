import inspect

import carl
from carl.envs.brax import CARLBraxHalfcheetah


class TestBraxEnvs:
    def test_envs(self):
        envs = inspect.getmembers(carl.envs.brax)

        for env_name, env_obj in envs:
            if inspect.isclass(env_obj) and "CARL" in env_name:
                try:
                    env_obj.get_context_features()

                    env = env_obj()
                    env._progress_instance()
                    env._update_context()
                    env.reset()

                except Exception as e:
                    print(f"Cannot instantiate {env_name} environment.")
                    raise e

    def test_context_propagation(self):
        contexts = {
            0: {"mass_torso": 20.0, "gravity": 5},
            1: {"mass_torso": 30.0, "gravity": 15},
        }
        env = CARLBraxHalfcheetah(contexts=contexts)
        env.reset()
        torso_idx = env.env.unwrapped._env.sys.link_names.index("torso")

        current_context = env.contexts[env.context_id]
        assert env.env.unwrapped._env.sys.gravity[-1] == current_context["gravity"], (
            "Gravity not set correctly in env."
        )
        assert (
            env.env.unwrapped._env.sys.link.inertia.mass[torso_idx]
            == current_context["mass_torso"]
        ), "Mass not set correctly in env."

        env.reset()
        current_context = env.contexts[env.context_id]
        assert env.env.unwrapped._env.sys.gravity[-1] == current_context["gravity"], (
            "Gravity does not change upon reset."
        )
        assert (
            env.env.unwrapped._env.sys.link.inertia.mass[torso_idx]
            == current_context["mass_torso"]
        ), "Mass does not change upon reset."
