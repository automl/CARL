import importlib.util as iutil
import inspect
import unittest

import carl.envs.gymnasium


class TestBox2DEnvs(unittest.TestCase):
    def test_envs(self):
        spec = iutil.find_spec("Box2D")
        found = spec is not None
        if found:
            envs = inspect.getmembers(carl.envs.gymnasium.box2d)

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
        else:
            print("Box2D not found, skipping tests.")


if __name__ == "__main__":
    spec = iutil.find_spec("Box2D")
    found = spec is not None
    if found:
        TestBox2DEnvs().test_envs()
    else:
        print("Box2D not found, skipping tests.")
