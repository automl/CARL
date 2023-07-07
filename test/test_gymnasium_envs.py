import inspect
import unittest

import carl.envs.gymnasium


class TestGymnasiumEnvs(unittest.TestCase):
    def test_envs(self):
        envs = inspect.getmembers(carl.envs.gymnasium)

        for env_name, env_obj in envs:
            if inspect.isclass(env_obj) and "CARL" in env_name:
                try:
                    env_obj.get_context_features()

                    env = env_obj()
                    env._update_context()
                except Exception as e:
                    print(f"Cannot instantiate {env_name} environment.")
                    raise e


if __name__ == "__main__":
    TestGymnasiumEnvs().test_envs()
