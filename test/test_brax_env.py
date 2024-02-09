import inspect
import unittest

import carl.envs.gymnasium


class TestBraxEnvs(unittest.TestCase):
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


if __name__ == "__main__":
    TestBraxEnvs().test_envs()
