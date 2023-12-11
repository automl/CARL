import inspect
import unittest

import gymnasium as gym

import carl
import carl.envs.gymnasium


class TestGymnasiumEnvs(unittest.TestCase):
    def test_envs(self):
        envs = inspect.getmembers(carl.envs.gymnasium)

        for env_name, env_obj in envs:
            if inspect.isclass(env_obj) and "CARL" in env_name:
                try:
                    env_obj.get_context_features()
                    env = env_obj()
                    env._progress_instance()
                    env._update_context()
                except Exception as e:
                    print(f"Cannot instantiate {env_name} environment.")
                    raise e


class TestGymnasiumRegistration(unittest.TestCase):
    def test_registration(self):
        registered_envs = gym.envs.registration.registry.keys()
        for e in carl.envs.__all__:
            if "RNA" not in e and "Brax" not in e:
                env_name = f"carl/{e}-v0"
                self.assertTrue(env_name in registered_envs)

    def test_make(self):
        for e in carl.envs.__all__:
            if "RNA" not in e and "Brax" not in e:
                env_name = f"carl/{e}-v0"
                env = gym.make(env_name)
                self.assertTrue(isinstance(env, gym.Env))


if __name__ == "__main__":
    unittest.main()
