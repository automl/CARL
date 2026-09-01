import importlib.util as iutil
import inspect
import unittest

import carl.envs.gymnasium

BOX2D_AVAILABLE = iutil.find_spec("Box2D") is not None


class TestBox2DEnvs(unittest.TestCase):
    def test_envs(self):
        if BOX2D_AVAILABLE:
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


class TestLunarLanderContext(unittest.TestCase):
    @unittest.skipIf(not BOX2D_AVAILABLE, "Box2D not found")
    def test_gravity_survives_reset(self):
        # gymnasium's LunarLander re-creates its Box2D world in reset(); the context gravity must still apply.
        from carl.envs.gymnasium.box2d import CARLLunarLander

        context = CARLLunarLander.get_context_space().get_default_context()
        context.update(GRAVITY_X=1.5, GRAVITY_Y=-3.0)
        env = CARLLunarLander(contexts={0: context}, obs_context_features=[])
        env.reset(seed=0)
        self.assertEqual(tuple(env.env.unwrapped.world.gravity), (1.5, -3.0))
        env.step(0)
        self.assertEqual(tuple(env.env.unwrapped.world.gravity), (1.5, -3.0))


if __name__ == "__main__":
    unittest.main()
