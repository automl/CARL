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


class TestCartPoleContext(unittest.TestCase):
    @staticmethod
    def _rollout(context: dict, n_steps: int = 20):
        from carl.envs.gymnasium.classic_control import CARLCartPole

        env = CARLCartPole(contexts={0: context}, obs_context_features=[])
        obs, _ = env.reset(seed=0)
        trajectory = [obs["obs"]]
        for _ in range(n_steps):
            obs, _, terminated, truncated, _ = env.step(1)
            trajectory.append(obs["obs"])
            if terminated or truncated:
                break
        return env, trajectory

    def test_derived_quantities_follow_context(self):
        from carl.envs.gymnasium.classic_control import CARLCartPole

        context = CARLCartPole.get_context_space().get_default_context()
        context.update(masscart=3.0, masspole=0.5, length=2.0)
        env, _ = self._rollout(context)
        base = env.env.unwrapped
        self.assertAlmostEqual(base.total_mass, 3.5)
        self.assertAlmostEqual(base.polemass_length, 1.0)

    def test_masscart_changes_dynamics(self):
        from carl.envs.gymnasium.classic_control import CARLCartPole

        default = CARLCartPole.get_context_space().get_default_context()
        heavy = dict(default, masscart=10.0)
        _, light_trajectory = self._rollout(default)
        _, heavy_trajectory = self._rollout(heavy)
        n = min(len(light_trajectory), len(heavy_trajectory))
        self.assertGreater(n, 1)
        self.assertFalse(
            all(
                (light_trajectory[i] == heavy_trajectory[i]).all() for i in range(1, n)
            ),
            "masscart has no effect on the trajectory",
        )


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
