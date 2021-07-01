from src.meta_env import MetaEnv

import brax
from brax.envs import Env as BraxEnv
from brax.envs import GymWrapper


class MetaBraxEnv(MetaEnv):
    def __init__(
            self,
            env: BraxEnv,
            contexts,
            instance_mode="rr",
            hide_context=False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None
    ):
        self.env = GymWrapper(env)
        super().__init__(
            env=env,
            contexts=contexts,
            instance_mode=instance_mode,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger
        )

    def _update_context(self):
        config = self.base_config.MergeFrom(parsedcontext)
        self.env.sys = brax.System(config)
