from src.meta_env import MetaEnv

DEFAULT_CONTEXT = {
    "gravity": 9.8,
    "masscart": 1.,
    "masspole":  0.1,
    "pole_length": 0.5,
    "force_magnifier": 10.,
    "update_interval": 0.02,
    "kinematics": 'euler'
}

class MetaCartPoleEnv(MetaEnv):
    def __init__(self,
            env: gym.Env = gccenvs.pendulum.PendulumEnv(),
            contexts: Dict[int, Dict] = {},  # ??? what should be the type of the dict keys?
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01):
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        super().__init__(
            env=env,
            contexts=contexts,
            instance_mode=instance_mode,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT.keys())  # allow to augment all values

    def _update_context(self):
        self.env.gravity = self.context["gravity"]
        self.env.masscart = self.context["masscart"]
        self.env.masspole = self.context["masspole"]
        self.env.length = self.context["pole_length"]
        self.env.force_mag = self.context["force_magnifier"]
        self.env.tau = self.context["update_interval"]
        self.env.kinematics_integrator = self.context["kinematics"]

