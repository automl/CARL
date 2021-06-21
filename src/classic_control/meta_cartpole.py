from src.meta_env import MetaEnv

class MetaCartPoleEnv(MetaEnv):
    def __init__(self, env, contexts, instance_mode):
        super().__init__(env, contexts, instance_mode)

    def _update_context(self):
        self.env.gravity = self.context["gravity"]
        self.env.masscart = self.context["masscart"]
        self.env.masspole = self.context["masspole"]
        self.env.length = self.context["pole_length"]
        self.env.force_mag = self.context["force_magnifier"]
        self.env.tau = self.context["update_interval"]
        self.env.kinematics_integrator = self.context["kinematics"]

