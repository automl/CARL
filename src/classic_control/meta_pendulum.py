from src.meta_env import MetaEnv

class MetaPendulumEnv(MetaEnv):
    def __init__(self, env, contexts, instance_mode):
        super().__init__(env, contexts, instance_mode)

    def _update_context(self):
        self.env.max_speed = self.context["max_speed"]
        self.env.max_torque = self.context["max_torque"]
        self.env.dt = self.context["dt"]
        self.env.l = self.context["l"]
        self.env.m = self.context["m"]
        self.env.g = self.context["g"]
