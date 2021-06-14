from src.meta_env import MetaEnv

class CartPoleEnv(MetaEnv):
    def __init__(self, ...):

    def _update_context(self):
        self.gravity = self.context["gravity"]

