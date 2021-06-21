from src.meta_env import MetaEnv

class MetaCartPoleEnv(MetaEnv):
    def __init__(self, env, contexts, instance_mode):
        super().__init__(env, contexts, instance_mode)

    def _update_context(self):
        self.env.LINK_LENGTH_1 = self.context["link_length_1"]
        self.env.LINK_LENGTH_2 = self.context["link_length_2"]
        self.env.LINK_MASS_1 = self.context["link_mass_1"]
        self.env.LINK_MASS_2 = self.context["link_mass_2"]
        self.env.LINK_COM_POS_1 = self.context["link_com_1"]
        self.env.LINK_COM_POS_2 = self.context["link_com_2"]
        self.env.MAX_VEL_1 = self.context["max_velocity_1"]
        self.env.MAX_VEL_2 = self.context["max_velocity_2"]
        self.env.book_or_nips = self.context["dynamics"]

