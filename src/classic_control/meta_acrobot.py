import gym
import numpy as np
from typing import Optional, Dict
from gym.envs.classic_control import AcrobotEnv
from src.meta_env import MetaEnv
from src.trial_logger import TrialLogger

DEFAULT_CONTEXT = {
    "link_length_1": 1,  # should be seen as 100% default and scaled
    "link_length_2": 1,  # should be seen as 100% default and scaled
    "link_mass_1": 1,  # should be seen as 100% default and scaled
    "link_mass_2": 1,  # should be seen as 100% default and scaled
    "link_com_1": 0.5,  # Percentage of the length of link one
    "link_com_2": 0.5,  # Percentage of the length of link one
    "link_moi": 1, # should be seen as 100% default and scaled
    "max_velocity_1": 4*np.pi,
    "max_velocity_2": 9*np.pi,
}

CONTEXT_BOUNDS = {
    "link_length_1": (0.1, 10), # Links can be shrunken and grown by a factor of 10
    "link_length_2": (0.1, 10),
    "link_mass_1": (0.1, 10),  # Link mass can be shrunken and grown by a factor of 10
    "link_mass_2": (0.1, 10),
    "link_com_1": (0, 1),  # Center of mass can move from one end to the other
    "link_com_2": (0, 1),
    "link_moi": (0.1, 10), # Moments on inertia can be shrunken and grown by a factor of 10
    "max_velocity_1": (0.4*np.pi, 40*np.pi), # Velocity can vary by a factor of 10 in either direction
    "max_velocity_2": (0.9*np.pi, 90*np.pi),
}

class MetaAcrobotEnv(MetaEnv):
    def __init__(
            self,
            env: gym.Env = AcrobotEnv,
            contexts: Dict[str, Dict] = {},
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None,
    ):
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        super().__init__(
            env=env,
            contexts=contexts,
            instance_mode=instance_mode,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT.keys())  # allow to augment all values
        self._update_context()

    def _update_context(self):
        self.env.LINK_LENGTH_1 = self.context["link_length_1"]
        self.env.LINK_LENGTH_2 = self.context["link_length_2"]
        self.env.LINK_MASS_1 = self.context["link_mass_1"]
        self.env.LINK_MASS_2 = self.context["link_mass_2"]
        self.env.LINK_COM_POS_1 = self.context["link_com_1"]
        self.env.LINK_COM_POS_2 = self.context["link_com_2"]
        self.env.LINK_MOI = self.context["link_moi"]
        self.env.MAX_VEL_1 = self.context["max_velocity_1"]
        self.env.MAX_VEL_2 = self.context["max_velocity_2"]

