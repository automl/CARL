from src.envs.meta_env import MetaEnv
from src.envs.rna.learna.src.data import parse_dot_brackets
from src.envs.rna.learna.src.learna.environment import RnaDesignEnvironment, RnaDesignEnvironmentConfig
import numpy as np
from typing import Optional, Dict, List
from src.trial_logger import TrialLogger

DEFAULT_CONTEXT = {
    "mutation_threshold": 5,
    "reward_exponent": 1,
    "state_radius": 5,
    "dataset": "eterna",
    "target_structure_ids": None
}

CONTEXT_BOUNDS = {
    "mutation_threshold": (0.1, np.inf, int),
    "reward_exponent": (0.1, np.inf, int),
    "state_radius": (1, np.inf, int),
    "dataset": ("eterna", "rfam_taneda", None),
    "target_structure_ids": (0, np.inf, List[int]) #This is conditional on the dataset (and also a list)
}

class MetaRnaDesignEnvironment(MetaEnv):
    def __init__(
            self,
            #TODO: add exception to runscript once everything else works
            # This actually needs to be initialized beforehand
            env: RnaDesignEnvironment,
            data_location: str,
            contexts: Dict[str, Dict] = {},
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None,
    ):
        """

        Parameters
        ----------
        env: gym.Env, optional
            Defaults to classic control environment mountain car from gym (MountainCarEnv).
        contexts: List[Dict], optional
            Different contexts / different environment parameter settings.
        instance_mode: str, optional
        """
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
        self.data_location = data_location

    def step(self, action):
        # Step function has a different name in this env
        state, reward, done = self.env.execute(action)
        if not self.hide_context:
            state = np.concatenate((state, np.array(list(self.context.values()))))
        self.step_counter += 1
        return state, reward, done, {}

    def _update_context(self):
        dot_brackets = parse_dot_brackets(
            dataset=self.context["dataset"],
            data_dir=self.data_location,
            target_structure_ids=self.context["target_structure_ids"],)
        env_config = RnaDesignEnvironmentConfig(
            mutation_threshold=self.context["mutation_threshold"],
            reward_exponent=self.context["reward_exponent"],
            state_radius=self.context["state_radius"],
        )
        self.env = RnaDesignEnvironment(dot_brackets, env_config)
