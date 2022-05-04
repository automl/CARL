from typing import Any, Dict, List, Optional, Union

import copy
import json

import brax
import numpy as np
from brax.envs.ur5e import _SYSTEM_CONFIG, Ur5e
from brax.envs.wrappers import GymWrapper, VectorWrapper, VectorGymWrapper
from google.protobuf import json_format, text_format
from google.protobuf.json_format import MessageToDict
from numpyencoder import NumpyEncoder

from carl.envs.carl_env import CARLEnv
from carl.utils.trial_logger import TrialLogger
from carl.context.selection import AbstractSelector

DEFAULT_CONTEXT = {
    "joint_stiffness": 40000,
    "gravity": -9.81,
    "friction": 0.6,
    "angular_damping": -0.05,
    "actuator_strength": 100,
    "joint_angular_damping": 50,
    "target_radius": 0.02,
    "target_distance": 0.5,
    "torso_mass": 1.0,
}

CONTEXT_BOUNDS = {
    "joint_stiffness": (1, np.inf, float),
    "gravity": (-np.inf, -0.1, float),
    "friction": (-np.inf, np.inf, float),
    "angular_damping": (-np.inf, np.inf, float),
    "actuator_strength": (1, np.inf, float),
    "joint_angular_damping": (0, 360, float),
    "target_radius": (0.01, np.inf, float),
    "target_distance": (0.01, np.inf, float),
    "torso_mass": (0, np.inf, float),
}


class CARLUr5e(CARLEnv):
    def __init__(
            self,
            env: Ur5e = Ur5e(),
            n_envs: int = 1,
            contexts: Dict[str, Dict] = {},
            hide_context=False,
            add_gaussian_noise_to_context: bool = False,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None,
            scale_context_features: str = "no",
            default_context: Optional[Dict] = DEFAULT_CONTEXT,
            state_context_features: Optional[List[str]] = None,
            dict_observation_space: bool = False,
            context_selector: Optional[Union[AbstractSelector, type(AbstractSelector)]] = None,
            context_selector_kwargs: Optional[Dict] = None,
    ):
        if n_envs == 1:
            env = GymWrapper(env)
        else:
            env = VectorGymWrapper(VectorWrapper(env, n_envs))

        self.base_config = MessageToDict(
            text_format.Parse(_SYSTEM_CONFIG, brax.Config())
        )
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        super().__init__(
            env=env,
            n_envs=n_envs,
            contexts=contexts,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features=scale_context_features,
            default_context=default_context,
            state_context_features=state_context_features,
            dict_observation_space=dict_observation_space,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
        )
        self.whitelist_gaussian_noise = list(
            DEFAULT_CONTEXT.keys()
        )  # allow to augment all values

    def _update_context(self) -> None:
        config = copy.deepcopy(self.base_config)
        config["gravity"] = {"z": self.context["gravity"]}
        config["friction"] = self.context["friction"]
        config["angularDamping"] = self.context["angular_damping"]
        for j in range(len(config["joints"])):
            config["joints"][j]["angularDamping"] = self.context[
                "joint_angular_damping"
            ]
            config["joints"][j]["stiffness"] = self.context["joint_stiffness"]
        for a in range(len(config["actuators"])):
            config["actuators"][a]["strength"] = self.context["actuator_strength"]
        config["bodies"][0]["mass"] = self.context["torso_mass"]
        # This converts the dict to a JSON String, then parses it into an empty brax config
        self.env.sys = brax.System(
            json_format.Parse(json.dumps(config, cls=NumpyEncoder), brax.Config())
        )
        self.env.target_idx = self.env.sys.body.index["Target"]
        self.env.torso_idx = self.env.sys.body.index["wrist_3_link"]
        self.env.target_radius = self.context["target_radius"]
        self.env.target_distance = self.context["target_distance"]

    def __getattr__(self, name: str) -> Any:
        if name in [
            "sys",
            "target_idx",
            "torso_idx",
            "target_radius",
            "target_distance",
        ]:
            return getattr(self.env._environment, name)
        else:
            return getattr(self, name)
