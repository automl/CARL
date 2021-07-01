from brax.envs import Ant
from brax.envs.ant import _SYSTEM_CONFIG
from src.envs.brax.meta_brax_env import MetaBraxEnv
from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict

DEFAULT_CONTEXT = {
    "joint_stiffness": 5000,
    "gravity": -9.8,
    "friction": 0.6,
    "angular_damping": -0.05,
    "actuator_strength": 300,
    "joint_angular_damping": 35,
    "torso_mass": 10,
}

CONTEXT_BOUNDS = {
    "joint_stiffness": (1, np.inf),
    "gravity": (0.1, np.inf),
    "friction": (-np.inf, np.inf),
    "angular_damping": (-np.inf, np.inf),
    "actuator_strength": (1, np.inf),
    "joint_angular_damping": (0, 360),
    "torso_mass": (0.1, np.inf),
}


class MetaAnt(MetaBraxEnv):
    def __init__(
            self,
            env: Ant,
            contexts,
            instance_mode="rr",
            hide_context=False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None
    ):
        self.base_config = MessageToDict(text_format.Parse(_SYSTEM_CONFIG, brax.Config()))
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
        config = self.base_config.__deepcopy__
        config["gravity"] = self.context["gravity"]
        config["friction"] = self.context["friction"]
        config["angularDamping"] = self.context["angular_damping"]
        for j in config["joints"]:
            config["joints"][j]["angularDamping"] = self.context["joint_angular_damping"]
            config["stiffness"][j]["angularDamping"] = self.context["joint_stiffness"]
        for a in config["actuators"]:
            config["actuators"][a]["strength"] = self.context["actuator_strength"]
        config["bodies"][0]["mass"] = self.context["torso_mass"]
        # This converts the dict to a JSON String, then parses it into an empty brax config
        self.env.sys = brax.System(json_format.Parse(json.dumps(config), brax.Config()))