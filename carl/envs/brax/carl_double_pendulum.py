import jax.numpy as jnp
import numpy as np
from brax.envs.inverted_double_pendulum import InvertedDoublePendulum

from carl.envs.brax.carl_brax_env import CARLBraxEnv
from carl.utils.types import Context

DEFAULT_CONTEXT = {
    "stiffness_factor": 1,
    "gravity_x": 1e-5,
    "gravity_z": -9.81,
    "friction": 0.8,
    "damping_factor": 1,
    "actuator_strength_factor": 1,
    "mass_cart": 10.5,
    "mass_pole_0": 4.2,
    "mass_pole_1": 4.2,
    "dt": 0.01,
}

CONTEXT_BOUNDS = {
    "stiffness_factor": (0, np.inf, float),
    "gravity_x": (-np.inf, np.inf, float),
    "gravity_z": (-np.inf, -0.1, float),
    "friction": (-np.inf, np.inf, float),
    "damping_factor": (-np.inf, np.inf, float),
    "actuator_strength_factor": (1, np.inf, float),
    "mass_cart": (0.1, np.inf, float),
    "mass_pole_0": (0.1, np.inf, float),
    "mass_pole_1": (0.1, np.inf, float),
    "dt": (0.0001, 0.03, float),
}


class CARLInvertedDoublePendulum(CARLBraxEnv):
    env_name: str = "inverted_double_pendulum"
    DEFAULT_CONTEXT: Context = DEFAULT_CONTEXT

    def _update_context(self) -> None:
        self.env: InvertedDoublePendulum
        config = {}
        config["gravity"] = jnp.array(
            [self.context["gravity_x"], 0, self.context["gravity_z"]]
        )
        config["dt"] = jnp.array(self.context["dt"])
        new_mass = self.env._env.sys.link.inertia.mass.at[0].set(
            self.context["mass_cart"]
        )
        new_mass = new_mass.at[1].set(self.context["mass_pole_0"])
        new_mass = new_mass.at[2].set(self.context["mass_pole_1"])
        # TODO: do we want to implement this?
        # new_com = self.env.sys.link.inertia.transform
        # new_inertia = self.env.sys.link.inertia.i
        inertia = self.env._env.sys.link.inertia.replace(mass=new_mass)
        config["link"] = self.env._env.sys.link.replace(inertia=inertia)
        new_stiffness = (
            self.context["stiffness_factor"] * self.env._env.sys.dof.stiffness
        )
        new_damping = self.context["damping_factor"] * self.env._env.sys.dof.damping
        config["dof"] = self.env._env.sys.dof.replace(
            stiffness=new_stiffness, damping=new_damping
        )
        new_gear = (
            self.context["actuator_strength_factor"] * self.env._env.sys.actuator.gear
        )
        config["actuator"] = self.env._env.sys.actuator.replace(gear=new_gear)
        geoms = self.env._env.sys.geoms
        geoms[0] = geoms[0].replace(friction=jnp.array([self.context["friction"]]))
        config["geoms"] = geoms
        self.env._env.sys = self.env._env.sys.replace(**config)
