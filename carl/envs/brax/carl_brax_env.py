from __future__ import annotations

from typing import Any

from dataclasses import asdict

import brax
import gymnasium
import numpy as np
from brax.base import Geometry
from brax.envs.wrappers.gym import GymWrapper, VectorGymWrapper
from brax.io import mjcf
from etils import epath
from gymnasium.wrappers.compatibility import EnvCompatibility
from jax import numpy as jp

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.types import Contexts


def set_geom_attr(
    geom: Geometry, data: dict[str, Any], context: dict[str, Any], key: str
) -> dict:
    """Set Geometry attribute

    Check whether the desired attribute is present both in the geometry and in the context.

    Parameters
    ----------
    geom : Geometry
        Brax geometry (a surface or spatial volume with a shape and material properties)
    data : dict[str, Any]
        Data from the Geometry dataclass, potentially already modified.
    context : dict[str, Any]
        The context to set.
    key : str
        The context feature to update.

    Returns
    -------
    dict
        Modified data from the geometry dataclas.
    """
    if key in context and key in data:
        value = getattr(geom, key)
        n_items = len(value)
        vec = jp.array([context[key]] * n_items)
        data[key] = vec
    return data


class CARLBraxEnv(CARLEnv):
    env_name: str
    backend: str = "spring"

    def __init__(
        self,
        env: brax.envs.env.Env | None = None,
        batch_size: int = 1,
        contexts: Contexts | None = None,
        obs_context_features: list[str]
        | None = None,  # list the context features which should be added to the state # TODO rename to obs_context_features?
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        **kwargs,
    ) -> None:
        """
        CARL Gymnasium Environment.

        Parameters
        ----------

        env : brax.envs.env.Env | None
            Brax environment, the default is None.
            If None, instantiate the env with brax' make function and
            `self.env_name` which is defined in each child class.
        batch_size : int
            Number of environments to batch together, by default 1.
        contexts : Contexts | None, optional
            Context set, by default None. If it is None, we build the
            context set with the default context.
        obs_context_features : list[str] | None, optional
            Context features which should be included in the observation, by default None.
            If they are None, add all context features.
        context_selector: AbstractSelector | type[AbstractSelector] | None, optional
            The context selector (class), after each reset selects a new context to use.
             If None, use a round robin selector.
        context_selector_kwargs : dict, optional
            Optional keyword arguments for the context selector, by default None.
            Only used when `context_selector` is not None.

        Attributes
        ----------
        env_name: str
            The registered gymnasium environment name.
        backend: str

        """
        if env is None:
            env = brax.envs.create(
                env_name=self.env_name, backend=self.backend, batch_size=batch_size
            )  # TODO arguments
            # Brax uses gym instead of gymnasium
            if batch_size == 1:
                env = GymWrapper(env)  # TODO do we need vector env?
            else:
                env = VectorGymWrapper(env)
            env = EnvCompatibility(env)
            # The observation space also needs to from gymnasium
            env.observation_space = gymnasium.spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                dtype=np.float32,
            )

        super().__init__(
            env=env,
            contexts=contexts,
            obs_context_features=obs_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )

    def _update_context(self) -> None:
        path = epath.resource_path("brax") / self.asset_path
        sys = mjcf.load(path)

        context = self.context

        sys = sys.replace(gravity=jp.array([0, 0, self.context["gravity"]]))
        sys = sys.replace(ang_damping=self.context["ang_damping"])
        sys.link.inertia.mass.at[sys.link_names.index("torso")].set(
            self.context["torso_mass"]
        )

        updated_geoms = []
        for i, geom in enumerate(sys.geoms):
            cls = type(geom)
            data = asdict(geom)
            data = set_geom_attr(geom, data, context, "friction")
            data = set_geom_attr(geom, data, context, "elasticity")

            geom_new = cls(**data)
            updated_geoms.append(geom_new)
        sys = sys.replace(geoms=updated_geoms)

        self.env.sys = sys

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            # "stiffness": UniformFloatContextFeature("stiffness", lower=1, upper=100000, default_value=5000),
            "gravity": UniformFloatContextFeature(
                "gravity", lower=-1000, upper=-0.01, default_value=-9.8
            ),
            "friction": UniformFloatContextFeature(
                "friction", lower=0, upper=100, default_value=0.6
            ),
            "elasticity": UniformFloatContextFeature(
                "elasticity", lower=0, upper=100, default_value=0.6
            ),  # TODO Check elasticity
            "ang_damping": UniformFloatContextFeature(
                "ang_damping", lower=-np.inf, upper=np.inf, default_value=-0.05
            ),
            # "actuator_strength": UniformFloatContextFeature("actuator_strength", lower=1, upper=100000, default_value=300),
            # "joint_angular_damping": UniformFloatContextFeature("joint_angular_damping", lower=0, upper=10000, default_value=35),
            "torso_mass": UniformFloatContextFeature(
                "torso_mass", lower=0.01, upper=np.inf, default_value=10
            ),
        }
