from __future__ import annotations

from typing import Any

from dataclasses import asdict

import brax
import gymnasium
import numpy as np
from brax.base import Geometry, Inertia, Link, System
from brax.io import mjcf
from etils import epath
from jax import numpy as jp

from carl.context.selection import AbstractSelector
from carl.envs.brax.wrappers import GymWrapper, VectorGymWrapper
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


def set_masses2(sys: System, context: dict[str, Any]) -> System:
    for cfname, cfvalue in context.items():
        if cfname.startswith("mass"):
            link_name = cfname.split("_")[-1]
            if link_name in sys.link_names:
                idx = sys.link_names.index(link_name)
                sys.link.inertia.mass.at[idx].set(cfvalue)
    return sys


def _set_masses(
    context: dict[str, Any], inertia: Inertia, link_names: list[str]
) -> Inertia:
    """Actual/helper method to set masses

    The required syntax for masses is as follows:
    `mass_<linkname>` where linkname is the name of the entity to update, e.g. torso.

    Parameters
    ----------
    context : dict[str, Any]
        Context to set
    inertia : Inertia
        The inertia dataclass.
    link_names : list[str]
        Available link names.

    Raises
    ------
    RuntimeError
        When link name not in available names

    Returns
    -------
    Inertia
        Update inertia dataclass.
    """
    inertia_data = asdict(inertia)
    for cfname, cfvalue in context.items():
        if cfname.startswith("mass"):
            link_name = cfname.split("_", 1)[-1]
            if link_name in link_names:
                idx = link_names.index(link_name)
                inertia_data["mass"] = inertia_data["mass"].at[idx].set(cfvalue)
            else:
                raise RuntimeError(
                    f"Link {link_name} not in available link names {link_names}. Probably "
                    "something went wrong during context creation."
                )
    inertia_new = Inertia(**inertia_data)
    return inertia_new


def set_masses(sys: System, context: dict[str, Any]) -> System:
    """Set masses

    The required syntax for masses is as follows:
    `mass_<linkname>` where linkname is the name of the entity to update, e.g. torso.

    Parameters
    ----------
    sys : System
        The brax system definition.
    context : dict[str, Any]
        Context to set.

    Returns
    -------
    System
        The updated system.
    """
    link_data = asdict(sys.link)
    inertia_new = _set_masses(context, sys.link.inertia, sys.link_names)
    link_data["inertia"] = inertia_new
    link_new = Link(**link_data)
    sys = sys.replace(link=link_new)
    return sys


def check_context(
    context: dict[str, Any], registered_context_features: list[str]
) -> None:
    for cfname in context.keys():
        if cfname not in registered_context_features and not cfname.startswith("mass_"):
            raise RuntimeError(
                f"Context feature {cfname} can not be updated in the brax system. Only "
                f"{registered_context_features} are possible."
            )


class CARLBraxEnv(CARLEnv):
    env_name: str
    backend: str = "spring"

    def __init__(
        self,
        env: brax.envs.env.Env | None = None,
        batch_size: int = 1,
        contexts: Contexts | None = None,
        obs_context_features: list[str] | None = None,
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
            bs = batch_size if batch_size != 1 else None
            env = brax.envs.create(
                env_name=self.env_name, backend=self.backend, batch_size=bs
            )
            # Brax uses gym instead of gymnasium
            if batch_size == 1:
                env = GymWrapper(env)
            else:
                env = VectorGymWrapper(env)

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
        context = self.context

        # Those context features can be updated + every feature starting with `mass_`
        registered_cfs = [
            "friction",
            "ang_damping",
            "gravity",
            "viscosity",
            "elasticity",
        ]
        check_context(context, registered_cfs)

        path = epath.resource_path("brax") / self.asset_path
        sys = mjcf.load(path)

        if "gravity" in context:
            sys = sys.replace(gravity=jp.array([0, 0, self.context["gravity"]]))
        if "ang_damping" in context:
            sys = sys.replace(ang_damping=self.context["ang_damping"])
        if "viscosity" in context:
            sys = sys.replace(ang_damping=self.context["viscosity"])

        sys = set_masses(sys, context)

        if "friction" in context or "elasticity" in context:
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
