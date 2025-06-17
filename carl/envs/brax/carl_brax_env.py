from __future__ import annotations

from typing import Any

from dataclasses import asdict

import brax
import gymnasium
import numpy as np
from brax.base import Inertia, Link, System
from brax.io import mjcf
from etils import epath
from jax import numpy as jp

from carl.context.selection import AbstractSelector
from carl.envs.brax.brax_walker_goal_wrapper import (
    BraxLanguageWrapper,
    BraxWalkerGoalWrapper,
)
from carl.envs.brax.wrappers import GymWrapper, VectorGymWrapper
from carl.envs.carl_env import CARLEnv
from carl.utils.types import Context, Contexts


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
        use_language_goals: bool = False,
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
            brax_env_instance = brax.envs.create(
                env_name=self.env_name, backend=self.backend, batch_size=bs
            )

            # import pdb; pdb.set_trace()

            # Create Gymnasium-compatible wrapper
            if batch_size == 1:
                env = GymWrapper(brax_env_instance)

            else:
                env = VectorGymWrapper(brax_env_instance)

            # import pdb; pdb.set_trace()

            # Convert to Gymnasium spaces
            env.observation_space = gymnasium.spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                dtype=np.float32,
            )
            env.action_space = gymnasium.spaces.Box(
                low=env.action_space.low,
                high=env.action_space.high,
                dtype=np.float32,
            )

        # Store the original Brax environment for context updates
        self._brax_env = env.unwrapped if hasattr(env, "unwrapped") else env

        if contexts is not None:
            if (
                "target_distance" in contexts[list(contexts.keys())[0]].keys()
                or "target_direction" in contexts[list(contexts.keys())[0]].keys()
            ):
                assert all(
                    [
                        "target_direction" in contexts[list(contexts.keys())[i]].keys()
                        for i in range(len(contexts))
                    ]
                ), "All contexts must have a 'target_direction' key"
                assert all(
                    [
                        "target_distance" in contexts[list(contexts.keys())[i]].keys()
                        for i in range(len(contexts))
                    ]
                ), "All contexts must have a 'target_distance' key"
                base_dir = contexts[list(contexts.keys())[0]]["target_direction"]
                base_dist = contexts[list(contexts.keys())[0]]["target_distance"]
                max_diff_dir = max(
                    [c["target_direction"] - base_dir for c in contexts.values()]
                )
                max_diff_dist = max(
                    [c["target_distance"] - base_dist for c in contexts.values()]
                )
                if max_diff_dir > 0.1 or max_diff_dist > 0.1:
                    env = BraxWalkerGoalWrapper(env, self.env_name, self.asset_path)
                    if use_language_goals:
                        env = BraxLanguageWrapper(env)

        self.use_language_goals = use_language_goals

        super().__init__(
            env=env,
            contexts=contexts,
            obs_context_features=obs_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )
        self.env.context = self.context

    def _get_unwrapped_env(self, env):
        """Return the base environment without any wrappers."""
        return env.unwrapped if hasattr(env, "unwrapped") else env

    @property
    def env(self) -> gymnasium.Env:
        """Return the environment as a Gymnasium environment."""
        return self._env

    @env.setter
    def env(self, env: Any) -> None:
        """Set the environment, ensuring Gymnasium compatibility."""
        self._env = env
        # Store the original Brax environment if needed
        if hasattr(env, "unwrapped"):
            self._brax_env = self._get_unwrapped_env(env)

    def _update_context(self) -> None:
        context = self.context

        # Those context features can be updated + every feature starting with `mass_`
        registered_cfs = [
            "friction",
            "ang_damping",
            "gravity",
            "viscosity",
            "elasticity",
            "target_distance",
            "target_direction",
            "target_radius",
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

        if "friction" in context:
            sys = sys.replace(
                geom_friction=sys.geom_friction.at[:, 0].set(context["friction"])
            )
        if "elasticity" in context:
            sys = sys.replace(
                elasticity=sys.elasticity.at[:].set(context["elasticity"])
            )

        self.env.unwrapped.sys = sys

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Overwrites reset in super to update context in wrapper."""
        last_context_id = self.context_id
        self._progress_instance()
        if self.context_id != last_context_id:
            self._update_context()
        self.env.context = self.context
        state, info = self.env.reset(seed=seed, options=options)
        state = self._add_context_to_state(state)
        info["context_id"] = self.context_id
        return state, info

    @classmethod
    def get_default_context(cls) -> Context:
        """Get the default context (without any goal features)

        Returns
        -------
        Context
            Default context.
        """
        default_context = cls.get_context_space().get_default_context()
        if "target_distance" in default_context:
            del default_context["target_distance"]
        if "target_direction" in default_context:
            del default_context["target_direction"]
        if "target_radius" in default_context:
            del default_context["target_radius"]
        return default_context

    @classmethod
    def get_default_goal_context(cls) -> Context:
        """Get the default context (with goal features)

        Returns
        -------
        Context
            Default context.
        """
        default_context = cls.get_context_space().get_default_context()
        return default_context
