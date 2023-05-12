from __future__ import annotations

import inspect

from brax.envs import Wrapper, State
from brax.envs.env import Env

import jax
from jax import numpy as jp

from carl.utils.types import Contexts, Context
from carl.context.selection import AbstractSelector, RoundRobinSelector



class CARLBraxWrapper(Wrapper):
    def __init__(
            self, 
            env: Env,
            contexts: Contexts = {},
            state_context_features: list[str] | None = None,  # list the context features which should be added to the state
            # context_mask: list[str] | None = None,
            dict_observation: bool = False,
            context_selector: AbstractSelector | type[AbstractSelector] | None = None,
            context_selector_kwargs: dict = None,
            **kwargs
        ):
        super().__init__(env)

        self.context = None
        self.contexts = contexts
        self.state_context_features = state_context_features
        # self.context_mask = context_mask
        self.dict_observation = dict_observation

        # Context Selector
        self.context_selector: type[AbstractSelector]
        if context_selector is None:
            self.context_selector = RoundRobinSelector(contexts=contexts)  # type: ignore [assignment]
        elif isinstance(context_selector, AbstractSelector):
            self.context_selector = context_selector  # type: ignore [assignment]
        elif inspect.isclass(context_selector) and issubclass(
            context_selector, AbstractSelector
        ):
            if context_selector_kwargs is None:
                context_selector_kwargs = {}
            _context_selector_kwargs = {"contexts": contexts}
            context_selector_kwargs.update(_context_selector_kwargs)
            self.context_selector = context_selector(**context_selector_kwargs)  # type: ignore [assignment]
        else:
            raise ValueError(
                f"Context selector must be None or an AbstractSelector class or instance. "
                f"Got type {type(context_selector)}."
            )
        

    @property
    def observation_size(self) -> int:
        if self.state_context_features:
            if self.dict_observation:
                raise NotImplementedError
            else:
                obs_size = self.env.observation_size + len(self.state_context_features)

        return obs_size
    
    def _progress_instance(self) -> None:
        """
        Progress instance.

        In this case instance is a specific context.
        1. Select instance with the instance_mode. If the instance_mode is random, randomly select
        the next instance from the set of contexts. If instance_mode is rr or roundrobin, select
        the next instance.
        2. If Gaussian noise should be added to whitelisted context features, do so.

        Returns
        -------
        None

        """
        context = self.context_selector.select()  # type: ignore [call-arg]
        self.context = context
    

    def reset(self, rng: jp.ndarray) -> State:  # TODO
        self._progress_instance()
        self._update_context()
        state = self.env.reset(rng)
        state = self._maybe_add_context(state)
        return state
    
    def _update_context(self) -> None:
        """
        Update the context feature values of the environment.

        Returns
        -------
        None

        """
        raise NotImplementedError
    
    def _maybe_add_context(self, state: State):
        if self.state_context_features:
            if self.dict_observation:
                raise NotImplementedError
                obs = {
                    "state": state["obs"],
                    "context": [self.context[k] for k in self.state_context_features],
                }
                state["obs"] = obs
            else:
                state["obs"] = jp.concatenate([state["obs"]], [self.context[k] for k in self.state_context_features])
        return state


    def step(self, state: State, action: jp.ndarray) -> State:  # TODO
        state = self.env.step(state, action)
        state = self._maybe_add_context(state)
        return state
    





DEFAULT_CONTEXT = {
    "joint_stiffness": 5000,
    "gravity": -9.8,
    "friction": 0.6,
    "angular_damping": -0.05,
    "actuator_strength": 300,
    "joint_angular_damping": 35,
    "torso_mass": 10,
}

# CONTEXT_BOUNDS = {
#     "joint_stiffness": (1, np.inf, float),
#     "gravity": (-np.inf, -0.1, float),
#     "friction": (-np.inf, np.inf, float),
#     "angular_damping": (-np.inf, np.inf, float),
#     "actuator_strength": (1, np.inf, float),
#     "joint_angular_damping": (0, np.inf, float),
#     "torso_mass": (0.1, np.inf, float),
# }

from brax.envs.ant import Ant
import copy
import brax
from google.protobuf import json_format, text_format
import json

class CARLAnt(CARLBraxWrapper):
    def __init__(
        self,
        env: Ant = Ant(legacy_spring=True),  #AutoResetWrapper(EpisodeWrapper(Ant(legacy_spring=True), episode_length=1000, action_repeat=1)),
        contexts: Contexts = {},
        state_context_features: list[str] | None = None,
        dict_observation: bool = False,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
    ):
        super().__init__(
            env=env,
            contexts=contexts,
            state_context_features=state_context_features,
            dict_observation=dict_observation,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs
        )

    def _update_context(self) -> None:
        self.env: Ant
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
        self.env._env.sys = brax.System(
            json_format.Parse(json.dumps(config), brax.Config())
        )
        

if __name__ == "__main__":
    from brax.envs.wrappers import EpisodeWrapper
    from brax.envs.ant import Ant
    from rich import print as printr

    print(brax.__version__)

    seed = 0

    rng=jax.random.PRNGKey(seed=seed)

    contexts = {
        0: DEFAULT_CONTEXT,
        1: DEFAULT_CONTEXT
    }
    state_context_features = list(DEFAULT_CONTEXT.keys())[:2]

    env = Ant()
    env = EpisodeWrapper(env=env, episode_length=1000, action_repeat=1)
    env = CARLAnt(
        env=env,
        contexts=contexts,
        state_context_features=state_context_features
    )

    printr(env.env.sys)
    state = env.reset(rng=rng)

    printr(state)
