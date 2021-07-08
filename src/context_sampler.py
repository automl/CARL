import numpy as np
import typing
from typing import List
from scipy.stats import norm

from src import envs


def get_default_context_and_bounds(env_name: str):
    # TODO make less hacky / make explicit
    env_defaults = getattr(envs, f"{env_name}_defaults")
    env_bounds = getattr(envs, f"{env_name}_bounds")

    return env_defaults, env_bounds


def sample_contexts(env_name: str, unknown_args: List[str], num_contexts: int, default_sample_std: float = 0.05):
    env_defaults, env_bounds = get_default_context_and_bounds(env_name=env_name)

    sample_dists = {}
    for key in env_defaults.keys():
        if key in unknown_args:
            if f"{key}_mean" in unknown_args:
                sample_mean = float(unknown_args[unknown_args.index(f"{key}_mean")+1])
            else:
                sample_mean = env_defaults[key]

            if f"{key}_std" in unknown_args:
                sample_std = float(unknown_args[unknown_args.index(f"{key}_std")+1])
            else:
                sample_std = default_sample_std

            sample_dists[key] = (norm(loc=sample_mean, scale=sample_std), env_bounds[key][2])

    contexts = {}
    for i in range(0, num_contexts):
        c = {}
        for k in env_defaults.keys():
            if k in sample_dists.keys():
                if sample_dists[k][1][0]==list:
                    length = np.random.randint(5e5)
                    arg_class = sample_dists[k][1][1]
                    context_list = [arg_class(sample_dists[k][0].rvs(size=1)[0]) for i in range(length)]
                    c[k] = context_list
                else:
                    c[k] = sample_dists[k][0].rvs(size=1)[0]
                    c[k] = sample_dists[k][1](c[k])
                c[k] = np.clip(c[k], env_bounds[k][0], env_bounds[k][1])
            else:
                c[k] = env_defaults[k]
        contexts[i] = c

    return contexts
