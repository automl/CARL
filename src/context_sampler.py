from typing import List
from scipy.stats import norm

from src import classic_control
from src import box2d


def get_default_context_and_bounds(env_name: str):
    # TODO makes separate folders harder to parse... there should be a better solution --> make explicit?
    # TODO maybe new folder structure? envs/box2d and envs/classic_control
    # TODO make less hacky
    try:
        env_defaults = getattr(classic_control, f"{env_name}_defaults")
        env_bounds = getattr(classic_control, f"{env_name}_bounds")
    except AttributeError:
        env_defaults = getattr(box2d, f"{env_name}_defaults")
        env_bounds = getattr(box2d, f"{env_name}_bounds")

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

            sample_dists[key] = norm(loc=sample_mean, scale=sample_std)

    contexts = {}
    for i in range(0, num_contexts):
        c = {}
        for k in env_defaults.keys():
            if k in sample_dists.keys():
                c[k] = sample_dists[k].rvs(size=1)[0]
                c[k] = max(env_bounds[k][0], min(c[k], env_bounds[k][1]))
            else:
                c[k] = env_defaults[k]
        contexts[i] = c

    return contexts
