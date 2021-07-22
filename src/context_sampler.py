import numpy as np
from typing import List, Dict, Any
from scipy.stats import norm

from src import envs


def get_default_context_and_bounds(env_name: str):
    # TODO make less hacky / make explicit
    env_defaults = getattr(envs, f"{env_name}_defaults")
    env_bounds = getattr(envs, f"{env_name}_bounds")

    return env_defaults, env_bounds


def sample_contexts(
        env_name: str,
        context_feature_args: List[str],
        num_contexts: int,
        default_sample_std_percentage: float = 0.05,
        fallback_sample_std: float = 0.1,
) -> Dict[Any, Any]:
    env_defaults, env_bounds = get_default_context_and_bounds(env_name=env_name)

    sample_dists = {}
    for key in env_defaults.keys():
        if key in context_feature_args:
            if f"{key}_mean" in context_feature_args:
                sample_mean = float(context_feature_args[context_feature_args.index(f"{key}_mean")+1])
            else:
                sample_mean = env_defaults[key]

            if f"{key}_std" in context_feature_args:
                sample_std = float(context_feature_args[context_feature_args.index(f"{key}_std")+1])
            else:
                sample_std = default_sample_std_percentage * np.abs(sample_mean)

            if sample_mean == 0:
                sample_std = fallback_sample_std  # TODO change this back to sample_std

            random_variable = norm(loc=sample_mean, scale=sample_std)
            context_feature_type = env_bounds[key][2]
            sample_dists[key] = (random_variable, context_feature_type)

    contexts = {}
    for i in range(0, num_contexts):
        c = {}
        for k in env_defaults.keys():
            if k in sample_dists.keys():
                random_variable = sample_dists[k][0]
                context_feature_type = sample_dists[k][1]
                lower_bound, upper_bound = env_bounds[k][0], env_bounds[k][1]
                if context_feature_type == list:
                    length = np.random.randint(5e5)
                    arg_class = sample_dists[k][1][1]
                    context_list = random_variable.rvs(size=length)
                    context_list = np.clip(context_list, lower_bound, upper_bound)
                    c[k] = [arg_class(c) for c in context_list]
                elif context_feature_type == "categorical":
                    choices = env_bounds[k][3]
                    choice = np.random.choice(choices)
                    c[k] = choice
                else:
                    c[k] = random_variable.rvs(size=1)[0]
                    c[k] = np.clip(c[k], lower_bound, upper_bound)
                    c[k] = context_feature_type(c[k])
            else:
                c[k] = env_defaults[k]
        contexts[i] = c

    return contexts
