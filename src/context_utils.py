import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import norm

from src import classic_control


def sample_contexts(env_name: str, unknown_args: List[str], num_contexts: int, default_sample_std: float = 0.05):
    # TODO makes separate folders harder to parse... there should be a better solution --> make explicit?
    env_defaults = getattr(classic_control, f"{env_name}_defaults")
    env_bounds = getattr(classic_control, f"{env_name}_bounds")

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
                c[k] = max(env_bounds[k][0], min(c[k], env_bounds[k][upper]))
            else:
                c[k] = env_defaults[k]
        contexts[i] = c

    return contexts


def get_context_bounds(context_keys: List[str], context_bounds: Dict[str, Tuple[float]]):
    """
    Get context bounds for specific features.

    Could add sophisticated method here.

    Parameters
    ----------
    context_keys: List[str]
        Names of context features.
    context_bounds: Dict[str, Tuple[float]]
        Dictionary containing lower and upper bound as a tuple, e.g., "context_feature_name": (-np.inf, np.inf)).

    Returns
    -------
    lower_bounds, upper_bounds: np.array, np.array
        Lower and upper bounds as arrays.

    """
    lower_bounds = np.empty(shape=len(context_keys))
    upper_bounds = np.empty(shape=len(context_keys))

    for i, context_key in enumerate(context_keys):
        l, u = context_bounds[context_key]
        lower_bounds[i] = l
        upper_bounds[i] = u

    return lower_bounds, upper_bounds


if __name__ == '__main__':
    DEFAULT_CONTEXT = {
        "min_position": -1.2,  # unit?
        "max_position": 0.6,  # unit?
        "max_speed": 0.07,  # unit?
        "goal_position": 0.5,  # unit?
        "goal_velocity": 0,  # unit?
        "force": 0.001,  # unit?
        "gravity": 0.0025,  # unit?
        "min_position_start": -0.6,
        "max_position_start": -0.4,
        "min_velocity_start": 0.,
        "max_velocity_start": 0.,
    }

    CONTEXT_BOUNDS = {
        "min_position": (-np.inf, np.inf),
        "max_position": (-np.inf, np.inf),
        "max_speed": (0, np.inf),
        "goal_position": (-np.inf, np.inf),
        "goal_velocity": (-np.inf, np.inf),
        "force": (-np.inf, np.inf),
        "gravity": (0, np.inf),
        "min_position_start": (-np.inf, np.inf),
        "max_position_start": (-np.inf, np.inf),
        "min_velocity_start": (-np.inf, np.inf),
        "max_velocity_start": (-np.inf, np.inf),
    }
    lower, upper = get_context_bounds(list(DEFAULT_CONTEXT.keys()), CONTEXT_BOUNDS)
