import numpy as np
from typing import List, Dict, Tuple


def get_context_bounds(
    context_keys: List[str], context_bounds: Dict[str, Tuple[float]]
):
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
        l, u, _ = context_bounds[context_key]
        lower_bounds[i] = l
        upper_bounds[i] = u

    return lower_bounds, upper_bounds


if __name__ == "__main__":
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
        "min_velocity_start": 0.0,
        "max_velocity_start": 0.0,
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
