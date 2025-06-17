from typing import Any, Dict, List, Tuple, Type

import numpy as np


def get_context_bounds(
    context_keys: List[str], context_bounds: Dict[str, Tuple[float, float, Type[Any]]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get context bounds for specific features.

    Could add sophisticated method here.

    Parameters
    ----------
    context_keys: List[str]
        Names of context features.
    context_bounds: Dict[str, Tuple[float, float, type]]
        Dictionary containing lower and upper bound as a tuple, e.g., "context_feature_name": (-np.inf, np.inf)).

    Returns
    -------
    lower_bounds, upper_bounds: np.array, np.array
        Lower and upper bounds as arrays.

    """
    lower_bounds = np.empty(shape=len(context_keys))
    upper_bounds = np.empty(shape=len(context_keys))

    for i, context_key in enumerate(context_keys):
        lower, upper, dtype = context_bounds[context_key]
        lower_bounds[i] = lower
        upper_bounds[i] = upper

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
        "min_position": (-np.inf, np.inf, float),
        "max_position": (-np.inf, np.inf, float),
        "max_speed": (0, np.inf, float),
        "goal_position": (-np.inf, np.inf, float),
        "goal_velocity": (-np.inf, np.inf, float),
        "force": (-np.inf, np.inf, float),
        "gravity": (0, np.inf, float),
        "min_position_start": (-np.inf, np.inf, float),
        "max_position_start": (-np.inf, np.inf, float),
        "min_velocity_start": (-np.inf, np.inf, float),
        "max_velocity_start": (-np.inf, np.inf, float),
    }
    lower, upper = get_context_bounds(list(DEFAULT_CONTEXT.keys()), CONTEXT_BOUNDS)
