import numpy as np
from gymnasium import spaces

DEFAULT_CONTEXT = {
    "mutation_threshold": 5,
    "reward_exponent": 1,
    "state_radius": 5,
    "dataset": "eterna",
    "target_structure_ids": None,
}
CONTEXT_BOUNDS = {
    "mutation_threshold": (0.1, np.inf, float),
    "reward_exponent": (0.1, np.inf, float),
    "state_radius": (1, np.inf, float),
    "dataset": ("eterna", "rfam_taneda", None),
    "target_structure_ids": (
        0,
        np.inf,
        [list, int],
    ),  # This is conditional on the dataset (and also a list)
}
ACTION_SPACE = spaces.Discrete(4)
OBSERVATION_SPACE = spaces.Box(low=-np.inf * np.ones(11), high=np.inf * np.ones(11))
