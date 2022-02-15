import numpy as np
from gym import spaces
SOLVER_LIST_ETERNA = []
SOLVER_LIST_RFAM_TANEDA = []
SOLVER_LIST_RFAM_LEARN = []

DEFAULT_CONTEXT = {
    "mutation_threshold": 5,
    "reward_exponent": 1,
    "state_radius": 5,
    "dataset": "rfam_taneda",
    "target_structure_ids": None,
    # if solvers is set to 'None', all solvers are eligible
    "solvers": None
}
CONTEXT_BOUNDS = {
    "mutation_threshold": (0.1, np.inf, float),
    "reward_exponent": (0.1, np.inf, float),
    "state_radius": (1, np.inf, float),
    "dataset": ("eterna", "rfam_taneda", "rfam_learn", None),
    "target_structure_ids": (
        0,
        np.inf,
        [list, int],
    ),  # This is conditional on the dataset (and also a list)
    #FIXME: depends on the dataset
    "solvers": None
}
ACTION_SPACE = spaces.Discrete(4)
OBSERVATION_SPACE = spaces.Box(low=-np.inf * np.ones(11), high=np.inf * np.ones(11))
