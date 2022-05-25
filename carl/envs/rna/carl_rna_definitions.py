import numpy as np
from gym import spaces

#FIXME: how much sense do these make? Eterna solvers are unique and I'm not sure how to get the solvers for taneda
#SOLVER_LIST_ETERNA = [22230]
#SOLVER_LIST_RFAM_TANEDA = [None]
#SOLVER_LIST_RFAM_LEARN = [None]

ID_LIST_ETERNA = np.arange(1, 101)
ID_LIST_RFAM_TANEDA = np.arange(1, 30)
ID_LIST_RFAM_LEARN = np.arange(1, 65001)

DEFAULT_CONTEXT = {
    "mutation_threshold": 5,
    "reward_exponent": 1,
    "state_radius": 5,
    "dataset": "rfam_taneda",
    "target_structure_ids": None,
    # if solvers is set to 'None', all solvers are eligible
#    "solvers": None,
}
CONTEXT_BOUNDS = {
    "mutation_threshold": (0.1, np.inf, float),
    "reward_exponent": (0.1, np.inf, float),
    "state_radius": (1, np.inf, float),
    "dataset": (None, None, "categorical", ["eterna", "rfam_taneda", "rfam_learn", None]),
    "target_structure_ids": (None, None, "conditional", {"eterna": ID_LIST_ETERNA, "rfam_taneda": ID_LIST_RFAM_TANEDA, "rfan_learn": ID_LIST_RFAM_LEARN, None: [None]}, "dataset"),
#    "solvers": {"eterna": SOLVER_LIST_ETERNA, "rfam_taneda": SOLVER_LIST_RFAM_TANEDA, "rfan_learn": SOLVER_LIST_RFAM_LEARN, None: [None]},
}

ACTION_SPACE = spaces.Discrete(4)
OBSERVATION_SPACE = spaces.Box(low=-np.inf * np.ones(11), high=np.inf * np.ones(11))
