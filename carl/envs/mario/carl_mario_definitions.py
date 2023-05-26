import numpy as np
from torch import Tensor

try:
    from carl.envs.mario.pcg_smb_env import generate_initial_noise
except FileNotFoundError:

    def generate_initial_noise(width: int, height: int, level_index: int) -> Tensor:
        return Tensor()


def gen_noise(*args, **kwargs):
    return generate_initial_noise(INITIAL_WIDTH, INITIAL_HEIGHT, INITIAL_LEVEL_INDEX)


INITIAL_WIDTH = 100
INITIAL_LEVEL_INDEX = 0
INITIAL_HEIGHT = 16
DEFAULT_CONTEXT = {
    "level_index": INITIAL_LEVEL_INDEX,
    "noise": 0,
    "mario_state": 0,
    "mario_inertia": 0.89,
}
CONTEXT_BOUNDS = {
    "level_index": (None, None, "categorical", np.arange(0, 14)),
    "noise": (-np.inf, np.inf, int),
    "mario_state": (None, None, "categorical", [0, 1, 2]),
    "mario_inertia": (0.5, 1.5, float),
}
CATEGORICAL_CONTEXT_FEATURES = ["level_index", "mario_state"]
