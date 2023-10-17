import numpy as np
from tqdm import tqdm
from typing import Dict


def get_renders(env_specs: Dict, n_renders: int):
    renders = {}
    states = {}
    for env_cls, contexts in env_specs.items():
        print(env_cls.__name__)
        env = env_cls(contexts=contexts)
        _renders = []
        _states = []
        for i in tqdm(range(n_renders)):
            s = env.reset()
            _states.append(s)
            _renders.append(env.render(mode="rgb_array"))
        renders[env_cls.__name__] = _renders
        states[env_cls.__name__] = np.array(_states)
    return renders, states
