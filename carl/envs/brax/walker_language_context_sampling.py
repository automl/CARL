import numpy as np
from carl.utils.types import Context, Contexts


def sample_walker_language_goals(
    num_contexts, low=5, high=2500, normal=False, mean=25000, std=0.1
):
    directions = [
        1,  # north
        3,  # south
        2,  # east
        4,  # west
        12,
        32,
        14,
        34,
        112,
        332,
        114,
        334,
        212,
        232,
        414,
        434,
    ]

    sampled_contexts: Contexts = {}

    for i in range(num_contexts):
        c: Context = {}
        c["target_direction"] = np.random.choice(directions)
        if normal:
            c["target_distance"] = np.round(
                min(max(np.random.normal(loc=mean, scale=std * mean), low), high),
                decimals=2,
            )
        else:
            c["target_distance"] = np.round(
                np.random.uniform(low=low, high=high), decimals=2
            )
        sampled_contexts[i] = c
    return sampled_contexts
