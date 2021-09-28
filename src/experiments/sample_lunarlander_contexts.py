import json
import numpy as np
from src.context.sampling import get_default_context_and_bounds

fname = "experiments/lunarLander_contexts_train_2intervals.json"
env_name = "CARLLunarLanderEnv"
env_defaults, env_bounds = get_default_context_and_bounds(env_name=env_name)

intervals = [(-20, -15), (-5, 1e-3)]
n_contexts = 100
n_c_per_interval = 100 // len(intervals)

contexts = {}
for interval_idx, interval in enumerate(intervals):
    gravities = np.random.uniform(*interval, size=n_c_per_interval)
    for i in range(n_c_per_interval):
        context = env_defaults.copy()
        context["GRAVITY_Y"] = gravities[i]
        contexts[i + interval_idx * n_c_per_interval] = context

with open(fname, 'w') as f:
    json.dump(contexts, f, indent="\t")