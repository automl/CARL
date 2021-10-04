import json
import numpy as np
from src.context.sampling import get_default_context_and_bounds
from src.experiments.policy_transfer import get_train_contexts_ll_exp1, get_train_contexts_ll, gravities

fname = "experiments/lunarLander_contexts_train_Gaussian.json"

if __name__ == '__main__':
    env_name = "CARLLunarLanderEnv"
    env_defaults, env_bounds = get_default_context_and_bounds(env_name=env_name)

    n_contexts = 100
    contexts = get_train_contexts_ll_exp1(n_contexts=n_contexts, env_default_context=env_defaults)
    contexts = get_train_contexts_ll(gravities=gravities, context_feature_key="GRAVITY_Y", n_contexts=n_contexts, env_default_context=env_defaults)

    with open(fname, 'w') as f:
        json.dump(contexts, f, indent="\t")