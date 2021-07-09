import tabulate

from src.envs import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

global_vars = vars()
vars = {k: v for k, v in global_vars.items() if "Env" in k or "Meta" in k}
env_names = [n for n in vars.keys() if "bounds" not in n and "defaults" not in n]
env_context_feature_names = {}


for env_name in env_names:
    defaults = pd.Series(vars[env_name + "_defaults"])
    bounds = vars[env_name + "_bounds"]
    bounds = {k: (v[0], v[1], v[2].__name__) for k, v in bounds.items()}
    bounds = pd.Series(bounds)

    df = pd.DataFrame()
    df["default"] = defaults
    df["bounds"] = bounds
    rows = df.index
    df.index = [r.replace("_", " ") for r in rows]

    env_context_feature_names[env_name] = defaults.keys()


