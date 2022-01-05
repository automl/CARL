from src.envs import *
import pandas as pd
import numpy as np
from pathlib import Path


def build():
    filepath = Path(__file__)
    csv_filename = filepath.parent.parent.parent.parent / "docs/source/environments/data/tab_overview_environments.csv"
    csv_filename.parent.mkdir(exist_ok=True, parents=True)
    if not csv_filename.is_file() or True:
        print("Build environment overview table.")
        # Create snapshot
        local_vars = globals().copy()

        # Filter envs
        mustinclude = "CARL"
        forbidden = ["defaults", "bounds"]

        entries = []
        for varname, var in local_vars.items():
            if mustinclude in varname and not np.any([f in varname for f in forbidden]):
                module = var.__module__
                env_family = module.split(".")[-2]

                env = var()
                action_space = str(env.env.action_space)
                observation_space = str(env.env.observation_space)
                context = env.contexts[list(env.contexts.keys())[0]]
                n_context_features = len(context)

                data = {
                    "Env. Family": env_family,
                    "Name": varname,
                    "# Context Features": n_context_features,
                    "Action Space": action_space,
                    "Obs. Space": observation_space,
                }
                entries.append(data)
                if len(entries) == 3:  # TODO change back
                    break
        df = pd.DataFrame(entries)
        df.to_csv(csv_filename, index=False)
        print("Done!")


if __name__ == '__main__':
    build()

