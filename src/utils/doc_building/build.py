from src.envs import *
import pandas as pd
import numpy as np
from pathlib import Path
from gym import spaces

# JUST FOR DOCS / is dynamic in code!!
MARIO_OBSERVATION_SPACE = spaces.Box(
            low=0,
            high=255,
            shape=[4, 64, 64],
            dtype=np.uint8,
        )  # is dynamic in code
MARIO_ACTION_SPACE = spaces.Discrete(n=10)


def build():
    filepath = Path(__file__)
    outdir = filepath.parent.parent.parent.parent / "docs/source/environments/data"
    print("Build environment overview table.")
    # Create snapshot
    local_vars = globals().copy()

    k_env_family = "Env. Family"
    k_env_name =  "Name"
    k_n_context_features = "# Context Features"
    k_action_space = "Action Space"
    k_obs_space = "Obs. Space"

    # Filter envs
    mustinclude = "CARL"
    forbidden = ["defaults", "bounds"]

    overview_table_entries = []
    bounds_entries = {}
    defaults_entries = {}
    for varname, var in local_vars.items():
        if mustinclude in varname and not np.any([f in varname for f in forbidden]):
            env_name = varname
            module = var.__module__
            env_family = module.split(".")[-2]

            env = var()
            action_space = str(env.env.action_space)
            observation_space = str(env.env.observation_space)
            context = env.contexts[list(env.contexts.keys())[0]]
            n_context_features = len(context)

            data = {
                k_env_family: env_family,
                k_env_name: varname,
                k_n_context_features: n_context_features,
                k_action_space: action_space,
                k_obs_space: observation_space,
            }
            overview_table_entries.append(data)

            defaults_entries[env_name] = local_vars[f"{env_name}_defaults"]
            bounds_entries[env_name] = local_vars[f"{env_name}_bounds"]

            # if len(overview_table_entries) == 3:  # TODO change back
            #     break

    # Add RNA and Mario Information
    env_families = ["RNA", "Mario"]
    env_names = ["CARLRnaDesignEnv", "CARLMarioEnv"]
    from src.envs.rna.carl_rna_definitions import DEFAULT_CONTEXT as rna_defaults
    from src.envs.rna.carl_rna_definitions import CONTEXT_BOUNDS as rna_bounds
    from src.envs.rna.carl_rna_definitions import ACTION_SPACE as rna_A
    from src.envs.rna.carl_rna_definitions import OBSERVATION_SPACE as rna_O
    from src.envs.mario.carl_mario_definitions import DEFAULT_CONTEXT as mario_defaults
    from src.envs.mario.carl_mario_definitions import CONTEXT_BOUNDS as mario_bounds
    unicorn_defaults = [rna_defaults, mario_defaults]
    N_context_features = [len(c) for c in unicorn_defaults]
    action_spaces = [rna_A, MARIO_ACTION_SPACE]
    observation_spaces = [rna_O, MARIO_OBSERVATION_SPACE]
    unicorn_bounds = [rna_bounds, mario_bounds]
    for i in range(len(env_names)):
        data = {
            k_env_family: env_families[i],
            k_env_name: env_names[i],
            k_n_context_features: N_context_features[i],
            k_action_space: action_spaces[i],
            k_obs_space: observation_spaces[i],
        }
        overview_table_entries.append(data)
        defaults_entries[env_names[i]] = unicorn_defaults[i]
        bounds_entries[env_names[i]] = unicorn_bounds[i]
    df = pd.DataFrame(overview_table_entries)

    # Save overview table
    csv_filename = outdir / "tab_overview_environments.csv"
    csv_filename.parent.mkdir(exist_ok=True, parents=True)
    overview_columns = [
        k_env_family,
        k_env_name,
        k_n_context_features,
        k_action_space,
        k_obs_space,
    ]
    save_df = df[overview_columns]
    save_df.to_csv(csv_filename, index=False)

    # Save context defaults as tables
    for env_name, defaults in defaults_entries.items():
        fname = outdir / f"context_defaults/{env_name}.csv"
        fname.parent.mkdir(parents=True, exist_ok=True)
        defaults_df = pd.Series(defaults)
        defaults_df.index.name = "Context Feature"
        defaults_df.name = "Default"
        defaults_df.to_csv(fname)

    # Save context bounds as tables
    for env_name, bounds in bounds_entries.items():
        fname = outdir / f"context_bounds/{env_name}.csv"
        fname.parent.mkdir(parents=True, exist_ok=True)
        bounds_df = pd.Series(bounds)
        bounds_df.index.name = "Context Feature"
        bounds_df.name = "Bounds"
        bounds_df.to_csv(fname)

    print("Done!")

    return df, defaults_entries, bounds_entries


if __name__ == '__main__':
    df, defaults_entries, bounds_entries = build()

