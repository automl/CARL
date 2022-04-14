from typing import Dict, Tuple

from pathlib import Path

import numpy as np
import pandas as pd
from gym import spaces

import carl.envs

# JUST FOR DOCS / is dynamic in code!!
MARIO_OBSERVATION_SPACE = spaces.Box(
    low=0,
    high=255,
    shape=[4, 64, 64],
    dtype=np.uint8,
)  # is dynamic in code
MARIO_ACTION_SPACE = spaces.Discrete(n=10)


def build() -> Tuple[pd.DataFrame, Dict, Dict]:
    filepath = Path(__file__)
    outdir = filepath.parent.parent.parent.parent / "docs/source/environments/data"
    print("Build environment overview table.")
    # Create snapshot
    local_vars = vars(carl.envs)

    k_env_family = "Env. Family"
    k_env_name = "Name"
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
    from carl.envs.mario.carl_mario_definitions import CONTEXT_BOUNDS as mario_bounds
    from carl.envs.mario.carl_mario_definitions import DEFAULT_CONTEXT as mario_defaults
    from carl.envs.rna.carl_rna_definitions import ACTION_SPACE as rna_A
    from carl.envs.rna.carl_rna_definitions import CONTEXT_BOUNDS as rna_bounds
    from carl.envs.rna.carl_rna_definitions import DEFAULT_CONTEXT as rna_defaults
    from carl.envs.rna.carl_rna_definitions import OBSERVATION_SPACE as rna_O

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

    env_names = list(defaults_entries.keys())

    for env_name in env_names:
        fname = outdir / f"context_definitions/{env_name}.csv"
        fname.parent.mkdir(parents=True, exist_ok=True)
        defaults = defaults_entries[env_name]
        defaults_df = pd.Series(defaults)
        defaults_df.index.name = "Context Feature"
        defaults_df.name = "Default"
        bounds = bounds_entries[env_name]
        bounds_df = pd.Series(bounds)
        bounds_df.index.name = "Context Feature"
        bounds_df.name = "Bounds"

        context_def_df = pd.concat([defaults_df, bounds_df], axis=1)
        context_def_df.to_csv(fname)

    print("Done!")

    return df, defaults_entries, bounds_entries


if __name__ == "__main__":
    df, defaults_entries, bounds_entries = build()
