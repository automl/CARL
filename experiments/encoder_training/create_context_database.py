import json
import hydra
from omegaconf import DictConfig

import os
import numpy as np

base_dir = os.getcwd()


@hydra.main("./configs", "db")
def generate_database(cfg: DictConfig) -> None:
    """
    Generate a database of context vectors.
    """
    out_file = os.path.join(base_dir, cfg.out_file)

    mixed = []
    with open(os.path.join(base_dir, cfg.json_file), "r") as f:
        context_var = json.load(f)

    for key in context_var:
        temp = []
        for context in context_var[key]:
            temp.append(context_var[key][context])

        mixed.append(temp)

    np.save(out_file, mixed, allow_pickle=True)


if __name__ == "__main__":
    generate_database()
