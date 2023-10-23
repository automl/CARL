import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import hydra
import sys
from omegaconf import DictConfig, OmegaConf

from experiments.common.utils.json_utils import lazy_json_load, lazy_json_dump
from experiments.benchmarking.training import get_contexts

from pathlib import Path
from rich import print as printr


base_dir = os.getcwd()


@hydra.main("./configs", "base", version_base="1.1")
def main(cfg: DictConfig):
    printr("Generating Context Sets")
    seed_train = 69274
    seed_test = 59466
    printr("Train Seed:", seed_train)
    printr("Test Seed:", seed_test)
    seeds = [seed_train, seed_test]
    context_sets = []
    for seed in seeds:
        cfg.seed = seed
        contexts = get_contexts(cfg=cfg)
        context_sets.append(contexts)
    
    ids = ["train", "test"]
    for identifier, contexts in zip(ids, context_sets):
        fn = Path(f"contexts_{identifier}.json")  
        lazy_json_dump(contexts, fn.resolve())    
        printr(fn.resolve())


if __name__ == "__main__":
    # Set the output_dir as the working directory
    sys.argv.append('hydra.run.dir=${output_dir}')
    main()
