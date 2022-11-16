from omegaconf import OmegaConf
from experiments.common.utils.json_utils import lazy_json_dump
from experiments.carlbench.context_sampling import ContextSampler
from rich import print as printr
from pathlib import Path
import numpy as np
import sys


if __name__ == "__main__":
    env_name = sys.argv[1]
    n_contexts = 1024
    config_path = "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/experiments/benchmarking/configs/experiments/context_efficiency.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.env = env_name
    cfg.seed = 42
    cfg.context_sampler.n_samples = n_contexts
    cfg = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg)

    contexts = ContextSampler(**cfg["context_sampler"]).sample_contexts()
    print(len(contexts))

    contexts_path = f"/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/runs/context_efficiency/{env_name}/contexts_evaluation.json"
    contexts_path = Path(contexts_path)
    print(contexts_path)
    contexts_path.parent.mkdir(parents=True, exist_ok=True)
    lazy_json_dump(contexts, contexts_path)