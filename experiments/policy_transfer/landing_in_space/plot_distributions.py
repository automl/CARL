import hydra
from hydra.utils import instantiate
import os
from rich import print as printr
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt


if __name__ == "__main__":
    seed = 5
    n_contexts = 1000
    cfg_fns = [
        "../../../experiments/benchmarking/configs/experiments/lis/gravities_normal.yaml",
        "../../../experiments/benchmarking/configs/experiments/lis/gravities_uniform.yaml",
    ]
    for cfg_fn in cfg_fns:
        cfg = OmegaConf.load(cfg_fn)
        cfg.landing_in_space.sample_function.seed = seed
        cfg.landing_in_space.sample_function.n_contexts = n_contexts
        printr(cfg)

        # Sample gravities
        gravities = instantiate(cfg.landing_in_space.sample_function)

        fig = plt.figure(figsize=(6, 4), dpi=250)
        ax = fig.add_subplot(111)
        ax.hist(gravities, bins=100, color="cornflowerblue")
        ax.set_xlabel("Gravity [m/s²]")
        ax.set_ylabel("Counts")
        ax.set_title(
            cfg.landing_in_space.sample_function._target_.split(".")[-1].split("_")[-1]
        )
        fig.set_tight_layout(True)
        fig.savefig("landing_in_space_distribution.png", bbox_inches="tight", dpi=300)
        plt.show()
