import hydra
import yaml
import os
from pathlib import Path
from rich import print
from omegaconf import DictConfig

# from searl.neuroevolution.searl_td3 import start_searl_td3_run  # TODO modify
import sys
sys.path.append("../../../carl/training")
from carl.experiments import start_searl_run
from experiments.hpo.searl.custom_searl_td3 import CustomSEARLforTD3


@hydra.main("configs", "config_td3.yaml")
def main(args: DictConfig):
    print(args)
    if args.config_file is None:
        print("no config file")
        config_file = Path(__file__).parent / "SEARL" / "configs/searl_td3_config.yml"
    else:
        config_file = args.config_file

    if args.expt_dir is None:
        print("no experiment dir")
        expt_dir = Path(__file__).parent / "experiments"
    else:
        expt_dir = args.expt_dir

    os.environ["LD_LIBRARY_PATH"] = f"$LD_LIBRARY_PATH:{str(Path.home())}/.mujoco/mujoco200/bin:/usr/lib/nvidia-384"

    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)

    for k, v in config_dict["seed"].items():
        v = args.seed
        config_dict["seed"][k] = v


    print(config_dict)
    start_searl_run(config_dict, expt_dir=expt_dir, searl_algorithm=CustomSEARLforTD3)


if __name__ == "__main__":
    main()

    # Start on command line to use with slurm:
    # python training/hpo/run_searl_td3.py --multirun seed=0,1,2,3,4 &
