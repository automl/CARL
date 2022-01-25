import argparse
import yaml
import os
from pathlib import Path

from searl.rl_algorithms.dqn import start_DQN_training

parser = argparse.ArgumentParser(description='define cluster setup')

parser.add_argument('--expt_dir', type=str, default=False, help='expt_dir')
parser.add_argument('--config_file', type=str, default=False, help='config_dir')
args = parser.parse_args()

if args.config_file == False:
    print("no config file")
    config_file = Path(os.getcwd()).parents[0] / "configs/dqn_config.yml"
else:
    config_file = args.config_file

if args.expt_dir == False:
    print("no experiment dir")
    expt_dir = Path(os.getcwd()).parents[0] / "experiments"
else:
    expt_dir = args.expt_dir


with open(config_file, 'r') as f:
    config_dict = yaml.load(f, Loader=yaml.Loader)

start_DQN_training(config_dict, expt_dir=expt_dir)
