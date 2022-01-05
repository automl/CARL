import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)  # go up twice
sys.path.insert(0, parentdir)
print(os.getcwd())
from src.experiments.policy_transfer.policy_transfer import train_env
from src.train import get_parser

if __name__ == "__main__":
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    train_env( args, unknown_args, parser)
