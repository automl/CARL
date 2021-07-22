import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)  # go up twice
sys.path.insert(0, parentdir)
print(os.getcwd())
from src.experiments.policy_transfer import train_env

if __name__ == "__main__":
    train_env()
