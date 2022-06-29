from typing import Union
from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, Any
import coax
from rich import print


def load_func_dict(path: Union[str, Path]):
    func_dict = coax.utils.load(filepath=path)
    return func_dict


def load_policy(cfg: DictConfig, weights_path: Union[str, Path]):
    func_dict = load_func_dict(weights_path)
    policy = func_dict["pi"]
    return policy
