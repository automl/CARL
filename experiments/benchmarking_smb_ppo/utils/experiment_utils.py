import random
import subprocess
from typing import Dict, Union

import numpy as np
import torch
import torch.backends.cudnn


def set_seed(seed: int, torch_deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def obs_to_tensor(obs: Union[Dict, torch.Tensor], device) -> Union[Dict, torch.Tensor]:
    if isinstance(obs, dict):
        return {k: obs_to_tensor(v, device) for k, v in obs.items()}
    else:
        return torch.tensor(obs, device=device)
    
def add_batch_dim(obs: Union[Dict, torch.Tensor]) -> Union[Dict, torch.Tensor]:
    if isinstance(obs, dict):
        return {k: add_batch_dim(v) for k, v in obs.items()}
    else:
        return obs.unsqueeze(0)


def run_cmd(cmd):
    out = (subprocess.check_output(cmd, shell=True)).decode("utf-8")[:-1]
    return out


def get_free_gpu_indices():
    out = run_cmd("nvidia-smi -q -d Memory | grep -A4 GPU")
    out = (out.split("\n"))[1:]
    out = [l for l in out if "--" not in l]

    total_gpu_num = int(len(out) / 5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i * 5: i * 5 + 1]][0])

    out = run_cmd("nvidia-smi --query-compute-apps=gpu_bus_id --format=csv")
    gpu_bus_ids_in_use = (out.split("\n"))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]
