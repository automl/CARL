from collections import namedtuple

import numpy as np
import torch

fields = ('state', 'action', 'next_state', 'reward', 'done', 'weight', 'index')
Transition = namedtuple('Transition', fields)
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)


def to_tensor(ndarray, requires_grad=False):
    return torch.from_numpy(ndarray).float().requires_grad_(requires_grad)


def feature_scaling(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
