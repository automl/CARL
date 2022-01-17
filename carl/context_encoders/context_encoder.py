"""
Base class for context encoder ot be used by all variations of AutoEncoders. 

Structure adapted from: https://github.com/AntixK/PyTorch-VAE
"""


import torch as th
from typing import List, Callable, Union, Any, TypeVar, Tuple

from abc import abstractmethod

class ContextEncoder(th.nn.Module):
    
    def __init__(self) -> None:
        super(ContextEncoder, self).__init__()
    
    def encode(self, input: th.Tensor) -> th.Tensor:
        """
        Encodes the context.
        """
        raise NotImplementedError

    def decode(self, input: th.Tensor) -> th.Tensor:
        """
        Decodes the context.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: th.Tensor) -> List[th.Tensor]:
        """
        Encodes the context.
        """
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> dict:
        """
        Encodes the context.
        """
        pass

    def get_representation(self) -> th.Tensor:
        """
        Encodes the context.
        """
        raise NotImplementedError