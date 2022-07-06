# Code from https://github.com/Mawiszus/TOAD-GAN
from typing import Tuple, Union

import torch.nn as nn


class ConvBlock(nn.Sequential):
    """Conv block containing Conv2d, BatchNorm2d and LeakyReLU Layers."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        ker_size: Union[int, Tuple[int, int]],
        padd: Union[str, Union[int, Tuple[int, int]]],
        stride: Union[int, Tuple[int, int]],
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=ker_size,
                stride=stride,
                padding=padd,
            ),
        )

        self.add_module("norm", nn.BatchNorm2d(out_channel))
        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
