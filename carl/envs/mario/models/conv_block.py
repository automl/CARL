# Code from https://github.com/Mawiszus/TOAD-GAN
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """ Conv block containing Conv2d, BatchNorm2d and LeakyReLU Layers. """
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
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
        ),
        self.add_module("norm", nn.BatchNorm2d(out_channel)),
        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))

