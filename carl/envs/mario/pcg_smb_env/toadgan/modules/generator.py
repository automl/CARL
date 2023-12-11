# Code from https://github.com/Mawiszus/TOAD-GAN
from argparse import Namespace

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .conv_block import ConvBlock


class Level_GeneratorConcatSkip2CleanAdd(nn.Module):
    """Patch based Generator. Uses namespace opt."""

    def __init__(self, opt: Namespace):
        super().__init__()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_current, N, (3, 3), opt.padd_size, 1)
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, (3, 3), opt.padd_size, 1)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, (3, 3), opt.padd_size, 1)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        self.tail = nn.Sequential(
            nn.Conv2d(
                N, opt.nc_current, kernel_size=(3, 3), stride=1, padding=opt.padd_size
            ),
        )

    def forward(self, x: Tensor, y: Tensor, temperature: float = 1) -> Tensor:
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = F.softmax(
            x * temperature, dim=1
        )  # Softmax is added here to allow for the temperature parameter
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind : (y.shape[2] - ind), ind : (y.shape[3] - ind)]
        return x + y
