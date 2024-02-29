"""Definition of a neural network for MNIST classification."""

import torch
from escnn import gspaces
from torch import nn

import bispectrum.modules as gtc_modules
import bispectrum.pooling as gtc_pooling


class Net(nn.Module):
    POOLING_MAP = {
        "bsp": gtc_pooling.BspGroupPooling,
        "tc": gtc_pooling.TCGroupPooling,
        "max": gtc_pooling.GroupPooling,
    }

    def __init__(self, config):
        super(Net, self).__init__()

        # Do we even need an external module here?
        conv_block = gtc_modules.GonR2ConvBlock(
            N=config.N,
            # Should this match SO2/O2? i.e no flip?
            action=gspaces.flipRot2dOnR2,
            n_channels=4,
            kernel_size=16,
            padding=0,
            bias=False,
        )
        self.model = self.model = torch.nn.Sequential(
            conv_block,
            self.POOLING_MAP[config.pooling](
                idx=None, group_type=config.group_type, in_type=conv_block.out_type
            ),
            gtc_modules.GTtoT(),
            gtc_modules.Ravel(),
            gtc_modules.FullyConnectedBlock(
                in_dim=config.out_dim, out_dim=config.out_dim
            ),
            gtc_modules.FullyConnectedBlock(
                in_dim=config.out_dim, out_dim=config.out_dim
            ),
            gtc_modules.FullyConnectedBlock(
                in_dim=config.out_dim, out_dim=config.out_dim
            ),
            gtc_modules.Linear(in_dim=config.out_dim, out_dim=10),
        )

    def forward(self, x):
        return self.model(x)
