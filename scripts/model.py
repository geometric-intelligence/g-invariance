"""Definition of a neural network for MNIST classification."""

import torch
from escnn import gspaces
from torch import nn

import gtc.modules as gtc_modules
import gtc.pooling as gtc_pooling


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        pooling_map = {
            "bsp": gtc_pooling.BspGroupPooling,
            "tc": gtc_pooling.TCGroupPooling,
            "max": gtc_pooling.GroupPooling,
        }
        conv_block = gtc_modules.GonR2ConvBlock(
            N=config.N,
            action=gspaces.flipRot2dOnR2,
            n_channels=4,
            kernel_size=16,
            padding=0,
            bias=False,
        )
        self.model = self.model = torch.nn.Sequential(
            conv_block,
            pooling_map[config.pooling](
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
            gtc_modules.Linear(in_dim=config.out_dim, out_dim=config.out_dim),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        return self.model(x)
