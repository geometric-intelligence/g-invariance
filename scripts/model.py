"""Definition of a neural network for MNIST classification."""

import torch
import torch.nn.functional as F
from torch import nn
from transform_datasets import transforms
import gtc.modules as gtc_modules
import gtc.pooling as gtc_pooling
from escnn import gspaces


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        blocks = [
            getattr(transforms, config.continuous_group),
            transforms.Resize((16, 16)),
            transforms.CircleCrop(),
            transforms.AddChannelDim(),
        ]
        self.model = torch.nn.Sequential(blocks)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        return self.model(x)


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        pooling_map = {
            "bsp": gtc_pooling.BspGroupPooling,
            "tc": gtc_pooling.TCGroupPooling,
            "max": gtc_pooling.GroupPooling,
        }
        self.model = self.model = torch.nn.Sequential(
            [
                gtc_modules.GonR2ConvBlock(
                    N=config.N,
                    action=gspaces.flipRot2dOnR2,
                    n_channels=4,
                    kernel_size=16,
                    padding=0,
                    bias=False,
                ),
                pooling_map[config.pooling](idx=None, group_type=config.group_type),
                gtc_modules.GTtoT(),
                gtc_modules.Ravel(),
                gtc_modules.FullyConnectedBlock(out_dim=config.out_dim),
                gtc_modules.FullyConnectedBlock(out_dim=config.out_dim),
                gtc_modules.FullyConnectedBlock(out_dim=config.out_dim),
                gtc_modules.Linear(out_dim=config.out_dim),
            ]
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        return self.model(x)
