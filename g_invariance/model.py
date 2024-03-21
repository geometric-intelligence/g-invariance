"""Definition of a neural network for MNIST classification with
various pooling approaches.
"""

import torch
import torch.nn.functional as F
from escnn import gspaces
from torch import nn

import g_invariance.modules as gtc_modules
import g_invariance.pooling as gtc_pooling
import math


class VanillaNet(nn.Module):
    def __init__(self, config):
        super(VanillaNet, self).__init__()

        layer_1_size = config.fc_sizes[0]
        layer_2_size = config.fc_sizes[1]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(16 * 16, layer_1_size)
        self.layer_2 = torch.nn.Linear(layer_1_size, layer_2_size)
        self.layer_3 = torch.nn.Linear(layer_2_size, config.fc_sizes[3])

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = F.relu(x)

        x = self.layer_2(x)
        x = F.relu(x)

        x = self.layer_3(x)
        return F.log_softmax(x, dim=1)


class GInvNet(nn.Module):
    # TODO: Currently the output size is hardcoded, should be
    # computed from the input size, conv and the pooling layer

    POOLING_MAP = {
        'bsp': gtc_pooling.BspGroupPooling,
        'tc': gtc_pooling.TCGroupPooling,
        'max': gtc_pooling.GroupPooling,
    }

    def __init__(self, config):
        super(GInvNet, self).__init__()
        conv_block = gtc_modules.GonR2ConvBlock(
            N=config.N,
            # Should this match SO2/O2? i.e no flip?
            action=gspaces.flipRot2dOnR2,
            n_channels=config.n_filters,
            kernel_size=16,
            padding=0,
            bias=False,
        )
        pooling_output_size = self.pooling_output_size(
            config.pooling, config.n_filters, config.group, config.N
        )
        print(pooling_output_size)
        self.model = self.model = torch.nn.Sequential(
            conv_block,
            self.POOLING_MAP[config.pooling](
                idx=None, group_type=config.group, in_type=conv_block.out_type
            ),
            gtc_modules.GTtoT(),
            gtc_modules.Ravel(),
            gtc_modules.FullyConnectedBlock(in_dim=pooling_output_size, out_dim=config.fc_sizes[0]),
            gtc_modules.FullyConnectedBlock(in_dim=config.fc_sizes[0], out_dim=config.fc_sizes[1]),
            gtc_modules.FullyConnectedBlock(in_dim=config.fc_sizes[1], out_dim=config.fc_sizes[2]),
            gtc_modules.Linear(in_dim=config.fc_sizes[2], out_dim=config.fc_sizes[3]),
        )

    @staticmethod
    def pooling_output_size(pooling_type, n_filters, group_type, group_size):
        if pooling_type == 'max':
            return n_filters
        elif pooling_type == 'bsp':
            if group_type == 'cyclic':
                return 4 * n_filters * group_size
            elif group_type == 'dihedral':
                return int(n_filters * (math.floor((group_size - 1) / 2) * 16 + 5))
            else:
                raise ValueError(f'unknown group_type: {group_type}')
        elif pooling_type == 'tc':
            return int(group_size * 2 * (group_size * 2 + 1) / 2 * n_filters)
        else:
            raise ValueError(f'unkown pooling_type: {pooling_type}')

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)
