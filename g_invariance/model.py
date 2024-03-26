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

        self.layer_1 = torch.nn.Linear(config.img_size * config.img_size, layer_1_size)
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


class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class GInvNet(nn.Module):
    POOLING_MAP = {
        'bsp': gtc_pooling.BspGroupPooling,
        'tc': gtc_pooling.TCGroupPooling,
        'max': gtc_pooling.GroupPooling,
    }

    def __init__(self, config):
        super(GInvNet, self).__init__()
        if config.group == 'cyclic':
            action = gspaces.rot2dOnR2
        elif config.group == 'dihedral':
            action = gspaces.flipRot2dOnR2
        else:
            raise ValueError(f'unknown group {config.group}')

        conv_block = gtc_modules.GonR2ConvBlock(
            N=config.N,
            action=action,
            n_channels=config.n_filters,
            kernel_size=config.img_size,
            padding=0,
            bias=False,
            n_input_channels=config.n_input_channels,
        )
        pooling_output_size = self.pooling_output_size(
            config.pooling, config.n_filters, config.group, config.N
        )
        first_fc_size = self.get_best_fc_size(
            pooling_output_size, config.fc_sizes, config.target_params_count
        )

        self.model = self.model = torch.nn.Sequential(
            conv_block,
            self.POOLING_MAP[config.pooling](
                idx=None,
                group_type=config.group,
                in_type=conv_block.out_type,
            ),
            gtc_modules.GTtoT(),
            gtc_modules.Ravel(),
            gtc_modules.FullyConnectedBlock(in_dim=pooling_output_size, out_dim=first_fc_size),
            gtc_modules.FullyConnectedBlock(in_dim=first_fc_size, out_dim=config.fc_sizes[0]),
            gtc_modules.FullyConnectedBlock(in_dim=config.fc_sizes[0], out_dim=config.fc_sizes[1]),
            nn.Dropout(p=0.2),
            gtc_modules.Linear(in_dim=config.fc_sizes[1], out_dim=config.fc_sizes[2]),
        )

    @staticmethod
    def estimate_param_count(pooling_output_size, first_fc, fc_sizes):
        param_count = pooling_output_size * first_fc
        # batch norm adds 2x output size in FC
        param_count += first_fc * fc_sizes[0] + 2 * pooling_output_size
        for i in range(0, len(fc_sizes) - 1):
            param_count += fc_sizes[i] * fc_sizes[i + 1]
        # batch norms
        param_count += 2 * (fc_sizes[0] + fc_sizes[1])
        return param_count

    @staticmethod
    def get_best_fc_size(pooling_output_size, fc_sizes, target_params_count):
        # run a bisection search to find the best fc size
        left = 0
        right = target_params_count
        while left < right:
            mid = (left + right) // 2
            size = GInvNet.estimate_param_count(pooling_output_size, mid, fc_sizes)
            if size < target_params_count:
                left = mid + 1
            else:
                right = mid
        return left + 1

    @staticmethod
    def pooling_output_size(pooling_type, n_filters, group_type, group_size):
        if pooling_type == 'max':
            return n_filters
        elif pooling_type == 'bsp':
            if group_type == 'cyclic':
                return 2 * n_filters * group_size
            elif group_type == 'dihedral':
                return int(n_filters * (math.floor((group_size - 1) / 2) * 16 + 5))
            else:
                raise ValueError(f'unknown group_type: {group_type}')
        elif pooling_type == 'tc':
            if group_type == 'cyclic':
                return int(group_size * (group_size + 1) / 2 * n_filters)
            elif group_type == 'dihedral':
                return int(group_size * (group_size * 2 + 1) * n_filters)
            else:
                raise ValueError(f'unknown group_type: {group_type}')
        else:
            raise ValueError(f'unkown pooling_type: {pooling_type}')

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)
