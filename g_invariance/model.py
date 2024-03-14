"""Definition of a neural network for MNIST classification with
various pooling approaches.
"""

import torch
from escnn import gspaces
from torch import nn
import torch.nn.functional as F
import g_invariance.modules as gtc_modules
import g_invariance.pooling as gtc_pooling

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

    # TODO: These sizes depend on more parameters - implement it.
    POOLING_MAP = {
        "bsp": (gtc_pooling.BspGroupPooling, 128),
        "tc": (gtc_pooling.TCGroupPooling, 544),
        "max": (gtc_pooling.GroupPooling, 4),
    }

    def __init__(self, config):
        super(Net, self).__init__()
        # FIXME: Temporarily hardcoded until a general formula
        # for the output size is implemented.
        # The default value is for dihedral group.
        if config.group == "dihedral":
            self.POOLING_MAP["bsp"] = (gtc_pooling.BspGroupPooling, 212)
        # Do we even need an external module here?
        conv_block = gtc_modules.GonR2ConvBlock(
            N=config.N,
            # Should this match SO2/O2? i.e no flip?
            action=gspaces.flipRot2dOnR2,
            n_channels=config.n_filters,
            kernel_size=16,
            padding=0,
            bias=False,
        )
        # TODO: Should be computed directly from the conv_block
        self.model = self.model = torch.nn.Sequential(
            conv_block,
            self.POOLING_MAP[config.pooling][0](
                idx=None, group_type=config.group, in_type=conv_block.out_type
            ),
            gtc_modules.GTtoT(),
            gtc_modules.Ravel(),
            gtc_modules.FullyConnectedBlock(
                in_dim=self.POOLING_MAP[config.pooling][1], out_dim=config.fc_sizes[0]
            ),
            gtc_modules.FullyConnectedBlock(in_dim=config.fc_sizes[0], out_dim=config.fc_sizes[1]),
            gtc_modules.FullyConnectedBlock(in_dim=config.fc_sizes[1], out_dim=config.fc_sizes[2]),
            gtc_modules.Linear(in_dim=config.fc_sizes[2], out_dim=config.fc_sizes[3]),
        )

    def forward(self, x):
        return F.log_softmax(self.model(x))
