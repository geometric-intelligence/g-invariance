import collections

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transform_datasets import transforms
from transform_datasets.patterns.natural import MNIST

from configs.data.o2_mnist_16 import dataset_config
from configs.data_loader.bs64_val02 import data_loader_config
from configs.loss.cross_entropy import loss_config
from configs.model.d16_cnn_tc_10 import model_config
from configs.optimizer.adam_5e5 import optimizer_config
from configs.scheduler.plateau import scheduler_config
from gtc.modules import FullyConnectedBlock, GonR2ConvBlock, GTtoT, Linear, Ravel
from gtc.pooling import GroupPooling
from gtc.trainer import GTrainer
from gtc.utils import Config, WBLogger


def model_config(model, fc_dims=[500, 64, 64]):
    # d16_cnn_bsp_10
    model_config = collections.OrderedDict(
        {
            "conv1": {
                "type": "GonR2ConvBlock",
                "N": N,
                "action": "flipRot2dOnR2",
                "n_channels": 4,
                "kernel_size": 16,
                "padding": 0,
                "bias": False,
            },
            "gpool": {"type": "BspGroupPooling", "group_type": "dihedral"},
            "ravel": {"type": "Ravel"},
            "FC1": {"type": "FullyConnectedBlock", "out_dim": 500},
            "FC2": {"type": "FullyConnectedBlock", "out_dim": 64},
            "FC3": {"type": "FullyConnectedBlock", "out_dim": 64},
            "linear": {"type": "Linear", "out_dim": 10},
        }
    )
    model_config = collections.OrderedDict(
        [
            (
                "conv1",
                {
                    "type": GonR2ConvBlock,
                    "params": {
                        "N": N,  # Assuming N is defined elsewhere
                        "action": gspaces.flipRot2dOnR2,  # Assuming gspaces is imported or defined elsewhere
                        "n_channels": 4,
                        "kernel_size": 16,
                        "padding": 0,
                        "bias": False,
                    },
                },
            ),
            (
                "gpool",
                {
                    "type": GroupPooling,
                    "params": {},
                },
            ),
            (
                "gttot",
                {
                    "type": GTtoT,
                    "params": {},
                },
            ),
            (
                "ravel",
                {
                    "type": Ravel,
                    "params": {},
                },
            ),
            (
                "FC1",
                {
                    "type": FullyConnectedBlock,
                    "params": {"out_dim": 1850},
                },
            ),
            (
                "FC2",
                {
                    "type": FullyConnectedBlock,
                    "params": {"out_dim": 64},
                },
            ),
            (
                "FC3",
                {
                    "type": FullyConnectedBlock,
                    "params": {"out_dim": 64},
                },
            ),
            (
                "linear",
                {
                    "type": Linear,
                    "params": {"out_dim": 10},
                },
            ),
        ]
    )
    return model_config
