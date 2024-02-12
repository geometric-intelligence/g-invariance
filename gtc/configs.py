import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transform_datasets.patterns.natural import MNIST
from transform_datasets.transforms import (O2, SO2, AddChannelDim, CenterMean,
                                           CircleCrop, Resize, UnitStd)

from configs.data.o2_mnist_16 import dataset_config
from configs.data_loader.bs64_val02 import data_loader_config
from configs.loss.cross_entropy import loss_config
from configs.model.d16_cnn_tc_10 import model_config
from configs.optimizer.adam_5e5 import optimizer_config
from configs.scheduler.plateau import scheduler_config
from gtc.modules import (FullyConnectedBlock, GonR2ConvBlock, GTtoT, Linear,
                         Ravel)
from gtc.pooling import GroupPooling
from gtc.trainer import GTrainer
from gtc.utils import Config, WBLogger

"""
MASTER CONFIG
"""

master_config = {
    "dataset": dataset_config,
    "data_loader": data_loader_config,
    "model": model_config,
    "loss": loss_config,
    "optimizer": optimizer_config,
    "scheduler": scheduler_config,
    "trainer": GTrainer,
    "seed": 0,
}

logger_config = Config(
    {
        "type": WBLogger,
        "params": {
            "project": "bispectrumnn",
            "data_project": "bispectrumnn",
            "entity": "johmathe",
            "log_interval": 1,
            "watch_interval": 1,
            "plot_interval": 1,
            "end_plotter": None,
            "step_plotter": None,
        },
    }
)


def dataset(group_continuous, dataset, seed):
    paths = {
        "mnist": "datasets/mnist/mnist_train.csv",
        "emnist": "datasets/emnist/emnist_letters_train.csv",
    }
    continous = {"o2": O2, "so2": SO2}
    pattern_config = Config(
        {
            "type": MNIST,
            "params": {"path": paths[dataset]},  # WAs /datasets
        }
    )
    transforms_config = {
        "0": Config(
            {
                "type": continous[group_continuous],
                "params": {"sample_method": "random"},
            }
        ),
        "1": Config({"type": Resize, "params": {"new_size": (16, 16)}}),
        "2": Config({"type": CircleCrop, "params": {}}),
        "3": Config({"type": AddChannelDim, "params": {}}),
    }
    dataset_config = {
        "pattern": pattern_config,
        "transforms": transforms_config,
        "seed": seed,
    }

    return dataset_config


def trainer(group, group_continous, dataset, pooling, seed):
    model_config = {
        "type": "D16CNN",
        "params": {
            "in_channels": 1,
            "out_channels": 10,
            "pooling": pooling,
            "group": group,
            "group_continuous": group_continous,
        },
    }
    scheduler_config = Config(
        {
            "type": ReduceLROnPlateau,
            "params": {"factor": 0.5, "patience": 2, "min_lr": 1e-4},
        }
    )
    optimizer_config = Config(
        {"type": optim.Adam, "params": {"lr": 5e-5, "weight_decay": 1e-5}}
    )
    loss_config = Config({"type": torch.nn.CrossEntropyLoss, "params": {}})
    master_config = {
        "dataset": dataset(group_continous, dataset, seed),
        "data_loader": data_loader_config,
        "model": model_config,
        "loss": loss_config,
        "optimizer": optimizer_config,
        "scheduler": scheduler_config,
        "trainer": GTrainer,
        "seed": seed,
    }
    return master_config
