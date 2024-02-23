import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transform_datasets import transforms
from transform_datasets.patterns.natural import MNIST

from gtc import model_config
from gtc.trainer import GTrainer
from gtc.utils import Config, TrainValLoader, WBLogger

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

data_loader_config = Config(
    {
        "type": TrainValLoader,
        "params": {
            "batch_size": 200,
            "fraction_val": 0.2,
            "num_workers": 5,
        },
    }
)


def get_dataset_config(group_continuous, dataset, seed):
    paths = {
        "mnist": "datasets/mnist/mnist_train.csv",
        "emnist": "datasets/emnist/emnist_letters_train.csv",
    }
    continous = {"o2": transforms.O2, "so2": transforms.SO2}
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
        "1": Config({"type": transforms.Resize, "params": {"new_size": (16, 16)}}),
        "2": Config({"type": transforms.CircleCrop, "params": {}}),
        "3": Config({"type": transforms.AddChannelDim, "params": {}}),
    }
    dataset_config = {
        "pattern": pattern_config,
        "transforms": transforms_config,
        "seed": seed,
    }

    return dataset_config


def get_trainer_config(group_continous, group, pooling, dataset, n_filters=10, seed=42):
    model = model_config.get_model_config(group, pooling, n_filters)
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
        "dataset": get_dataset_config(group_continous, dataset, seed),
        "data_loader": data_loader_config,
        "model": model,
        "loss": loss_config,
        "optimizer": optimizer_config,
        "scheduler": scheduler_config,
        "trainer": GTrainer,
        "seed": seed,
    }
    return master_config
