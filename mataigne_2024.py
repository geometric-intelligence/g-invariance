"""
This module performs hyperparameter tuning for the MNIST dataset using the ASHA algorithm.

It defines a training function `train_func` that trains a PyTorch model using the provided configuration.
It also defines a function `tune_mnist_asha` that performs hyperparameter tuning using the ASHA algorithm.

Example:
    results = tune_mnist_asha(num_samples=100)
"""

import pytorch_lightning as pl
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train import lightning as ray_lightning
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

import g_invariance.train as g_train


def train_func(config):
    """
    Trains a PyTorch model using the provided configuration.

    Args:
        config (dict): Configuration parameters for training.

    Returns:
        None
    """
    config_param = g_train.read_config_from_file()
    setup_wandb(config)
    config_param.update(config)
    dm = g_train.MNISTDataModule(config_param)
    model = g_train.MNISTClassifier(config_param)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=ray_lightning.RayDDPStrategy(),
        callbacks=[ray_lightning.RayTrainReportCallback()],
        plugins=[ray_lightning.RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs=config_param.max_epochs,
    )
    trainer = ray_lightning.prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


def tune_mnist_asha(num_samples=10):
    """
    Performs hyperparameter tuning for the MNIST dataset using the ASHA algorithm.

    Args:
        num_samples (int): The number of hyperparameter configurations to sample.

    Returns:
        None
    """
    scaling_config = ScalingConfig(
        num_workers=8, use_gpu=True, resources_per_worker={"CPU": 4, "GPU": 1}
    )

    run_config = RunConfig(
        callbacks=[WandbLoggerCallback(project="g_invariance")],
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    # From the paper:
    #     Table 2. Classification accuracy and parameters count of the different G-CNNs
    # models with G-TC, full or selective G-bispectrum or Max G-pooling. The experiments
    # are conducted on the SO(2)/O(2)-MNIST/EMNIST datasets. There are K = 24 filters
    # for the C8-CNN. For D8-CNN, there are K = 4/20 filters on O(2)-MNIST/EMNIST
    # respectively. The MLP specifications are detailed in Appendix D.
    # The different models are matched to have equivalent numbers of parameters.
    # **size of all MLP layers set to 26.

    # from Figure 4: bruteforce N filters in [2 3 4 5 6 7 8 9 10] and do:
    # Max G pooling, G-Triple correlation, Selective bi-spectrum.

    # C8 CNN:            SO2-MNIST        SO2-EMNIST
    # G-TC             # [64,64,64,10]  # [64,64,64,26]
    # Full G-bispect.  # [20,20,20,10]  # [26,26,26,26]
    # Sel. G-bispect.  # [64,64,64,10]  # [64,64,64,26]
    # Max G-pool.      # [275,64,64,10] # [275,64,64,26]

    # D8 CNN             O(2)-MNIST       O(2)-EMNIST
    # G-TC              # [64,64,64,10]   # [50,64,64,26]
    # Sel. G-bispect.   # [500,64,64,10]  # [32,64,64,10]
    # Max G-pool.       # [1850,64,64,10] # [350,64,64,26]

    # Refer to the matainge et al 2024 paper:
    # Efficient, Complete G-Invariance for G-Equivariant Networks
    # via Algorithmic Reduction
    def spec_to_fc_size(spec):
        fc_sizes = {
            "cyclic": {
                "MNIST": {
                    "tc": [64, 64, 64, 10],
                    "bsp": [64, 64, 64, 10],
                    "max": [275, 64, 64, 10],
                },
                "EMNIST": {
                    "tc": [64, 64, 64, 26],
                    "bsp": [64, 64, 64, 26],
                    "max": [275, 64, 64, 26],
                },
            },
            "dihedral": {
                "MNIST": {
                    "tc": [64, 64, 64, 10],
                    "bsp": [500, 64, 64, 10],
                    "max": [1850, 64, 64, 10],
                },
                "EMNIST": {
                    "tc": [50, 64, 64, 26],
                    # TODO: Update the paper, last layer is inconsistent with this.
                    "bsp": [64, 64, 64, 26],
                    "max": [350, 64, 64, 26],
                },
            },
        }
        config = spec.config["train_loop_config"]
        return fc_sizes[config["group"]][config["dataset_name"]][config["pooling"]]

    def data_augmentation(spec):
        data_augmentation_map = {"cyclic": "sO2", "dihedral": "o2"}
        config = spec.config["train_loop_config"]
        return data_augmentation_map[config["group"]]

    def n_filters(spec):
        n_filters_map = {
            "cyclic": {"MNIST": 24, "EMNIST": 24},
            "dihedral": {"MNIST": 4, "EMNIST": 20},
        }
        config = spec.config["train_loop_config"]
        return n_filters_map[config["group"]]

    search_space = {
        "pooling": tune.grid_search(["bsp", "tc", "max"]),
        "dataset_name": tune.grid_search(["MNIST", "EMNIST"]),
        "group": tune.grid_search(["cyclic", "dihedral"]),
        # The next parameters depend on the previous ones.
        "fc_sizes": tune.sample_from(spec_to_fc_size),
        "data_augmentation": tune.sample_from(data_augmentation),
        "n_filters": tune.sample_from(n_filters),
    }
    scheduler = ASHAScheduler(max_t=100, grace_period=5, reduction_factor=2)

    def trial_str_creator(trial):
        config = trial.config["train_loop_config"]
        return f"pooling={config['pooling']},ds={config['dataset_name']},group={config['group']}"

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            trial_name_creator=trial_str_creator,
        ),
    )
    return tuner.fit()


if __name__ == "__main__":

    tune_mnist_asha(num_samples=100)
