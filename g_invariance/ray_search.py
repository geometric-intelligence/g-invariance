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

import g_invariance.train as g_invariance_train


def train_func(config):
    """
    Trains a PyTorch model using the provided configuration.

    Args:
        config (dict): Configuration parameters for training.

    Returns:
        None
    """
    config_param = g_invariance_train.read_config_from_file()
    setup_wandb(config)
    config_param.update(config)
    print(config_param)
    dm = g_invariance_train.MNISTDataModule(config_param)
    model = g_invariance_train.MNISTClassifier(config_param)

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
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 12, "GPU": 1}
    )

    run_config = RunConfig(
        callbacks=[WandbLoggerCallback(project="g_invariance_mnist")],
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
    search_space = {
        "pooling": tune.grid_search(["tc"]),  # "bsp", "tc", "max"]),
        "dataset": tune.grid_search(["mnist"]),
        "group": tune.grid_search(["c8", "c32", "c64", "d4", "d16", "d32"]),
    }
    scheduler = ASHAScheduler(max_t=100, grace_period=5, reduction_factor=2)

    def trial_str_creator(trial):
        config = trial.config["train_loop_config"]
        return f"pooling={config['pooling']},ds={config['dataset']},group={config['group']}"

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
