"""
Implements the paper: Efficient, Complete G-Invariance for G-Equivariant Networks
via Algorithmic Reduction, Mataigne et al, 2024.
"""

import pytorch_lightning as pl
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train import lightning as ray_lightning
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

import g_invariance.train as g_train

MAX_SEED = 65535


def train_func(config):
    """
    Trains a PyTorch model using the provided configuration.

    Args:
        config (dict): Configuration parameters for training.

    Returns:
        None
    """
    config_param = g_train.read_config_from_file()
    config_param.update(config)
    dm = g_train.MNISTDataModule(config_param)
    model = g_train.MNISTClassifier(
        config_param, wandb_setup_callback=setup_wandb, extra_config=config
    )

    pl.seed_everything(config_param.seed)
    trainer = pl.Trainer(
        devices='auto',
        accelerator='auto',
        strategy=ray_lightning.RayDDPStrategy(),
        callbacks=[ray_lightning.RayTrainReportCallback()],
        plugins=[ray_lightning.RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs=config_param.max_epochs,
    )
    trainer = ray_lightning.prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


def search_pooling():
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={'CPU': 3, 'GPU': 1}
    )

    run_config = RunConfig(
        callbacks=[WandbLoggerCallback(project='g-invariance')],
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute='ptl/val_accuracy',
            checkpoint_score_order='max',
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    def spec_to_target_params(spec):
        fc_sizes = {
            'cyclic': {
                'MNIST': 35000,
                'EMNIST': 35000,
            },
            'dihedral': {
                'MNIST': 140000,
                'EMNIST': 42000,
            },
        }
        config = spec['train_loop_config']
        return fc_sizes[config['group']][config['dataset_name']]

    def data_augmentation(spec):
        data_augmentation_map = {'cyclic': 'so2', 'dihedral': 'o2'}
        config = spec['train_loop_config']
        return data_augmentation_map[config['group']]

    def n_filters(spec):
        n_filters_map = {
            'cyclic': {'MNIST': 24, 'EMNIST': 24},
            'dihedral': {'MNIST': 4, 'EMNIST': 20},
        }
        config = spec['train_loop_config']
        return n_filters_map[config['group']][config['dataset_name']]

    def fc_sizes(spec):
        fc_sizes_map = {'MNIST': [64, 64, 10], 'EMNIST': [64, 64, 26]}
        config = spec['train_loop_config']
        return fc_sizes_map[config['dataset_name']]

    search_space = {
        'pooling': tune.grid_search(['bsp', 'tc', 'max']),
        'dataset_name': tune.grid_search(['MNIST', 'EMNIST']),
        'group': tune.grid_search(['cyclic', 'dihedral']),
        'img_size': tune.grid_search([16, 28]),
        'target_params_count': tune.sample_from(spec_to_target_params),
        'data_augmentation': tune.sample_from(data_augmentation),
        'fc_sizes': tune.sample_from(fc_sizes),
        'n_filters': tune.sample_from(n_filters),
        'seed': tune.randint(0, MAX_SEED),
    }

    def trial_str_creator(trial):
        config = trial.config['train_loop_config']
        seed = config['seed']
        return f"pooling={config['pooling']},ds={config['dataset_name']},group={config['group']},seed={seed}"

    tuner = tune.Tuner(
        ray_trainer,
        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            metric='ptl/val_accuracy',
            mode='max',
            # TODO: Infer num samples from search space.
            num_samples=10,
            trial_name_creator=trial_str_creator,
        ),
    )
    return tuner.fit()


if __name__ == '__main__':
    results = search_pooling()
    results.get_dataframe().to_csv('ray_results.csv')
