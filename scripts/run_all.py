import itertools
import gtc.utils as gtc_utils
from gtc.utils import run_trainer
from gtc import configs

DATASETS_PATH = 'datasets/'

def run_experiments():
    n_filters = [
        2,
        4,
        6,
        8,
        24,
    ]
    groups_continuous = ["o2", "so2"]
    groups = ["c8", "d16"]  # TODO: Complete code for the dn groups. "c32", "c64","d4", "d32"
    poolings = ["max", "tc", "bsp"]  # fbsp is full bsp, bsp is partial bsp TODO: Implement "fbsp",
    datasets = ["mnist"]  # , "emnist"]

    # Dataset
    for group_continuous, dataset in itertools.product(groups_continuous, datasets):
        print(f"generating dataset for {group_continuous} {dataset}...")
        # Is the seed necessary for datasets?
        dataset_config = configs.get_dataset_config(group_continuous, dataset, seed=42)
        # TODO: Make this work with joblib
        gtc_utils.create_dataset(config=dataset_config, prefix=DATASETS_PATH)

    # Train
    for group, group_continuous, pooling, dataset, filters in itertools.product(
        groups, groups_continuous, poolings, datasets, n_filters
    ):
        print(f"running experiment for: group: {group} contiguous: {group_continuous} pooling: {pooling} dataset: {dataset} filters: {filters}...")
        config = configs.get_trainer_config(group_continuous, group, pooling, dataset, filters)
        run_trainer(
            logger_config=configs.logger_config,
            master_config=config,
            n_examples=100000,
            entity="johan-atmo",
            project="bispectrumnn",
            prefix=DATASETS_PATH
        )


run_experiments()
