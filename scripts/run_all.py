import itertools
from gtc.utils import create_dataset_run
from gtc.utils import run_trainer
from gtc import configs


def run_experiments():
    n_filters = [
        2,
        4,
        6,
        8,
        24,
    ]
    groups_continuous = ["o2", "so2"]
    groups = [
        "c8",
        "d16",
    ]  # TODO: Complete code for the dn groups. "c32", "c64","d4", "d32"
    poolings = [
        "max",
        "tc",
        "bsp",
    ]  # fbsp is full bsp, bsp is partial bsp TODO: Implement "fbsp",
    datasets = ["mnist", "emnist"]

    # Dataset
    for group_continuous, dataset in itertools.product(groups_continuous, datasets):
        print(f"generating dataset for {group_continuous} {dataset}...")
        # Is the seed necessary for datasets?
        dataset_config = configs.dataset(group_continuous, dataset, seed=42)
        # TODO: Make this work with joblib
        create_dataset_run(
            dataset_config=dataset_config,
            data_project="bispectrumnn",
            entity="johan-atmo",
        )

    # Train
    for group, group_continuous, dataset in itertools.product(
        groups, poolings, datasets
    ):
        config = configs.trainer(group_continuous, group, "maxpool", dataset, n_filters)
        run_trainer(
            master_config=config,
            n_examples=100000000,
            entity="johan-atmo",
            project="bispectrumnn",
            config=config,  # Nina.
        )


run_experiments()
