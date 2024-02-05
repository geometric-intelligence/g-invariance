import argparse
import copy
import subprocess as sp

import wandb
import itertools

n_filters =[2, 4, 6, 8, 24]  # TODO: Allow the n_filter hyperparameter (number of filters in the GConv) to be passed to the training
groups_continuous = ["o2", "so2"]
groups = ["c8", "c32", "c64", "d4", "d16", "d32"]  # TODO: Complete code for the dn groups.
poolings = ["max", "tc", "fbsp", "bsp"]  #fbsp is full bsp, bsp is partial bsp
#datasets = ["10", "26"] #10 for mnist, 26 for emnist
datasets = ["mnist", "emnist"]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_agents", type=int, help="number of parallel agents to run", default=2)
parser.add_argument(
    "--devices", nargs="+", help="list of devices to run on", default=[0, 1]
)
args = parser.parse_args()

# Dataset
commands = []
for group, group_continuous, dataset in itertools.product(groups,groups_continuous, datasets):
    config = group_continuous + dataset + "_" + group + "_maxpool"
    command = "python scripts/run_data_agent.py --config {} --project {} --entity {}".format(
        config,
        "bispectrumnn",
        "simonmataigne",
    )
    commands.append(command)

processes = [sp.Popen(command, shell=True) for command in commands]
for p in processes:
    p.wait()

# Train
commands = []
for group, group_continuous, pooling, dataset in itertools.product(groups, poolings, datasets):
    config = group_continuous + dataset + "_" + group + "_" + pooling + "pool"
    command = "python scripts/run_train_agent.py --config {} --project {} --entity {}".format(
        config,
        "bispectrumnn",
        "simonmataigne",
    )
    commands.append(command)

    processes = [sp.Popen(command, shell=True) for command in commands]
    for p in processes:
        p.wait()
