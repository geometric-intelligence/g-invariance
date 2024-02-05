from torch_tools.config import Config
from transform_datasets.patterns.natural import MNIST
from transform_datasets.transforms import O2, AddChannelDim, CircleCrop, Resize

"""
DATASET
"""

pattern_config = Config(
    {
        "type": MNIST,
        "params": {"path": "datasets/emnist/emnist_letters_train.csv"},  # WAs /datasets
    }
)


transforms_config = {
    "0": Config(
        {
            "type": O2,
            "params": {"sample_method": "random"},
        }
    ),
    "1": Config({"type": Resize, "params": {"new_size": (16, 16)}}),
    "2": Config({"type": CircleCrop, "params": {}}),
    "3": Config({"type": AddChannelDim, "params": {}}),
}


dataset_config = {"pattern": pattern_config, "transforms": transforms_config, "seed": 2}
