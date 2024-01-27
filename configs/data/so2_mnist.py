from torch_tools.config import Config
from transform_datasets.patterns.natural import MNIST
from transform_datasets.transforms import (
    SO2,
    AddChannelDim,
    CenterMean,
    CircleCrop,
    UnitStd,
)

"""
DATASET
"""

pattern_config = Config(
    {
        "type": MNIST,
        "params": {"path": "datasets/mnist/mnist_train.csv"},
    }
)


transforms_config = {
    "0": Config(
        {
            "type": SO2,
            "params": {"fraction_transforms": 1 / 360, "sample_method": "random"},
        }
    ),
    "1": Config({"type": CircleCrop, "params": {}}),
    "2": Config({"type": AddChannelDim, "params": {}}),
}


dataset_config = {"pattern": pattern_config, "transforms": transforms_config, "seed": 0}
