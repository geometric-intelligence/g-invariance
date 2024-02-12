from gtc.utils import Config
from gtc.utils import TrainValLoader

"""
DATA_LOADER
"""

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
