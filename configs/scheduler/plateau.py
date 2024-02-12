from torch.optim.lr_scheduler import ReduceLROnPlateau
from gtc.utils import Config

"""
SCHEDULER
"""
scheduler_config = Config(
    {
        "type": ReduceLROnPlateau,
        "params": {"factor": 0.5, "patience": 2, "min_lr": 1e-4},
    }
)
