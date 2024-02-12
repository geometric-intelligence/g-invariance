from torch.optim import Adam
from gtc.utils import Config

"""
OPTIMIZER
"""
optimizer_config = Config({"type": Adam, "params": {"lr": 5e-5, "weight_decay": 1e-5}})
