from torch.optim import Adam
from gtc.utils import Config

"""
OPTIMIZER
"""
optimizer_config = Config({"type": Adam, "params": {"lr": 0.001}})
