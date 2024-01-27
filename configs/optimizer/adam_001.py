from torch.optim import Adam
from torch_tools.config import Config

"""
OPTIMIZER
"""
optimizer_config = Config({"type": Adam, "params": {"lr": 0.001}})
