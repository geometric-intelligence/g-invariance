from torch.optim import Adam
from torch_tools.config import Config

"""
OPTIMIZER
"""
optimizer_config = Config({"type": Adam, "params": {"lr": 5e-5, "weight_decay": 1e-5}})
