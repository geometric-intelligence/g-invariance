import torch
from gtc.utils import Config

loss_config = Config({"type": torch.nn.CrossEntropyLoss, "params": {}})
