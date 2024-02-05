from collections import OrderedDict

from escnn import gspaces, nn
from torch_tools.config import Config

from gtc.modules import FullyConnectedBlock, GonR2ConvBlock, GTtoT, Linear, Ravel
from gtc.pooling import GroupPooling

N = 16


"""
CONV 1
"""

conv1 = Config(
    {
        "type": GonR2ConvBlock,
        "params": {
            "N": N,
            "action": gspaces.flipRot2dOnR2,
            "n_channels": 4,
            "kernel_size": 16,
            "padding": 0,
            "bias": False,
        },
    }
)


"""
GROUP POOL
"""

gpool = Config(
    {
        "type": GroupPooling,
        "params": {},
    }
)


"""
GT to T
"""

gttot = Config(
    {
        "type": GTtoT,
        "params": {},
    }
)


"""
RAVEL
"""

ravel = Config(
    {
        "type": Ravel,
        "params": {},
    }
)


"""
FC1
"""

FC1 = Config({"type": FullyConnectedBlock, "params": {"out_dim": 1850}})


"""
FC2
"""

FC2 = Config({"type": FullyConnectedBlock, "params": {"out_dim": 64}})


"""
FC3
"""

FC3 = Config({"type": FullyConnectedBlock, "params": {"out_dim": 64}})


"""
LINEAR
"""
linear = Config({"type": Linear, "params": {"out_dim": 10}})


"""
MODEL CONFIG
"""

model_config = OrderedDict(
    {
        "conv1": conv1,
        "gpool": gpool,
        "gttot": gttot,
        "ravel": ravel,
        "FC1": FC1,
        "FC2": FC2,
        "FC3": FC3,
        "linear": linear,
    }
)
